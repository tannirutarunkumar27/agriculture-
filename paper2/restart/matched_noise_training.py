"""R14 fixed-work public simulation core. Caller must hold the study writer lock."""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from paper2.federated import clone_state
from paper2.restart import matched_noise_protocol as protocol
from paper2.restart import public_refit_training as refit
from paper2.restart.noisy_aggregation import aggregate_simulation
from paper2.restart.privacy_accounting import gaussian_budget
from paper2.restart.validation_search import read, sha, identity, atomic_json, atomic_checkpoint


def validate_run(run):
    parent = {k: run[k] for k in ('fold', 'seed', 'rounds', 'target_scale', 'learning_rate')}
    parent.update(id=run['reference_id'], method='fixed_public_equal_client_control')
    refit.validate_run(parent)
    variants = {'clipped_average': None, 'noisy_sigma2': 2., 'noisy_sigma4': 4., 'noisy_sigma8': 8.}
    expected_keys = set(parent) | {'reference_id', 'variant', 'clip_norm', 'noise_multiplier', 'public_slots'}
    if set(run) != expected_keys or run['method'] != 'fixed_public_matched_noise_simulation' or run['variant'] not in variants:
        raise ValueError('Unsupported matched run schema')
    if run['noise_multiplier'] != variants[run['variant']] or run['clip_norm'] != .04 or run['learning_rate'] != .01 or run['target_scale'] != 1.:
        raise ValueError('Matched clipping/noise/work settings differ')
    if run['seed'] not in protocol.refit.protocol.SEEDS or run['fold'] not in {f'outer_{i}' for i in range(8)}:
        raise ValueError('Unsupported matched fold/seed')
    if run['id'] != f"{run['fold']}_{run['variant']}_seed{run['seed']}" or run['reference_id'] != f"{run['fold']}_equal_public_seed{run['seed']}":
        raise ValueError('Matched run/reference identity differs')
    slots = run['public_slots']
    if not isinstance(slots, list) or len(slots) != 7 or len(set(slots)) != 7 or not all(isinstance(s, str) and s for s in slots) or slots != sorted(slots):
        raise ValueError('Seven sorted public slots required')


def training_context(run, settings, fold, x, y, token):
    validate_run(run); protocol.require_simulation(settings)
    if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 2 or x.shape[1] < 1 or y.ndim != 1 or not len(y) or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite float32 training arrays required')
    if fold['id'] != run['fold'] or fold['kind'] != 'outer_refit' or fold['validation'] or len(fold['train']) != len(y) or len(set(fold['train'])) != len(y) or set(fold['train']) & set(fold['test']):
        raise ValueError('Training isolation differs')
    slots = run['public_slots']; lookup = {row: i for i, row in enumerate(fold['train'])}
    if sorted(fold['clients']) != slots or any(not fold['clients'][s] for s in slots):
        raise ValueError('Public client roster differs')
    members = [i for s in slots for i in fold['clients'][s]]
    if sorted(members) != sorted(lookup): raise ValueError('Client membership must partition training rows')
    groups = [[lookup[i] for i in fold['clients'][s]] for s in slots]
    signature = identity({'token': token, 'run': run, 'settings': settings, 'fold': fold,
                          'x_shape': list(x.shape), 'x_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                          'y_sha256': hashlib.sha256(y.tobytes()).hexdigest()})
    return signature, groups


def check_state(state, features):
    shapes = {'0.weight': (32, features), '0.bias': (32,), '2.weight': (1, 32), '2.bias': (1,)}
    if not isinstance(state, dict) or set(state) != set(shapes): raise ValueError('Model parameter keys differ')
    for name, shape in shapes.items():
        v = state[name]
        if not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or v.device.type != 'cpu' or v.shape != shape or not torch.isfinite(v).all():
            raise ValueError('Invalid model tensor schema')


def check_checkpoint(saved, run, signature, groups, x, y, *, complete=False, check_metrics=False):
    if set(saved) != {'signature', 'done', 'state', 'history', 'events'} or saved['signature'] != signature or type(saved['done']) is not int or not 1 <= saved['done'] <= run['rounds'] or (complete and saved['done'] != run['rounds']):
        raise ValueError('Checkpoint identity/budget differs')
    if saved['events'] != protocol.planned_events(run)[:saved['done']]:
        raise ValueError('Checkpoint noise ledger differs')
    if [h['round'] for h in saved['history']] != list(range(1, saved['done'] + 1)):
        raise ValueError('Checkpoint round history differs')
    for h in saved['history']:
        values = h['training_equal_client_fit']
        if h['local_steps'] != 28 or any(type(values[n]) not in (int, float) or not math.isfinite(values[n]) for n in ('equal_client_model_mse', 'equal_client_constant_mse', 'equal_client_optimal_constant')) or values['equal_client_model_mse'] < 0 or values['equal_client_constant_mse'] < 0 or values['worse_than_matched_constant'] != (values['equal_client_model_mse'] > values['equal_client_constant_mse']):
            raise ValueError('Checkpoint training work/metrics differ')
    check_state(saved['state'], x.shape[1])
    if check_metrics:
        model = refit.make_regressor(x.shape[1], 32); model.load_state_dict(saved['state'])
        measured = refit.matched_fit(y, refit.predict_raw(model, x, run['target_scale']), groups)
        if measured != saved['history'][-1]['training_equal_client_fit']:
            raise ValueError('Checkpoint training metrics do not match state')
    return saved


def receipt(folder, run, token, signature):
    return {'identity': token, 'signature': signature, 'run': run, 'rounds': run['rounds'],
            'model_representation': 'normalized_target; predict_raw restores units once',
            'training_only': True, 'simulation_only': True, 'application_privacy_certified': False,
            'assets': {n: sha(Path(folder) / n) for n in ('model.pt', 'checkpoint.pt', 'accounting.json')}}


def run_accounting(run, token):
    events = protocol.planned_events(run)
    return {'identity': token, 'events': events, 'rounds': run['rounds'],
            'conditional_budget': gaussian_budget([e['event'] for e in events], 1e-5) if events else None,
            'simulation_only': True, 'application_privacy_certified': False}


def verify_fit(folder, run, settings, fold, x, y, token):
    """No model forward: strict schema, final checkpoint, event and receipt verification."""
    folder = Path(folder); signature, groups = training_context(run, settings, fold, x, y, token)
    if read(folder / 'fit.json') != receipt(folder, run, token, signature): raise ValueError('Fit receipt/assets differ')
    saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    check_checkpoint(saved, run, signature, groups, x, y, complete=True)
    model = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True); check_state(model, x.shape[1])
    if any(not torch.equal(v, saved['state'][k]) for k, v in model.items()): raise ValueError('Final model differs from checkpoint')
    if read(folder / 'accounting.json') != run_accounting(run, token): raise ValueError('Final accounting differs')
    return read(folder / 'fit.json')


def noise_rng(run, round_number):
    stream = int.from_bytes(hashlib.sha256((protocol.NAMESPACE + run['id']).encode()).digest()[:8], 'big')
    return np.random.default_rng(np.random.SeedSequence([run['seed'], stream, round_number]))


def fit(folder, run, settings, fold, x, y, token, stop_after=None):
    signature, groups = training_context(run, settings, fold, x, y, token)
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= run['rounds']):
        raise ValueError('Invalid interruption round')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'fit.json').exists(): return verify_fit(folder, run, settings, fold, x, y, token)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(run['seed'])
    model = refit.make_regressor(x.shape[1], 32); done = 0; history = []; events = []
    if (folder / 'checkpoint.pt').exists():
        saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
        check_checkpoint(saved, run, signature, groups, x, y, check_metrics=True)
        model.load_state_dict(saved['state']); done, history, events = saved['done'], saved['history'], saved['events']
    tx, ty = torch.from_numpy(x), torch.from_numpy(refit.normalized_targets(y, run['target_scale']))
    expected = protocol.planned_events(run); slots = tuple(run['public_slots'])
    for r in range(done + 1, run['rounds'] + 1):
        clients = {}
        for n, (slot, ids) in enumerate(zip(slots, groups)):
            update = refit.train_regression_client(model, tx[ids], ty[ids], slot, r, fold['preprocessing_identity'],
                run['seed'] + 1009 * r + n, learning_rate=run['learning_rate'], batch_size=settings['batch_size'])
            if update.steps != 4: raise ValueError('Local optimizer work differs')
            clients[slot] = update.state
        if run['noise_multiplier'] is None:
            state = refit.control_average(clone_state(model), clients, slots, run['clip_norm'])
        else:
            state, event, _ = aggregate_simulation(clone_state(model), clients, slots, run['clip_norm'],
                                                   run['noise_multiplier'], noise_rng(run, r), simulation_only=True)
            if event != expected[r - 1]['event']: raise ValueError('Observed noise scale differs from frozen plan')
            events.append(expected[r - 1])
        check_state(state, x.shape[1]); model.load_state_dict(state)
        history.append({'round': r, 'local_steps': 28,
                        'training_equal_client_fit': refit.matched_fit(y, refit.predict_raw(model, x, run['target_scale']), groups)})
        atomic_checkpoint(folder / 'checkpoint.pt', {'signature': signature, 'done': r,
                          'state': clone_state(model), 'history': history, 'events': events})
        if stop_after == r and r < run['rounds']: return None
    atomic_checkpoint(folder / 'model.pt', clone_state(model))
    atomic_json(folder / 'accounting.json', run_accounting(run, token))
    atomic_json(folder / 'fit.json', receipt(folder, run, token, signature))
    return verify_fit(folder, run, settings, fold, x, y, token)
