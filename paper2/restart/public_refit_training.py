"""Train-only fixed-budget refit of the R12 equal-client control. No test input."""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from paper2.federated import clone_state
from paper2.restart.control_convergence import normalized_targets
from paper2.restart.diagnose_public_control_fit import matched_fit
from paper2.restart.work_scaled_regression_training import make_regressor, train_regression_client
from paper2.restart.work_scaled_simulation_training import control_average
from paper2.restart.validation_search import read, sha, identity, atomic_json, atomic_checkpoint

SETTINGS = {'hidden': 32, 'batch_size': 32, 'local_steps': 4, 'gradient_clip_norm': 1.,
            'aggregation': 'equal_client_parameter_average', 'optimizer': 'plain_SGD_no_momentum_no_weight_decay',
            'noise_enabled': False, 'knowledge_enabled': False, 'ranking_enabled': False}


def fit_receipt(folder, run, token, signature):
    return {'identity': token, 'signature': signature, 'run': run, 'rounds': run['rounds'],
            'model_representation': 'normalized_target; predict_raw multiplies by target_scale exactly once',
            'checkpoint_choice': 'last fixed inner-selected refit round', 'training_only': True,
            'assets': {n: sha(Path(folder) / n) for n in ('model.pt', 'checkpoint.pt')}}


def validate_run(run):
    if run['method'] != 'fixed_public_equal_client_control':
        raise ValueError('Unsupported refit method')
    if type(run['seed']) is not int or run['seed'] < 0 or type(run['rounds']) is not int or not 1 <= run['rounds'] <= 240:
        raise ValueError('Invalid fixed refit seed/budget')
    for name in ('learning_rate', 'target_scale'):
        if type(run[name]) not in (int, float) or not math.isfinite(run[name]) or run[name] <= 0:
            raise ValueError('Positive finite refit settings required')


def load_training_fold(data, run):
    """Never index outer-test arrays; checks bind all split metadata/assets."""
    data = Path(data)
    m = read(data / 'manifest.json')
    folder = data / 'folds' / run['fold']
    if any(sha(folder / n) != h for n, h in m['assets'][run['fold']].items()):
        raise ValueError('Refit fold asset changed')
    fold = read(folder / 'partition.json')
    if fold['kind'] != 'outer_refit' or fold['validation'] or set(fold['train']) & set(fold['test']):
        raise ValueError('Outer training isolation changed')
    if read(folder / 'preprocessing.json')['identity'] != fold['preprocessing_identity']:
        raise ValueError('Preprocessing identity differs')
    with np.load(folder / 'arrays.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'x_train', 'y_train', 'ids_train', 'x_test', 'y_test', 'ids_test'}:
            raise ValueError('Unexpected outer asset schema')
        arrays = {n: saved[n].copy() for n in ('x_train', 'y_train', 'ids_train')}
    np.testing.assert_array_equal(arrays['ids_train'], fold['train'])
    return fold, arrays


def predict_raw(model, x, scale):
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite prediction scale required')
    if x.dtype != np.float32 or x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite float32 prediction matrix required')
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(x))[:, 0].numpy() * scale
    if not np.isfinite(p).all():
        raise FloatingPointError('Refit prediction overflow')
    return p


def checked_checkpoint(saved, run, signature, groups, x, y):
    if saved['signature'] != signature or type(saved['done']) is not int or not 1 <= saved['done'] <= run['rounds']:
        raise ValueError('Refit checkpoint identity/budget changed')
    if [h['round'] for h in saved['history']] != list(range(1, saved['done'] + 1)):
        raise ValueError('Refit checkpoint history changed')
    for h in saved['history']:
        values = h['training_equal_client_fit']
        if h['local_steps'] != len(groups) * SETTINGS['local_steps'] or any(not math.isfinite(values[n]) for n in ('equal_client_model_mse', 'equal_client_constant_mse', 'equal_client_optimal_constant')) or values['equal_client_model_mse'] < 0 or values['equal_client_constant_mse'] < 0 or values['worse_than_matched_constant'] != (values['equal_client_model_mse'] > values['equal_client_constant_mse']):
            raise ValueError('Invalid refit checkpoint work/metrics')
    model = make_regressor(x.shape[1], SETTINGS['hidden'])
    model.load_state_dict(saved['state'])
    if any(not torch.isfinite(v).all() for v in saved['state'].values()):
        raise ValueError('Nonfinite refit checkpoint')
    fit = matched_fit(y, predict_raw(model, x, run['target_scale']), groups)
    if fit != saved['history'][-1]['training_equal_client_fit']:
        raise ValueError('Refit checkpoint training metrics changed')
    return model


def fit(folder, run, settings, fold, x, y, token, stop_after=None):
    """Caller must own the eventual coordinator writer lock. Training tensors only."""
    validate_run(run)
    if settings != SETTINGS:
        raise ValueError('Refit optimizer/work must match R12')
    if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 2 or x.shape[1] < 1 or y.ndim != 1 or not len(y) or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid refit training arrays')
    if fold['kind'] != 'outer_refit' or fold['validation'] or len(fold['train']) != len(y) or len(set(fold['train'])) != len(y) or set(fold['train']) & set(fold['test']):
        raise ValueError('Invalid refit training isolation')
    lookup = {i: n for n, i in enumerate(fold['train'])}
    slots = tuple(sorted(fold['clients']))
    members = [i for slot in slots for i in fold['clients'][slot]]
    if not slots or any(not fold['clients'][s] for s in slots) or sorted(members) != sorted(lookup):
        raise ValueError('Clients must partition refit training rows')
    groups = [[lookup[i] for i in fold['clients'][s]] for s in slots]
    signature = identity({'token': token, 'run': run, 'settings': settings, 'fold': fold,
                          'x_shape': list(x.shape), 'x_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                          'y_sha256': hashlib.sha256(y.tobytes()).hexdigest()})
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'fit.json').exists():
        receipt = read(folder / 'fit.json')
        if receipt != fit_receipt(folder, run, token, signature):
            raise ValueError('Completed refit changed')
        return receipt
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(run['seed'])
    model = make_regressor(x.shape[1], settings['hidden'])
    history, done = [], 0
    if (folder / 'checkpoint.pt').exists():
        saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
        model = checked_checkpoint(saved, run, signature, groups, x, y)
        done, history = saved['done'], saved['history']
    tx, ty = torch.from_numpy(x), torch.from_numpy(normalized_targets(y, run['target_scale']))
    for r in range(done + 1, run['rounds'] + 1):
        clients = {}
        for n, (slot, ids) in enumerate(zip(slots, groups)):
            update = train_regression_client(model, tx[ids], ty[ids], slot, r, fold['preprocessing_identity'],
                run['seed'] + 1009 * r + n, learning_rate=run['learning_rate'], batch_size=settings['batch_size'])
            if update.steps != settings['local_steps']:
                raise ValueError('Public local-step budget changed')
            clients[slot] = update.state
        model.load_state_dict(control_average(clone_state(model), clients, slots, None))
        history.append({'round': r, 'local_steps': len(slots) * settings['local_steps'],
                        'training_equal_client_fit': matched_fit(y, predict_raw(model, x, run['target_scale']), groups)})
        atomic_checkpoint(folder / 'checkpoint.pt', {'signature': signature, 'done': r,
                          'history': history, 'state': clone_state(model)})
        if stop_after == r and r < run['rounds']:
            return None
    atomic_checkpoint(folder / 'model.pt', clone_state(model))
    receipt = fit_receipt(folder, run, token, signature)
    atomic_json(folder / 'fit.json', receipt)
    return receipt
