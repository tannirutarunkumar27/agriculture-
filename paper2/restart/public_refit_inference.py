"""Identity-bound once-only raw-unit outer inference; caller owns writer lock."""
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch

from paper2.restart import public_refit_training as training
from paper2.restart.validation_search import read, sha, identity, atomic_json
from paper2.restart.verify_outer_study import scores

ASSETS = {'fit.json', 'model.pt', 'checkpoint.pt', 'partition_context.json',
          'inference_started.json', 'test_cache.npz', 'result.json'}


def training_signature(run, fold, arrays, token):
    x, y = arrays['x_train'], arrays['y_train']
    return identity({'token': token, 'run': run, 'settings': training.SETTINGS, 'fold': fold,
                          'x_shape': list(x.shape), 'x_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                          'y_sha256': hashlib.sha256(y.tobytes()).hexdigest()})


def verify_checkpoint(folder, run, fold, arrays, token, complete=False):
    checkpoint = torch.load(Path(folder) / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if checkpoint['signature'] != training_signature(run, fold, arrays, token) or type(checkpoint['done']) is not int or not 1 <= checkpoint['done'] <= run['rounds'] or (complete and checkpoint['done'] != run['rounds']):
        raise ValueError('Fixed refit round checkpoint differs')
    if [h['round'] for h in checkpoint['history']] != list(range(1, checkpoint['done'] + 1)):
        raise ValueError('Refit round history differs')
    for h in checkpoint['history']:
        fit = h['training_equal_client_fit']
        if h['local_steps'] != len(fold['clients']) * 4 or any(not math.isfinite(fit[n]) for n in ('equal_client_model_mse', 'equal_client_constant_mse', 'equal_client_optimal_constant')) or fit['equal_client_model_mse'] < 0 or fit['equal_client_constant_mse'] < 0 or fit['worse_than_matched_constant'] != (fit['equal_client_model_mse'] > fit['equal_client_constant_mse']):
            raise ValueError('Invalid saved training work/metrics')
    shapes = {'0.weight': (32, arrays['x_train'].shape[1]), '0.bias': (32,), '2.weight': (1, 32), '2.bias': (1,)}
    if set(checkpoint['state']) != set(shapes): raise ValueError('Checkpoint model keys differ')
    for k, shape in shapes.items():
        v = checkpoint['state'][k]
        if not isinstance(v, torch.Tensor) or v.shape != shape or v.dtype != torch.float32 or v.device.type != 'cpu' or not torch.isfinite(v).all():
            raise ValueError('Invalid checkpoint tensor schema')
    return checkpoint


def verify_fit(folder, run, fold, arrays, token):
    """Verify trained assets/rounds against input bytes without model forward calls."""
    folder = Path(folder); x = arrays['x_train']
    receipt = read(folder / 'fit.json')
    if receipt != training.fit_receipt(folder, run, token, training_signature(run, fold, arrays, token)):
        raise ValueError('Refit assets/signature/units changed')
    checkpoint = verify_checkpoint(folder, run, fold, arrays, token, True)
    state = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    shapes = {'0.weight': (32, x.shape[1]), '0.bias': (32,), '2.weight': (1, 32), '2.bias': (1,)}
    if set(state) != set(shapes) or set(checkpoint['state']) != set(shapes):
        raise ValueError('Refit model keys differ')
    for k, shape in shapes.items():
        v, saved_v = state[k], checkpoint['state'][k]
        if not isinstance(v, torch.Tensor) or not isinstance(saved_v, torch.Tensor) or v.shape != shape or saved_v.shape != shape or v.dtype != torch.float32 or saved_v.dtype != torch.float32 or v.device.type != 'cpu' or saved_v.device.type != 'cpu' or not torch.isfinite(v).all() or not torch.equal(v, saved_v):
            raise ValueError('Final refit model does not match checkpoint')
    return receipt


def context_for(fold, run):
    return {'fold': run['fold'], 'test_group': fold['test_group'], 'ids_test': fold['test'],
            'support': fold['support']['test'], 'preprocessing_identity': fold['preprocessing_identity']}


def load_test(data, run, fold):
    data = Path(data)
    hashes = read(data / 'manifest.json')['assets'][run['fold']]
    folder = data / 'folds' / run['fold']
    if any(sha(folder / n) != h for n, h in hashes.items()):
        raise ValueError('Evaluation fold changed')
    with np.load(folder / 'arrays.npz', allow_pickle=False) as saved:
        arrays = {n: saved[n].copy() for n in ('x_test', 'y_test', 'ids_test')}
    np.testing.assert_array_equal(arrays['ids_test'], fold['test'])
    return arrays


def evaluation_identity(run, context, arrays, features):
    x, y, ids = (arrays[n] for n in ('x_test', 'y_test', 'ids_test'))
    if x.dtype != np.float32 or y.dtype != np.float32 or ids.dtype != np.int64 or x.ndim != 2 or y.ndim != 1 or ids.ndim != 1 or not len(y) or x.shape != (len(y), features) or len(ids) != len(y) or len(set(ids.tolist())) != len(ids) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned evaluation schema required')
    if context['fold'] != run['fold'] or context['ids_test'] != ids.tolist() or len(context['support']) != len(ids):
        raise ValueError('Evaluation context/IDs differ')
    if any(type(row[n]) is not bool for row in context['support'] for n in ('all_species_seen', 'composition_seen')):
        raise ValueError('Boolean support panel flags required')
    return identity({'run': run, 'context': context, 'x_shape': list(x.shape),
                     'arrays_sha256': {n: hashlib.sha256(arrays[n].tobytes()).hexdigest()
                                       for n in ('x_test', 'y_test', 'ids_test')}})


def measured_panels(y, p, context):
    panels = {}
    for name in ('all_species_seen', 'composition_seen'):
        mask = np.array([r[name] for r in context['support']], dtype=bool)
        panels[name] = {'seen': scores(y[mask], p[mask]) if mask.any() else None,
                        'unseen': scores(y[~mask], p[~mask]) if (~mask).any() else None}
    return panels


def intent(token, fitted, run, input_identity):
    return {'identity': token, 'fit_signature': fitted['signature'], 'model_sha256': fitted['assets']['model.pt'],
            'input_identity': input_identity, 'target_scale': run['target_scale'], 'output_units': 't/ha'}


def evaluate_once(folder, run, fold, train_arrays, test_arrays, token):
    folder = Path(folder)
    if (folder / 'completion.json').exists():
        return verify_completed(folder, run, fold, train_arrays, test_arrays, token)
    if (folder / 'inference_started.json').exists():
        raise RuntimeError('Ambiguous prior outer inference; automatic repeat forbidden')
    fitted = verify_fit(folder, run, fold, train_arrays, token)
    context = context_for(fold, run)
    if read(folder / 'partition_context.json') != context:
        raise ValueError('Saved support context changed')
    input_id = evaluation_identity(run, context, test_arrays, train_arrays['x_train'].shape[1])
    atomic_json(folder / 'inference_started.json', intent(token, fitted, run, input_id))
    # Intent is committed before model loading/prediction. Never clear it to retry.
    model = training.make_regressor(train_arrays['x_train'].shape[1], 32)
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    prediction = training.predict_raw(model, test_arrays['x_test'], run['target_scale'])
    y = test_arrays['y_test']
    result = {'identity': token, 'run': run, 'input_identity': input_id,
              'test': scores(y, prediction), 'panels': measured_panels(y, prediction, context)}
    with (folder / 'test_cache.tmp').open('wb') as stream:
        np.savez_compressed(stream, y=y, prediction=prediction, ids=test_arrays['ids_test'])
    os.replace(folder / 'test_cache.tmp', folder / 'test_cache.npz')
    atomic_json(folder / 'result.json', result)
    atomic_json(folder / 'completion.json', {'identity': token, 'assets': {n: sha(folder / n) for n in sorted(ASSETS)}})
    return verify_completed(folder, run, fold, train_arrays, test_arrays, token)


def verify_completed(folder, run, fold, train_arrays, test_arrays, token):
    """No model forward calls: verify fixed refit, context and cached raw predictions."""
    folder = Path(folder)
    receipt = read(folder / 'completion.json')
    if receipt['identity'] != token or set(receipt['assets']) != ASSETS or any(sha(folder / n) != h for n, h in receipt['assets'].items()):
        raise ValueError('Outer completion assets changed')
    fitted = verify_fit(folder, run, fold, train_arrays, token)
    context = context_for(fold, run)
    input_id = evaluation_identity(run, context, test_arrays, train_arrays['x_train'].shape[1])
    if read(folder / 'partition_context.json') != context or read(folder / 'inference_started.json') != intent(token, fitted, run, input_id):
        raise ValueError('Outer inference/context binding differs')
    with np.load(folder / 'test_cache.npz', allow_pickle=False) as cache:
        if set(cache.files) != {'y', 'prediction', 'ids'} or cache['prediction'].dtype != np.float32:
            raise ValueError('Unexpected raw prediction cache schema')
        np.testing.assert_array_equal(cache['y'], test_arrays['y_test'])
        np.testing.assert_array_equal(cache['ids'], test_arrays['ids_test'])
        expected = {'identity': token, 'run': run, 'input_identity': input_id,
                    'test': scores(cache['y'], cache['prediction']),
                    'panels': measured_panels(cache['y'], cache['prediction'], context)}
    if read(folder / 'result.json') != expected:
        raise ValueError('Outer result/raw cached metrics differ')
    return expected
