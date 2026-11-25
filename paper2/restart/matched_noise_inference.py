"""R14 once-only inference with noise-aware fit validation and cached verification."""
import os
from pathlib import Path
import numpy as np
import torch

from paper2.restart import public_refit_inference as common
from paper2.restart import matched_noise_training as training
from paper2.restart import matched_noise_protocol as protocol
from paper2.restart.validation_search import read, sha, atomic_json

ASSETS = common.ASSETS | {'accounting.json'}
context_for = common.context_for
load_test = common.load_test


def verify_fit(folder, run, fold, arrays, token):
    return training.verify_fit(folder, run, protocol.config(), fold, arrays['x_train'], arrays['y_train'], token)


def verify_completed(folder, run, fold, train_arrays, test_arrays, token):
    folder = Path(folder); receipt = read(folder / 'completion.json')
    if receipt['identity'] != token or set(receipt['assets']) != ASSETS or any(sha(folder / n) != h for n, h in receipt['assets'].items()):
        raise ValueError('Matched completion assets differ')
    fitted = verify_fit(folder, run, fold, train_arrays, token); context = context_for(fold, run)
    input_id = common.evaluation_identity(run, context, test_arrays, train_arrays['x_train'].shape[1])
    if read(folder / 'partition_context.json') != context or read(folder / 'inference_started.json') != common.intent(token, fitted, run, input_id):
        raise ValueError('Matched inference/context binding differs')
    with np.load(folder / 'test_cache.npz', allow_pickle=False) as cache:
        if set(cache.files) != {'y', 'prediction', 'ids'} or cache['prediction'].dtype != np.float32:
            raise ValueError('Unexpected matched cache schema')
        np.testing.assert_array_equal(cache['y'], test_arrays['y_test'])
        np.testing.assert_array_equal(cache['ids'], test_arrays['ids_test'])
        expected = {'identity': token, 'run': run, 'input_identity': input_id,
                    'test': common.scores(cache['y'], cache['prediction']),
                    'panels': common.measured_panels(cache['y'], cache['prediction'], context),
                    'simulation_only': True, 'application_privacy_certified': False}
    if read(folder / 'result.json') != expected: raise ValueError('Matched cached metrics differ')
    return expected


def evaluate_once(folder, run, fold, train_arrays, test_arrays, token):
    folder = Path(folder)
    if (folder / 'completion.json').exists(): return verify_completed(folder, run, fold, train_arrays, test_arrays, token)
    if (folder / 'inference_started.json').exists(): raise RuntimeError('Ambiguous outer inference; automatic repeat forbidden')
    fitted = verify_fit(folder, run, fold, train_arrays, token); context = context_for(fold, run)
    if read(folder / 'partition_context.json') != context: raise ValueError('Saved matched context differs')
    input_id = common.evaluation_identity(run, context, test_arrays, train_arrays['x_train'].shape[1])
    atomic_json(folder / 'inference_started.json', common.intent(token, fitted, run, input_id))
    model = training.refit.make_regressor(train_arrays['x_train'].shape[1], 32)
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    prediction = training.refit.predict_raw(model, test_arrays['x_test'], run['target_scale'])
    y = test_arrays['y_test']
    result = {'identity': token, 'run': run, 'input_identity': input_id,
              'test': common.scores(y, prediction), 'panels': common.measured_panels(y, prediction, context),
              'simulation_only': True, 'application_privacy_certified': False}
    with (folder / 'test_cache.tmp').open('wb') as stream:
        np.savez_compressed(stream, y=y, prediction=prediction, ids=test_arrays['ids_test'])
    os.replace(folder / 'test_cache.tmp', folder / 'test_cache.npz')
    atomic_json(folder / 'result.json', result)
    atomic_json(folder / 'completion.json', {'identity': token, 'assets': {n: sha(folder / n) for n in sorted(ASSETS)}})
    return verify_completed(folder, run, fold, train_arrays, test_arrays, token)
