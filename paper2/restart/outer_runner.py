"""Identity-bound, capped outer study coordinator; dry-run is the default."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import traceback
import numpy as np
from paper2.restart.validation_search import ROOT, read, sha, identity, runtime, atomic_json, writer_lock, verify_frozen
from paper2.restart.outer_protocol import make_plan
from paper2.restart.outer_training import fit, evaluate_once, verify_evaluation

CODE = [Path(__file__), ROOT/'paper2/restart/outer_training.py', ROOT/'paper2/restart/outer_protocol.py',
        ROOT/'paper2/restart/regression_training.py', ROOT/'paper2/restart/validation_search.py', ROOT/'paper2/federated.py']


def verify_plan(package):
    package = Path(package).resolve()
    manifest = read(package/'manifest.json')
    if any(sha(package/name) != expected for name, expected in manifest['artifact_sha256'].items()):
        raise ValueError('Outer plan artifact changed')
    if runtime() != manifest['runtime']:
        raise ValueError('Runtime changed')
    inner, _, _ = verify_frozen(ROOT/'paper2/results/restart_validation_protocol_v1')
    if sha(ROOT/'paper2/results/restart_validation_protocol_v1/manifest.json') != manifest['search_manifest_sha256']:
        raise ValueError('Search manifest changed')
    data = Path(manifest['data'])
    if str(data) != inner['data'] or sha(data/'manifest.json') != manifest['data_manifest_sha256']:
        raise ValueError('Outer data binding changed')
    config, plan = read(package/'config.json'), read(package/'plan.json')
    selected = read(package/'selected.json')
    if sha(package/'selected.json') != manifest['selection_sha256'] or plan != make_plan(selected, read(data/'manifest.json'), config['seeds']):
        raise ValueError('Outer selection/plan changed')
    return manifest, config, plan


def freeze_runner(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    manifest, _, plan = verify_plan(package)
    if output.exists():
        raise FileExistsError(output)
    root = ROOT/'paper2/results'
    if root not in output.parents or package in output.parents or output in package.parents:
        raise ValueError('Unsafe runner package output')
    output.mkdir(parents=True)
    (output/'source_snapshot').mkdir()
    for source in CODE:
        shutil.copyfile(source, output/'source_snapshot'/source.name)
    record = {'plan_package': str(package), 'plan_manifest_sha256': sha(package/'manifest.json'),
              'runtime': runtime(), 'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in CODE},
              'planned_runs': len(plan), 'status': 'FROZEN_EXECUTION_CODE_NO_OUTER_INFERENCE'}
    atomic_json(output/'manifest.json', record)
    return record['status']


def verify_runner(package):
    package = Path(package).resolve()
    record = read(package/'manifest.json')
    if runtime() != record['runtime'] or sha(Path(record['plan_package'])/'manifest.json') != record['plan_manifest_sha256']:
        raise ValueError('Execution binding changed')
    for relative, expected in record['code_sha256'].items():
        if sha(ROOT/relative) != expected or sha(package/'source_snapshot'/Path(relative).name) != expected:
            raise ValueError('Execution code changed')
    manifest, config, plan = verify_plan(record['plan_package'])
    return record, manifest, config, plan


def load_fold(data, run):
    prepared = read(data/'manifest.json')
    folder = data/'folds'/run['fold']
    for name, expected in prepared['assets'][run['fold']].items():
        if sha(folder/name) != expected:
            raise ValueError('Fold asset changed')
    fold = read(folder/'partition.json')
    if fold['kind'] != 'outer_refit' or fold['validation'] or set(fold['train']) & set(fold['test']):
        raise ValueError('Invalid outer isolation')
    if read(folder/'preprocessing.json')['identity'] != fold['preprocessing_identity']:
        raise ValueError('Preprocessing mismatch')
    with np.load(folder/'arrays.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'x_train','y_train','ids_train','x_test','y_test','ids_test'}:
            raise ValueError('Unexpected outer arrays')
        arrays = {name: saved[name].copy() for name in saved.files}
    for split in ('train', 'test'):
        np.testing.assert_array_equal(arrays['ids_'+split], fold[split])
        x, y = arrays['x_'+split], arrays['y_'+split]
        if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid outer arrays')
    if arrays['x_train'].shape[1] != arrays['x_test'].shape[1] or len(fold['support']['test']) != len(fold['test']):
        raise ValueError('Feature/support mismatch')
    return fold, arrays


def coordinate(plan, config, data, output, binding, max_runs, retry_failed=False):
    """Caller owns output writer lock; this function also supports synthetic tests."""
    done, executed = [], 0
    for run in plan:
        folder = output/'runs'/run['id']
        token = identity({'binding': binding, 'run': run})
        if (folder/'completion.json').exists():
            result = verify_evaluation(folder, token)
            if result['run'] != run:
                raise ValueError('Completed run configuration changed')
            fold = read(data/'folds'/run['fold']/'partition.json')
            context = {'fold': run['fold'], 'test_group': fold['test_group'],
                'ids_test': fold['test'], 'support': fold['support']['test'], 'preprocessing_identity': fold['preprocessing_identity']}
            if read(folder/'partition_context.json') != context:
                raise ValueError('Completed support context changed')
            with np.load(folder/'test_cache.npz', allow_pickle=False) as cached:
                np.testing.assert_array_equal(cached['ids'], fold['test'])
            done.append(run['id'])
            continue
        if executed >= max_runs:
            continue
        if (folder/'failure.json').exists() and not retry_failed:
            raise RuntimeError('Inspect failure and explicitly retry: '+run['id'])
        if (folder/'inference_started.json').exists():
            raise RuntimeError('Ambiguous inference cannot be automatically retried: '+run['id'])
        folder.mkdir(parents=True, exist_ok=True)
        attempts = read(folder/'attempts.json') if (folder/'attempts.json').exists() else []
        attempts.append({'started': datetime.now(timezone.utc).isoformat(), 'status': 'IN_PROGRESS'})
        atomic_json(folder/'attempts.json', attempts)
        executed += 1
        try:
            fold, arrays = load_fold(data, run)
            atomic_json(folder/'partition_context.json', {'fold': run['fold'], 'test_group': fold['test_group'],
                'ids_test': fold['test'], 'support': fold['support']['test'], 'preprocessing_identity': fold['preprocessing_identity']})
            fit(folder, run, config['neural_settings'], fold, arrays['x_train'], arrays['y_train'], token)
            evaluate_once(folder, run, config['neural_settings'], arrays['x_test'], arrays['y_test'], arrays['ids_test'], token)
            atomic_json(folder/'context_receipt.json', {'identity': token, 'sha256': sha(folder/'partition_context.json')})
            attempts[-1]['status'] = 'COMPLETE'
            done.append(run['id'])
        except Exception:
            attempts[-1].update(status='FAILED', traceback=traceback.format_exc())
            atomic_json(folder/'failure.json', attempts[-1])
            atomic_json(folder/'attempts.json', attempts)
            atomic_json(output/'progress.json', {'status': 'FAILED', 'completed': len(done), 'planned': len(plan), 'failed': [run['id']]})
            raise
        atomic_json(folder/'attempts.json', attempts)
    progress = {'status': 'COMPLETE' if len(done) == len(plan) else 'PARTIAL', 'completed': len(done),
                'planned': len(plan), 'pending': len(plan)-len(done), 'executed_this_invocation': executed}
    atomic_json(output/'progress.json', progress)
    return progress


def execute(package, output, max_runs=1, resume=False, retry_failed=False, dry_run=True):
    if type(max_runs) is not int or max_runs < 1:
        raise ValueError('Positive execution cap required')
    package, output = Path(package).resolve(), Path(output).resolve()
    record, manifest, config, plan = verify_runner(package)
    paths = (package, Path(record['plan_package']), Path(manifest['data']))
    if ROOT/'paper2/results' not in output.parents or any(output == p or p in output.parents or output in p.parents for p in paths):
        raise ValueError('Execution output must be separate')
    if dry_run:
        return {'status': 'DRY_RUN_NO_INFERENCE', 'planned': len(plan), 'max_runs': max_runs}
    binding = identity(record)
    if output.exists():
        if not resume:
            raise ValueError('Existing output requires explicit resume')
    else:
        output.mkdir(parents=True)
    with writer_lock(output):
        if (output/'binding.json').exists():
            if read(output/'binding.json')['identity'] != binding:
                raise ValueError('Output binding changed')
        elif resume or any(p.name != 'writer.lock' for p in output.iterdir()):
            raise ValueError('Unbound output; refusing recovery')
        else:
            atomic_json(output/'binding.json', {'identity': binding})
        return coordinate(plan, config, Path(manifest['data']), output, binding, max_runs, retry_failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT/'paper2/results/restart_outer_execution_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--max-runs', type=int, default=1)
    args = parser.parse_args()
    result = freeze_runner(ROOT/'paper2/results/restart_outer_protocol_v1', args.output) if args.freeze else execute(args.package, args.output, args.max_runs, args.resume, args.retry_failed, not args.execute)
    print(json.dumps(result, indent=2))
