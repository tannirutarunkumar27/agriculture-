"""Bounded R12 inner-only coordinator. The previously frozen engine is unchanged."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import statistics
import traceback

import torch

from paper2.restart.control_convergence import candidate, load_inner
from paper2.restart.prepare_public_data import verify as verify_data, new_output
from paper2.restart.validation_search import ROOT, read, sha, identity, runtime, atomic_json, writer_lock

PROTOCOL = ROOT / 'paper2/results/restart_control_search_protocol_v1'
EXECUTION = ROOT / 'paper2/results/restart_control_search_execution_v1'
CODE = [Path(__file__), *[Path(__file__).with_name(n) for n in (
    'validation_search.py', 'prepare_public_data.py', 'public_preprocessing.py',
    'regression_training.py', 'noisy_aggregation.py', 'privacy_accounting.py',
    'verify_control_search.py')], ROOT / 'paper2/federated.py']


def make_plan(config, data_manifest):
    return [{'id': f'{fold}_scale{scale}_lr{lr}', 'fold': fold,
             'target_scale': scale, 'learning_rate': lr, 'rounds': config['rounds'],
             'report_rounds': config['report_rounds'], 'seed': config['seed']}
            for fold in sorted(n for n in data_manifest['assets'] if '_inner_' in n)
            for scale in config['target_scales'] for lr in config['learning_rates']]


def verify_protocol(package=PROTOCOL):
    package = Path(package).resolve()
    m = read(package / 'manifest.json')
    # Verify immutable package contents before volatile host metadata.  This
    # makes a tampered protocol diagnosable even when it is inspected from a
    # different (but otherwise compatible) runtime.
    for name, field in (('config.json', 'config_sha256'), ('plan.json', 'plan_sha256')):
        if sha(package / name) != m[field]:
            raise ValueError('Frozen protocol artifact changed')
    for name, expected in m['source_sha256'].items():
        if sha(Path(__file__).with_name(name)) != expected or sha(package / 'source_snapshot' / name) != expected:
            raise ValueError('Frozen candidate source changed')
    if runtime() != m['runtime']:
        raise ValueError('Protocol runtime changed')
    config, plan = read(package / 'config.json'), read(package / 'plan.json')
    if config != read(ROOT / 'paper2/configs/restart_control_search_v1.json'):
        raise ValueError('Unsupported R12 configuration')
    data = Path(m['data']).resolve()
    if verify_data(data)['manifest_sha256'] != m['data_manifest_sha256']:
        raise ValueError('Public data binding changed')
    if plan != make_plan(config, read(data / 'manifest.json')) or len(plan) != 224 or m['planned'] != 224:
        raise ValueError('Expanded plan differs from frozen grid')
    return m, config, plan, data


def freeze(output):
    _, _, plan, data = verify_protocol()
    output = new_output(output, data)
    safe_output(output, (PROTOCOL, data))
    output.mkdir(parents=True)
    (output / 'source_snapshot').mkdir()
    for p in CODE:
        shutil.copyfile(p, output / 'source_snapshot' / p.name)
    atomic_json(output / 'manifest.json', {
        'protocol': str(PROTOCOL), 'protocol_manifest_sha256': sha(PROTOCOL / 'manifest.json'),
        'source_sha256': {p.name: sha(p) for p in CODE}, 'runtime': runtime(),
        'planned': len(plan), 'mode': 'INNER_VALIDATION_DEVELOPMENT_ONLY',
        'outer_model_inference': False, 'privacy_claim': False})
    return {'status': 'FROZEN_EXECUTION_NO_TRAINING', 'planned': len(plan)}


def verify_execution(package):
    package = Path(package).resolve()
    m = read(package / 'manifest.json')
    if m['protocol'] != str(PROTOCOL) or sha(PROTOCOL / 'manifest.json') != m['protocol_manifest_sha256'] or runtime() != m['runtime']:
        raise ValueError('Execution protocol/runtime changed')
    if set(m['source_sha256']) != {p.name for p in CODE}:
        raise ValueError('Execution source inventory changed')
    for p in CODE:
        if sha(p) != m['source_sha256'][p.name] or sha(package / 'source_snapshot' / p.name) != sha(p):
            raise ValueError('Execution source changed')
    _, config, plan, data = verify_protocol()
    if m['planned'] != len(plan) or m['outer_model_inference'] is not False or m['privacy_claim'] is not False:
        raise ValueError('Execution scope changed')
    return m, config, plan, data


def safe_output(output, protected):
    output = Path(output).resolve()
    if ROOT / 'paper2/results' not in output.parents:
        raise ValueError('Output must be a separate results directory')
    for p in protected:
        p = Path(p).resolve()
        if output == p or p in output.parents or output in p.parents:
            raise ValueError('Output overlaps a protected package')
    return output


def validate_checkpoint(checkpoint, run, token, complete=False):
    if checkpoint['identity'] != token or type(checkpoint['done']) is not int:
        raise ValueError('Checkpoint identity/progress changed')
    done, history = checkpoint['done'], checkpoint['history']
    if not 1 <= done <= run['rounds'] or (complete and done != run['rounds']):
        raise ValueError('Checkpoint budget mismatch')
    if [h['round'] for h in history] != list(range(1, done + 1)):
        raise ValueError('Checkpoint round sequence changed')
    for h in history:
        v, fit = h['validation'], h['training_equal_client_fit']
        if type(v['samples']) is not int or v['samples'] < 1:
            raise ValueError('Invalid validation count')
        for name in ('rmse', 'mae'):
            if not math.isfinite(v[name]) or v[name] < 0:
                raise ValueError('Invalid saved validation metric')
        if v['r2'] is not None and not math.isfinite(v['r2']):
            raise ValueError('Nonfinite saved R2')
        for name in ('equal_client_model_mse', 'equal_client_constant_mse', 'equal_client_optimal_constant'):
            if not math.isfinite(fit[name]) or (name.endswith('mse') and fit[name] < 0):
                raise ValueError('Invalid saved training metric')
        if fit['worse_than_matched_constant'] != (fit['equal_client_model_mse'] > fit['equal_client_constant_mse']):
            raise ValueError('Training constant comparison changed')
    best = min(history, key=lambda h: h['validation']['rmse'])
    if checkpoint['best'] != best['validation']['rmse'] or checkpoint['best_round'] != best['round']:
        raise ValueError('Best checkpoint / first-tie rule changed')
    expected = {}
    for r in run['report_rounds']:
        if r <= done:
            selected = min(history[:r], key=lambda h: h['validation']['rmse'])
            expected[str(r)] = {'last': history[r-1], 'best_round': selected['round'],
                                'best_validation_rmse': selected['validation']['rmse']}
    if checkpoint['snapshots'] != expected:
        raise ValueError('Budget-prefix reports changed')
    for name in ('state', 'best_state'):
        state = checkpoint[name]
        if not state or any(not isinstance(v, torch.Tensor) or v.device.type != 'cpu' or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Invalid saved model state')
    return checkpoint


def completed(folder, run, token):
    folder = Path(folder)
    receipt = read(folder / 'completion.json')
    if receipt['identity'] != token or set(receipt['assets']) != {'result.json', 'checkpoint.pt'}:
        raise ValueError('Completion receipt changed')
    if any(sha(folder / n) != h for n, h in receipt['assets'].items()):
        raise ValueError('Completed assets changed')
    result = read(folder / 'result.json')
    saved = validate_checkpoint(torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True), run, token, True)
    if result != {'identity': token, 'run': run, 'snapshots': saved['snapshots'], 'history': saved['history'],
                  'outer_model_inference': False, 'privacy_claim': False}:
        raise ValueError('Completed result / checkpoint mismatch')
    return result


def select_complete(plan, results, config):
    """No pooled weighting, no partial selections, first exact configuration tie."""
    expected_ids = [r['id'] for r in plan]
    if len(set(expected_ids)) != len(plan) or set(results) != set(expected_ids):
        raise ValueError('Complete unique search required for selection')
    for run in plan:
        result = results[run['id']]
        if result['run'] != run or result['outer_model_inference'] is not False or result['privacy_claim'] is not False:
            raise ValueError('Selection candidate/scope mismatch')
        h = result['history']
        if [v['round'] for v in h] != list(range(1, run['rounds'] + 1)) or any(not math.isfinite(v['validation']['rmse']) or v['validation']['rmse'] < 0 for v in h):
            raise ValueError('Incomplete/nonfinite candidate curve')
    configs = [(s, lr) for s in config['target_scales'] for lr in config['learning_rates']]
    selections = []
    for outer in sorted({r['fold'].split('_inner_')[0] for r in plan}):
        rows = [r for r in plan if r['fold'].split('_inner_')[0] == outer]
        folds = sorted({r['fold'] for r in rows})
        if len(folds) != 7 or len(rows) != 7 * len(configs):
            raise ValueError('Seven inner groups per outer group required')
        options = []
        for scale, lr in configs:
            matches = [r for r in rows if (r['target_scale'], r['learning_rate']) == (scale, lr)]
            if sorted(r['fold'] for r in matches) != folds:
                raise ValueError('Configuration fold coverage changed')
            best = [min(results[r['id']]['history'], key=lambda h: h['validation']['rmse']) for r in matches]
            options.append({'target_scale': scale, 'learning_rate': lr,
                            'equal_inner_group_mean_best_validation_rmse': statistics.mean(h['validation']['rmse'] for h in best),
                            'best_rounds': [h['round'] for h in best], 'inner_folds': [r['fold'] for r in matches],
                            'round_cap_hits': sum(h['round'] == config['rounds'] for h in best)})
        chosen = min(options, key=lambda o: o['equal_inner_group_mean_best_validation_rmse'])
        selections.append({'outer_fold': outer, 'selected': chosen,
                           'refit_rounds': int(statistics.median(chosen['best_rounds'])), 'all_configurations': options,
                           'selected_lr_at_grid_edge': chosen['learning_rate'] in (min(config['learning_rates']), max(config['learning_rates'])),
                           'selected_scale_at_grid_edge': chosen['target_scale'] in (min(config['target_scales']), max(config['target_scales']))})
    return {'status': 'COMPLETE_INNER_SELECTION', 'candidates': len(plan), 'selections': selections,
            'outer_model_inference': False, 'privacy_claim': False,
            'interpretation': 'Fixed-budget development selection; grid edges and cap hits are not convergence proof. Prior outer outcomes exposed.'}


def execute(package, output, cap=1, resume=False, dry_run=True, retry_failed=False):
    if type(cap) is not int or cap < 1:
        raise ValueError('Positive integer cap required')
    package = Path(package).resolve()
    manifest, config, plan, data = verify_execution(package)
    output = safe_output(output, (package, PROTOCOL, data))
    if dry_run:
        return {'status': 'DRY_RUN', 'planned': len(plan), 'cap': cap}
    if output.exists() and not resume:
        raise ValueError('Explicit resume required')
    output.mkdir(parents=True, exist_ok=True)
    binding = identity(manifest)
    results, executed = {}, 0
    with writer_lock(output):
        if (output / 'binding.json').exists():
            if read(output / 'binding.json') != {'identity': binding}:
                raise ValueError('Output binding changed')
        elif resume or any(p.name != 'writer.lock' for p in output.iterdir()):
            raise ValueError('Unbound output cannot be resumed')
        else:
            atomic_json(output / 'binding.json', {'identity': binding})
        runs_root = output / 'runs'
        if runs_root.exists() and set(p.name for p in runs_root.iterdir()) - {r['id'] for r in plan}:
            raise ValueError('Unexpected candidate directories')
        if (output / 'selection.json').exists() and not all((runs_root / r['id'] / 'completion.json').exists() for r in plan):
            raise ValueError('Selection exists for incomplete study')
        for run in plan:
            folder = runs_root / run['id']
            token = identity({'binding': binding, 'run': run})
            attempts_path = folder / 'attempts.json'
            attempts = read(attempts_path) if attempts_path.exists() else []
            if (folder / 'completion.json').exists():
                results[run['id']] = completed(folder, run, token)
                if attempts and attempts[-1]['status'] == 'STARTED':
                    attempts[-1]['status'] = 'CACHE_VERIFIED_AFTER_INTERRUPTION'
                    atomic_json(attempts_path, attempts)
                continue
            if executed >= cap:
                continue
            if attempts and attempts[-1]['status'] == 'FAILED' and not retry_failed:
                raise RuntimeError('Inspect failure and explicitly request --retry-failed')
            folder.mkdir(parents=True, exist_ok=True)
            if attempts and attempts[-1]['status'] == 'STARTED':
                attempts[-1]['status'] = 'INTERRUPTED'
            attempts.append({'attempt': len(attempts) + 1, 'identity': token, 'status': 'STARTED',
                             'started_utc': datetime.now(timezone.utc).isoformat()})
            atomic_json(attempts_path, attempts)
            try:
                fold, arrays = load_inner(data, run['fold'])
                if (folder / 'checkpoint.pt').exists():
                    saved = validate_checkpoint(torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True), run, token)
                    # A partial checkpoint has no final asset receipt yet: verify its
                    # saved last/best models against the recorded inner metrics first.
                    from paper2.restart.verify_control_search import recompute_metrics
                    recompute_metrics(saved, run, fold, arrays)
                candidate(folder, run, fold, arrays, token)
                results[run['id']] = completed(folder, run, token)
                attempts[-1]['status'] = 'COMPLETE'
                attempts[-1]['completion_sha256'] = sha(folder / 'completion.json')
                atomic_json(attempts_path, attempts)
                executed += 1
            except Exception:
                attempts[-1].update(status='FAILED', traceback=traceback.format_exc())
                atomic_json(attempts_path, attempts)
                raise
        if len(results) == len(plan):
            selection = select_complete(plan, results, config)
            if (output / 'selection.json').exists():
                if read(output / 'selection.json') != selection:
                    raise ValueError('Saved selection differs')
            else:
                atomic_json(output / 'selection.json', selection)
        progress = {'status': 'COMPLETE' if len(results) == len(plan) else 'PARTIAL', 'completed': len(results),
                    'planned': len(plan), 'pending': len(plan) - len(results), 'executed': executed,
                    'selection_available': len(results) == len(plan), 'outer_model_inference': False, 'privacy_claim': False}
        atomic_json(output / 'progress.json', progress)
        return progress


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--package', type=Path, default=EXECUTION)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--cap', type=int, default=1)
    args = p.parse_args()
    if args.freeze and (args.execute or args.resume or args.retry_failed):
        p.error('Freeze cannot be combined with execution/recovery')
    print(freeze(args.output) if args.freeze else execute(args.package, args.output, args.cap, args.resume, not args.execute, args.retry_failed))
