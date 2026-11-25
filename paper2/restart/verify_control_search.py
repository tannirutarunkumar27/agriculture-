"""Independent R12 checkpoint/metric audit; never performs outer inference."""
import argparse
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from paper2.restart import control_search_runner as runner
from paper2.restart.control_convergence import load_inner
from paper2.restart.work_scaled_regression_training import make_regressor
from paper2.restart.validation_search import ROOT, read, identity, sha, atomic_json, writer_lock


def recompute_metrics(checkpoint, run, fold, arrays):
    torch.set_num_threads(1)
    model = make_regressor(arrays['x_train'].shape[1], 32).eval()
    model.load_state_dict(checkpoint['state'])
    with torch.no_grad():
        pt = model(torch.from_numpy(arrays['x_train']))[:, 0].numpy() * run['target_scale']
        pv = model(torch.from_numpy(arrays['x_validation']))[:, 0].numpy() * run['target_scale']
    error = pv.astype(np.float64) - arrays['y_validation'].astype(np.float64)
    y = arrays['y_validation'].astype(np.float64)
    total = float(np.sum((y - y.mean()) ** 2))
    measured = {'samples': len(y), 'rmse': float(np.sqrt(np.mean(error ** 2))),
                'mae': float(np.mean(abs(error))), 'negative_predictions': int((pv < 0).sum()),
                'r2': None if len(y) < 2 or total == 0 else 1 - float(np.sum(error ** 2)) / total}
    last = checkpoint['history'][-1]
    for name, v in measured.items():
        expected = last['validation'][name]
        if (v is None or expected is None) and v != expected:
            raise ValueError('Undefined validation metric differs')
        if v is not None and expected is not None and not math.isclose(v, expected, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError('Final validation metrics do not reproduce')
    lookup = {i: n for n, i in enumerate(fold['train'])}
    groups = [[lookup[i] for i in ids] for ids in fold['clients'].values()]
    train_y = arrays['y_train'].astype(np.float64)
    constant = float(np.mean([train_y[g].mean() for g in groups]))
    mse = float(np.mean([np.mean((pt[g].astype(np.float64) - train_y[g]) ** 2) for g in groups]))
    constant_mse = float(np.mean([np.mean((constant - train_y[g]) ** 2) for g in groups]))
    for name, v in (('equal_client_model_mse', mse), ('equal_client_constant_mse', constant_mse),
                    ('equal_client_optimal_constant', constant)):
        if not math.isclose(v, last['training_equal_client_fit'][name], rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError('Equal-client training metric differs')
    model.load_state_dict(checkpoint['best_state'])
    with torch.no_grad():
        pb = model(torch.from_numpy(arrays['x_validation']))[:, 0].numpy() * run['target_scale']
    best = float(np.sqrt(np.mean((pb.astype(np.float64) - y) ** 2)))
    if not math.isclose(best, checkpoint['best'], rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError('Best validation model metric differs')
    return {'final_training_equal_client_mse': mse, 'best_validation_rmse': best,
            'best_round': checkpoint['best_round']}


def audit(package, study, output):
    package, study = Path(package).resolve(), Path(study).resolve()
    m, config, plan, data = runner.verify_execution(package)
    output = runner.safe_output(output, (package, runner.PROTOCOL, data, study))
    if output.exists():
        raise FileExistsError(output)
    binding = identity(m)
    rows, pending, failures, results = [], [], [], {}
    with writer_lock(study):
        if read(study / 'binding.json') != {'identity': binding}:
            raise ValueError('Study binding differs')
        runs = study / 'runs'
        if runs.exists() and set(p.name for p in runs.iterdir()) - {r['id'] for r in plan}:
            raise ValueError('Unexpected candidate directories')
        for run in plan:
            folder = runs / run['id']
            if (folder / 'attempts.json').exists():
                attempts = read(folder / 'attempts.json')
                failures.extend({'id': run['id'], 'attempt': a['attempt']} for a in attempts if a['status'] == 'FAILED')
            token = identity({'binding': binding, 'run': run})
            if not (folder / 'completion.json').exists():
                pending.append(run['id'])
                if (folder / 'checkpoint.pt').exists():
                    saved = runner.validate_checkpoint(torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True), run, token)
                    fold, arrays = load_inner(data, run['fold'])
                    recompute_metrics(saved, run, fold, arrays)
                continue
            result = runner.completed(folder, run, token)
            saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            fold, arrays = load_inner(data, run['fold'])
            measured = recompute_metrics(saved, run, fold, arrays)
            rows.append({'id': run['id'], **measured, 'completion_sha256': sha(folder / 'completion.json')})
            results[run['id']] = result
        selection_path = study / 'selection.json'
        if pending and selection_path.exists():
            raise ValueError('Incomplete search selected settings')
        if not pending:
            selected = read(selection_path)
            # Independently recompute configuration ordering, equal-group means and median refit budget.
            for record in selected['selections']:
                outer = record['outer_fold']
                options = []
                for scale in config['target_scales']:
                    for lr in config['learning_rates']:
                        matches = [r for r in plan if r['fold'].split('_inner_')[0] == outer and r['target_scale'] == scale and r['learning_rate'] == lr]
                        if len(matches) != 7:
                            raise ValueError('Selection fold coverage differs')
                        best = [min(results[r['id']]['history'], key=lambda h: h['validation']['rmse']) for r in matches]
                        options.append((sum(h['validation']['rmse'] for h in best) / 7, scale, lr, [h['round'] for h in best]))
                chosen = min(options, key=lambda r: r[0])
                c = record['selected']
                if (c['target_scale'], c['learning_rate']) != chosen[1:3] or not math.isclose(c['equal_inner_group_mean_best_validation_rmse'], chosen[0], rel_tol=1e-12) or record['refit_rounds'] != statistics.median(chosen[3]):
                    raise ValueError('Independent selection differs')
            if selected != runner.select_complete(plan, results, config):
                raise ValueError('Complete selection record differs')
        report = {'status': 'PASS_COMPLETE_INNER_SEARCH' if not pending else 'PASS_PARTIAL_INNER_SEARCH',
                  'verified': len(rows), 'planned': len(plan), 'pending': len(pending), 'rows': rows,
                  'failed_attempts': failures, 'selection_available': not pending, 'outer_model_inference': False,
                  'privacy_claim': False, 'scope': 'Final/best inner-model reinference for verification only; previous outer development exposure disclosed'}
    output.mkdir(parents=True)
    atomic_json(output / 'verification.json', report)
    return {k: v for k, v in report.items() if k != 'rows'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, default=runner.EXECUTION)
    p.add_argument('--study', type=Path, default=ROOT / 'paper2/results/restart_control_search_v1')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(audit(args.package, args.study, args.output))
