"""Independent cached raw-output metrics/support audit; no model forward calls."""
import argparse
from pathlib import Path

import numpy as np

from paper2.restart import public_refit_runner as runner
from paper2.restart import public_refit_inference as inference
from paper2.restart import public_refit_training as training
from paper2.restart.validation_search import read, sha, identity, atomic_json, writer_lock


def independent_scores(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Finite aligned cached outputs required')
    error = p - y; denominator = float(np.sum((y - y.mean()) ** 2))
    return {'samples': len(y), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'mae': float(np.mean(abs(error))), 'negative_predictions': int((p < 0).sum()),
            'r2': None if len(y) < 2 or denominator == 0 else 1 - float(np.sum(error ** 2)) / denominator}


def audit(package, study, output):
    package, study = Path(package).resolve(), Path(study).resolve()
    m, _, plan, data = runner.verify_execution(package)
    output = runner.safe_output(output, (package, runner.PROTOCOL, data, study, runner.protocol.integrity.STUDY))
    if output.exists(): raise FileExistsError(output)
    binding = identity(m); rows = []; pending = []; failures = []
    with writer_lock(study):
        if read(study / 'binding.json') != {'identity': binding}: raise ValueError('Study binding differs')
        if (study / 'runs').exists() and set(p.name for p in (study / 'runs').iterdir()) - {r['id'] for r in plan}:
            raise ValueError('Unexpected run inventory')
        for run in plan:
            folder = study / 'runs' / run['id']; token = identity({'binding': binding, 'run': run})
            complete = (folder / 'completion.json').exists()
            attempts = runner.verify_journal(folder, token, complete)
            failures.extend({'id': run['id'], 'attempt': a['attempt']} for a in attempts if a['status'] == 'FAILED')
            if not complete:
                pending.append(run['id']); continue
            fold, a = training.load_training_fold(data, run)
            test = inference.load_test(data, run, fold)
            result = inference.verify_completed(folder, run, fold, a, test, token)
            context = inference.context_for(fold, run)
            with np.load(folder / 'test_cache.npz', allow_pickle=False) as cache:
                measured = independent_scores(cache['y'], cache['prediction'])
                if measured != result['test']: raise ValueError('Independent raw-unit metrics differ')
                panels = {}
                for name in ('all_species_seen', 'composition_seen'):
                    mask = np.array([v[name] for v in context['support']], bool)
                    panels[name] = {'seen': independent_scores(cache['y'][mask], cache['prediction'][mask]) if mask.any() else None,
                                    'unseen': independent_scores(cache['y'][~mask], cache['prediction'][~mask]) if (~mask).any() else None}
                if panels != result['panels']: raise ValueError('Independent support-panel metrics differ')
            rows.append({'id': run['id'], 'group': fold['test_group'], 'seed': run['seed'], 'test': measured,
                         'panels': panels, 'completion_sha256': sha(folder / 'completion.json'),
                         'journal_sha256': sha(folder / 'attempts.json')})
        report = {'status': 'PASS_COMPLETE_REFIT' if not pending else 'PASS_PARTIAL_REFIT',
                  'verified': len(rows), 'planned': len(plan), 'pending': len(pending), 'failed_attempts': failures,
                  'rows': rows, 'model_reinference': False, 'privacy_claim': False,
                  'execution_manifest_sha256': sha(package / 'manifest.json'),
                  'scope': 'development; partial groups/seeds are not final comparative evidence'}
    output.mkdir(parents=True); atomic_json(output / 'verification.json', report)
    return {k: v for k, v in report.items() if k != 'rows'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--package', type=Path, default=runner.EXECUTION)
    p.add_argument('--study', type=Path, default=runner.STUDY); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(audit(args.package, args.study, args.output))
