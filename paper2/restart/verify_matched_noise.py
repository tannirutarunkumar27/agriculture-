"""Independent R14 cached metrics, journal, model and cumulative ledger audit."""
import argparse
from pathlib import Path
import math
import numpy as np

from paper2.restart import matched_noise_runner as runner
from paper2.restart import matched_noise_inference as inference
from paper2.restart.verify_public_refit import independent_scores
from paper2.restart.validation_search import read, sha, identity, atomic_json, writer_lock


def audit(package, study, output):
    m, config, plan, data = runner.verify_execution(package)
    study = Path(study); output = runner.safe_output(output, package)
    if output.exists(): raise FileExistsError(output)
    if study.resolve() == output or study.resolve() in output.parents or output in study.resolve().parents:
        raise ValueError('Separate audit output required')
    binding = identity(m); rows = []; pending = []; failures = []
    with writer_lock(study):
        if read(study / 'binding.json') != {'identity': binding}: raise ValueError('Study binding differs')
        if (study / 'runs').exists() and set(p.name for p in (study / 'runs').iterdir()) - {r['id'] for r in plan}:
            raise ValueError('Unexpected run inventory')
        for run in plan:
            folder = study / 'runs' / run['id']; token = identity({'binding': binding, 'run': run})
            complete = (folder / 'completion.json').exists()
            journal = runner.common.verify_journal(folder, token, complete)
            failures.extend({'id': run['id'], 'attempt': a['attempt']} for a in journal if a['status'] == 'FAILED')
            if not complete:
                pending.append(run['id']); continue
            fold, train = runner.training.refit.load_training_fold(data, run)
            test = inference.load_test(data, run, fold)
            result = inference.verify_completed(folder, run, fold, train, test, token)
            with np.load(folder / 'test_cache.npz', allow_pickle=False) as cache:
                measured = independent_scores(cache['y'], cache['prediction'])
                if measured != result['test']: raise ValueError('Independent metrics differ')
                panels = {}
                for name in ('all_species_seen', 'composition_seen'):
                    mask = np.array([r[name] for r in fold['support']['test']], bool)
                    panels[name] = {'seen': independent_scores(cache['y'][mask], cache['prediction'][mask]) if mask.any() else None,
                                    'unseen': independent_scores(cache['y'][~mask], cache['prediction'][~mask]) if (~mask).any() else None}
                if panels != result['panels']: raise ValueError('Independent panels differ')
            rows.append({'id': run['id'], 'variant': run['variant'], 'reference_id': run['reference_id'],
                         'group': fold['test_group'], 'seed': run['seed'], 'test': measured, 'panels': panels,
                         'completion_sha256': sha(folder / 'completion.json'), 'journal_sha256': sha(folder / 'attempts.json')})
        ledger = runner.collect_ledger(study, plan, config, data, binding)
        if ledger != read(study / 'shared_ledger.json'): raise ValueError('Shared ledger differs from checkpoints')
        # Independently recompute scalar rho/epsilon for each protected slot.
        for budget in ledger['per_slot']:
            events = [e['event'] for e in ledger['events'] if budget['slot'] in e['slots']]
            rho = math.fsum(e['sensitivity'] ** 2 / (2 * e['noise_std'] ** 2) for e in events)
            epsilon = rho + 2 * math.sqrt(rho * math.log(1e5))
            if not math.isclose(rho, budget['rho'], rel_tol=1e-12) or not math.isclose(epsilon, budget['epsilon_upper_bound'], rel_tol=1e-12) or budget['releases'] != len(events):
                raise ValueError('Independent composed accounting differs')
        report = {'status': 'PASS_COMPLETE_MATCHED_STUDY' if not pending else 'PASS_PARTIAL_MATCHED_STUDY',
                  'verified': len(rows), 'planned': len(plan), 'pending': len(pending), 'failed_attempts': failures,
                  'rows': rows, 'noise_events': ledger['event_count'], 'prior_events': ledger['prior_event_count'],
                  'model_reinference': False, 'application_privacy_certified': False,
                  'execution_manifest_sha256': sha(Path(package) / 'manifest.json'),
                  'scope': 'development public simulation; prior nonprivate selection and outer exposure'}
    output.mkdir(parents=True); atomic_json(output / 'verification.json', report)
    return {k: v for k, v in report.items() if k != 'rows'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--package', type=Path, default=runner.EXECUTION)
    p.add_argument('--study', type=Path, default=runner.STUDY); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(audit(args.package, args.study, args.output))
