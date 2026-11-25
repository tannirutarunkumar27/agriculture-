"""Supplemental attempt/provenance checks without changing frozen R12 code."""
import argparse
from datetime import datetime
import math
from pathlib import Path

from paper2.restart import control_search_runner as search
from paper2.restart.validation_search import ROOT, read, sha, identity, atomic_json, writer_lock

STUDY = ROOT / 'paper2/results/restart_control_search_v1'
METRIC_AUDIT = ROOT / 'paper2/results/restart_control_search_verified_v4/verification.json'


def verify_attempts(study, plan, binding):
    rows = []
    for run in plan:
        folder = Path(study) / 'runs' / run['id']
        path = folder / 'attempts.json'
        if not path.exists():
            raise ValueError('Missing attempt journal: ' + run['id'])
        attempts = read(path)
        if not isinstance(attempts, list) or not attempts:
            raise ValueError('Nonempty attempt journal required')
        token = identity({'binding': binding, 'run': run})
        for i, attempt in enumerate(attempts, 1):
            if type(attempt['attempt']) is not int or attempt['attempt'] != i or attempt['identity'] != token:
                raise ValueError('Attempt sequence/identity changed')
            stamp = datetime.fromisoformat(attempt['started_utc'])
            if stamp.tzinfo is None:
                raise ValueError('Timezone-aware attempt timestamp required')
            if i < len(attempts):
                if attempt['status'] not in ('FAILED', 'INTERRUPTED'):
                    raise ValueError('Invalid earlier attempt lifecycle')
            elif attempt['status'] not in ('COMPLETE', 'CACHE_VERIFIED_AFTER_INTERRUPTION'):
                raise ValueError('Completed search has nonterminal attempt')
            if attempt['status'] == 'FAILED' and not attempt.get('traceback'):
                raise ValueError('Failed attempt lost traceback')
        receipt = read(folder / 'completion.json')
        receipt_hash = sha(folder / 'completion.json')
        if receipt['identity'] != token:
            raise ValueError('Journal/completion identity mismatch')
        if attempts[-1]['status'] == 'COMPLETE' and attempts[-1].get('completion_sha256') != receipt_hash:
            raise ValueError('Attempt completion receipt changed')
        rows.append({'id': run['id'], 'attempts': len(attempts), 'journal_sha256': sha(path),
                     'completion_sha256': receipt_hash,
                     'failed_attempts': sum(a['status'] == 'FAILED' for a in attempts)})
    return rows


def verify_search(package=search.EXECUTION, study=STUDY, metric_audit=METRIC_AUDIT):
    manifest, config, plan, data = search.verify_execution(package)
    metric_audit = Path(metric_audit)
    audited = read(metric_audit)
    if audited['status'] != 'PASS_COMPLETE_INNER_SEARCH' or audited['verified'] != len(plan) or audited['pending'] != 0 or audited['outer_model_inference'] is not False:
        raise ValueError('Complete inner-model metric audit required')
    rows = audited['rows']
    if len(rows) != len(plan) or len({r['id'] for r in rows}) != len(rows) or {r['id'] for r in rows} != {r['id'] for r in plan}:
        raise ValueError('Metric audit coverage differs')
    by_id = {r['id']: r for r in rows}
    binding = identity(manifest)
    with writer_lock(study):
        if read(Path(study) / 'binding.json') != {'identity': binding}:
            raise ValueError('Study binding changed')
        if {p.name for p in (Path(study) / 'runs').iterdir()} != {r['id'] for r in plan}:
            raise ValueError('Study run inventory changed')
        journals = verify_attempts(study, plan, binding)
        results = {}
        for run, journal in zip(plan, journals):
            result = search.completed(Path(study) / 'runs' / run['id'], run,
                                      identity({'binding': binding, 'run': run}))
            row = by_id[run['id']]
            best = min(result['history'], key=lambda h: h['validation']['rmse'])
            if row['completion_sha256'] != journal['completion_sha256'] or row['best_round'] != best['round']:
                raise ValueError('Metric audit model/receipt binding changed')
            for reported, expected in ((row['best_validation_rmse'], best['validation']['rmse']),
                (row['final_training_equal_client_mse'], result['history'][-1]['training_equal_client_fit']['equal_client_model_mse'])):
                if not math.isclose(reported, expected, rel_tol=1e-7, abs_tol=1e-7):
                    raise ValueError('Metric audit values differ from checked history')
            results[run['id']] = result
        selected = search.select_complete(plan, results, config)
        if read(Path(study) / 'selection.json') != selected:
            raise ValueError('Selection changed')
        report = {'status': 'PASS_COMPLETE_SEARCH_PROVENANCE', 'verified': len(plan),
                  'attempts': sum(r['attempts'] for r in journals),
                  'failed_attempts': sum(r['failed_attempts'] for r in journals), 'journals': journals,
                  'execution_manifest_sha256': sha(Path(package) / 'manifest.json'),
                  'selection_sha256': sha(Path(study) / 'selection.json'),
                  'metric_audit_sha256': sha(metric_audit), 'source_sha256': sha(__file__),
                  'outer_model_inference': False, 'retraining': False}
    return report, selected, data


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = search.safe_output(args.output, (STUDY, search.PROTOCOL, search.EXECUTION))
    if output.exists():
        raise FileExistsError(output)
    report, _, _ = verify_search()
    output.mkdir(parents=True)
    atomic_json(output / 'verification.json', report)
    print({k: v for k, v in report.items() if k != 'journals'})
