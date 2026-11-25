"""Separately frozen capped coordinator for the unnoised40-run public refit."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

from paper2.restart import public_refit_protocol as protocol
from paper2.restart import public_refit_training as training
from paper2.restart import public_refit_inference as inference
from paper2.restart.control_search_runner import safe_output
from paper2.restart.validation_search import ROOT, read, sha, identity, atomic_json, writer_lock, runtime

PROTOCOL = ROOT / 'paper2/results/restart_public_refit_protocol_v1'
EXECUTION = ROOT / 'paper2/results/restart_public_refit_execution_v1'
STUDY = ROOT / 'paper2/results/restart_public_refit_study_v1'
CODE = [Path(__file__), Path(inference.__file__), Path(__file__).with_name('verify_public_refit.py')]


def freeze(output):
    protocol.verify(PROTOCOL)
    m = read(PROTOCOL / 'manifest.json')
    output = safe_output(output, (PROTOCOL, Path(m['data']), protocol.integrity.STUDY))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True); (output / 'source_snapshot').mkdir()
    for p in CODE:
        shutil.copyfile(p, output / 'source_snapshot' / p.name)
    atomic_json(output / 'manifest.json', {'status': 'FROZEN_EXECUTION_NO_OUTER_INFERENCE', 'protocol': str(PROTOCOL),
        'protocol_manifest_sha256': sha(PROTOCOL / 'manifest.json'), 'runtime': runtime(), 'planned': 40,
        'source_sha256': {p.name: sha(p) for p in CODE}, 'privacy_claim': False})
    return {'status': 'FROZEN_EXECUTION_NO_OUTER_INFERENCE', 'planned': 40}


def verify_execution(package):
    package = Path(package).resolve(); m = read(package / 'manifest.json')
    if m['protocol'] != str(PROTOCOL) or sha(PROTOCOL / 'manifest.json') != m['protocol_manifest_sha256'] or runtime() != m['runtime'] or m['planned'] != 40 or m['privacy_claim'] is not False:
        raise ValueError('Execution protocol/runtime changed')
    if set(m['source_sha256']) != {p.name for p in CODE}:
        raise ValueError('Execution source inventory changed')
    for p in CODE:
        if sha(p) != m['source_sha256'][p.name] or sha(package / 'source_snapshot' / p.name) != sha(p):
            raise ValueError('Execution source changed')
    protocol.verify(PROTOCOL)
    parent = read(PROTOCOL / 'manifest.json')
    return m, read(PROTOCOL / 'config.json'), read(PROTOCOL / 'plan.json'), Path(parent['data'])


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_journal(folder, token, complete=False):
    path = Path(folder) / 'attempts.json'
    if not path.exists():
        if complete: raise ValueError('Missing completed attempt journal')
        if Path(folder).exists() and any(Path(folder).iterdir()):
            raise ValueError('Unjournaled refit assets cannot be recovered')
        return []
    attempts = read(path)
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('Nonempty attempt journal required')
    for i, a in enumerate(attempts, 1):
        if a['identity'] != token or type(a['attempt']) is not int or a['attempt'] != i or datetime.fromisoformat(a['started_utc']).tzinfo is None:
            raise ValueError('Attempt identity/sequence/time changed')
        if a['status'] not in ('STARTED', 'FAILED', 'INTERRUPTED', 'COMPLETE', 'CACHE_VERIFIED_AFTER_INTERRUPTION', 'CACHE_VERIFIED_AFTER_FAILURE'):
            raise ValueError('Unknown attempt state')
        if i < len(attempts) and a['status'] not in ('FAILED', 'INTERRUPTED'):
            raise ValueError('Earlier attempt not interrupted/failed')
        if a['status'] == 'FAILED' and not a.get('traceback'):
            raise ValueError('Failed attempt lost traceback')
    if complete:
        if attempts[-1]['status'] not in ('COMPLETE', 'CACHE_VERIFIED_AFTER_INTERRUPTION', 'CACHE_VERIFIED_AFTER_FAILURE') or attempts[-1].get('completion_sha256') != sha(Path(folder) / 'completion.json'):
            raise ValueError('Completed attempt receipt changed')
    return attempts


def execute(package, output, cap=1, resume=False, dry_run=True, retry_failed=False):
    if type(cap) is not int or cap < 1:
        raise ValueError('Positive integer execution cap required')
    package = Path(package).resolve()
    m, config, plan, data = verify_execution(package)
    output = safe_output(output, (package, PROTOCOL, data, protocol.integrity.STUDY))
    if dry_run:
        return {'status': 'DRY_RUN_NO_TRAINING_OR_INFERENCE', 'planned': len(plan), 'cap': cap}
    if output.exists() and not resume:
        raise ValueError('Explicit resume required')
    output.mkdir(parents=True, exist_ok=True)
    binding = identity(m); done = executed = 0
    with writer_lock(output):
        if (output / 'binding.json').exists():
            if read(output / 'binding.json') != {'identity': binding}: raise ValueError('Study binding changed')
        elif resume or any(p.name != 'writer.lock' for p in output.iterdir()):
            raise ValueError('Unbound output cannot be recovered')
        else:
            atomic_json(output / 'binding.json', {'identity': binding})
        runs = output / 'runs'
        if runs.exists() and set(p.name for p in runs.iterdir()) - {r['id'] for r in plan}:
            raise ValueError('Unexpected refit run directories')
        for run in plan:
            folder = runs / run['id']; token = identity({'binding': binding, 'run': run})
            attempts = verify_journal(folder, token)
            if (folder / 'completion.json').exists():
                fold, a = training.load_training_fold(data, run)
                tests = inference.load_test(data, run, fold)
                inference.verify_completed(folder, run, fold, a, tests, token)
                if not attempts: raise ValueError('Completed cache lost attempt history')
                if attempts[-1]['status'] == 'STARTED':
                    attempts[-1].update(status='CACHE_VERIFIED_AFTER_INTERRUPTION', completion_sha256=sha(folder / 'completion.json'), finished_utc=now())
                    atomic_json(folder / 'attempts.json', attempts)
                elif attempts[-1]['status'] == 'FAILED':
                    attempts.append({'attempt': len(attempts) + 1, 'identity': token, 'started_utc': now(),
                                     'status': 'CACHE_VERIFIED_AFTER_FAILURE', 'completion_sha256': sha(folder / 'completion.json'), 'finished_utc': now()})
                    atomic_json(folder / 'attempts.json', attempts)
                else:
                    verify_journal(folder, token, True)
                verify_journal(folder, token, True); done += 1
                continue
            if executed >= cap: continue
            if (folder / 'inference_started.json').exists():
                raise RuntimeError('Ambiguous outer inference cannot be retried')
            if attempts and attempts[-1]['status'] == 'FAILED' and not retry_failed:
                raise RuntimeError('Inspect failed training and explicitly request --retry-failed')
            if attempts and attempts[-1]['status'] not in ('FAILED', 'STARTED'):
                raise ValueError('Terminal journal without completion receipt')
            folder.mkdir(parents=True, exist_ok=True)
            if attempts and attempts[-1]['status'] == 'STARTED': attempts[-1]['status'] = 'INTERRUPTED'
            attempts.append({'attempt': len(attempts) + 1, 'identity': token, 'status': 'STARTED', 'started_utc': now()})
            atomic_json(folder / 'attempts.json', attempts)
            try:
                fold, a = training.load_training_fold(data, run)
                context = inference.context_for(fold, run)
                if (folder / 'partition_context.json').exists() and read(folder / 'partition_context.json') != context:
                    raise ValueError('Recovered support context differs')
                atomic_json(folder / 'partition_context.json', context)
                if (folder / 'checkpoint.pt').exists():
                    inference.verify_checkpoint(folder, run, fold, a, token)
                training.fit(folder, run, config['neural_settings'], fold, a['x_train'], a['y_train'], token)
                tests = inference.load_test(data, run, fold)
                inference.evaluate_once(folder, run, fold, a, tests, token)
                attempts[-1].update(status='COMPLETE', completion_sha256=sha(folder / 'completion.json'), finished_utc=now())
                atomic_json(folder / 'attempts.json', attempts)
                verify_journal(folder, token, True)
                done += 1; executed += 1
            except Exception:
                attempts[-1].update(status='FAILED', traceback=traceback.format_exc(), finished_utc=now())
                atomic_json(folder / 'attempts.json', attempts)
                atomic_json(output / 'progress.json', {'status': 'FAILED', 'failed_run': run['id'], 'completed': done, 'planned': len(plan)})
                raise
        result = {'status': 'COMPLETE' if done == len(plan) else 'PARTIAL', 'completed': done,
                  'planned': len(plan), 'pending': len(plan) - done, 'executed': executed, 'privacy_claim': False,
                  'scope': 'development_unnoised_fixed_public_equal_client_refit'}
        atomic_json(output / 'progress.json', result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--package', type=Path, default=EXECUTION); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--freeze', action='store_true'); p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true'); p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--cap', type=int, default=1); args = p.parse_args()
    if args.freeze and (args.execute or args.resume or args.retry_failed): p.error('Freeze is separate from execution')
    print(freeze(args.output) if args.freeze else execute(args.package, args.output, args.cap, args.resume, not args.execute, args.retry_failed))
