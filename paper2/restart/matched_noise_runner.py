"""Bounded R14 coordinator; exact resume, once-only test inference, carried ledger."""
import argparse
from pathlib import Path
import shutil
import traceback
import torch

from paper2.restart import matched_noise_protocol as protocol
from paper2.restart import matched_noise_training as training
from paper2.restart import matched_noise_inference as inference
from paper2.restart import public_refit_runner as common
from paper2.restart.validation_search import ROOT, read, sha, identity, atomic_json, writer_lock, runtime

EXECUTION = protocol.RESULTS / 'restart_matched_noise_execution_v1'
STUDY = protocol.RESULTS / 'restart_matched_noise_study_v1'
CODE = [Path(__file__), Path(training.__file__), Path(inference.__file__), Path(__file__).with_name('verify_matched_noise.py')]


def safe_output(output, package=EXECUTION):
    return common.safe_output(output, (package, protocol.PROTOCOL, protocol.OLD_STUDY, protocol.OLD_PACKAGE,
        protocol.old.PROTOCOL, common.STUDY, common.EXECUTION, common.PROTOCOL,
        protocol.RESULTS / 'restart_public_data_v1', common.protocol.integrity.STUDY))


def freeze(output):
    protocol.verify(protocol.PROTOCOL); output = safe_output(output, protocol.PROTOCOL)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); (output / 'source_snapshot').mkdir()
    for p in CODE: shutil.copyfile(p, output / 'source_snapshot' / p.name)
    atomic_json(output / 'manifest.json', {'status': 'FROZEN_MATCHED_EXECUTION_NO_TRAINING', 'planned': 160,
        'protocol_manifest_sha256': sha(protocol.PROTOCOL / 'manifest.json'), 'runtime': runtime(),
        'source_sha256': {p.name: sha(p) for p in CODE}, 'application_privacy_certified': False})
    return {'status': 'FROZEN_MATCHED_EXECUTION_NO_TRAINING', 'planned': 160}


def verify_execution(package):
    package = Path(package); m = read(package / 'manifest.json')
    if m['status'] != 'FROZEN_MATCHED_EXECUTION_NO_TRAINING' or m['planned'] != 160 or m['runtime'] != runtime() or m['protocol_manifest_sha256'] != sha(protocol.PROTOCOL / 'manifest.json') or m['application_privacy_certified'] is not False:
        raise ValueError('Matched execution binding differs')
    if set(m['source_sha256']) != {p.name for p in CODE}: raise ValueError('Matched source inventory differs')
    for p in CODE:
        if sha(p) != m['source_sha256'][p.name] or sha(package / 'source_snapshot' / p.name) != sha(p):
            raise ValueError('Matched execution source differs')
    protocol.verify(protocol.PROTOCOL)
    return m, read(protocol.PROTOCOL / 'config.json'), read(protocol.PROTOCOL / 'plan.json'), Path(read(protocol.PROTOCOL / 'parents.json')['data'])


def collect_ledger(output, plan, config, data, binding):
    """Pure reconstruction; caller owns lock. Saved historical rows retain exact IDs."""
    output = Path(output); prior = read(protocol.PROTOCOL / 'prior_ledger.json')
    events = protocol.check_prior(prior)
    for run in plan:
        folder = output / 'runs' / run['id']; token = identity({'binding': binding, 'run': run})
        common.verify_journal(folder, token)
        if (folder / 'checkpoint.pt').exists():
            fold, a = training.refit.load_training_fold(data, run)
            signature, groups = training.training_context(run, config, fold, a['x_train'], a['y_train'], token)
            saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            training.check_checkpoint(saved, run, signature, groups, a['x_train'], a['y_train'])
            events.extend(saved['events'])
    ledger = {'events': events, 'event_count': len(events), 'prior_event_count': len(prior['events']),
              'per_slot': protocol.per_slot(events), 'mode': 'SIMULATION_ACCOUNTING',
              'excludes_unnoised_controls': True, 'nonprivate_selection_covered': False,
              'application_privacy_certified': False, 'study_dp_claim_allowed': False}
    if (output / 'shared_ledger.json').exists():
        previous = read(output / 'shared_ledger.json'); mapping = {e['event_id']: e for e in events}
        if previous['event_count'] != len(previous['events']) or previous['per_slot'] != protocol.per_slot(previous['events']):
            raise ValueError('Existing ledger is internally inconsistent')
        if previous['events'][:len(prior['events'])] != prior['events'] or any(mapping.get(e['event_id']) != e for e in previous['events']):
            raise ValueError('Committed ledger event removed/changed')
        for field in ('prior_event_count', 'mode', 'excludes_unnoised_controls', 'nonprivate_selection_covered', 'application_privacy_certified', 'study_dp_claim_allowed'):
            if previous[field] != ledger[field]: raise ValueError('Ledger scope differs')
    return ledger


def save_ledger(output, plan, config, data, binding):
    ledger = collect_ledger(output, plan, config, data, binding)
    atomic_json(Path(output) / 'shared_ledger.json', ledger)
    return ledger['event_count']


def execute(package, output, cap=1, resume=False, dry_run=True, retry_failed=False):
    if type(cap) is not int or cap < 1: raise ValueError('Positive integer cap required')
    m, config, plan, data = verify_execution(package); output = safe_output(output, package)
    if dry_run: return {'status': 'DRY_RUN', 'planned': len(plan), 'cap': cap, 'training_started': False}
    if output.exists() and not resume: raise ValueError('Explicit resume required')
    output.mkdir(parents=True, exist_ok=True); binding = identity(m); done = executed = 0
    with writer_lock(output):
        if (output / 'binding.json').exists():
            if read(output / 'binding.json') != {'identity': binding}: raise ValueError('Study binding differs')
        elif resume or any(p.name != 'writer.lock' for p in output.iterdir()):
            raise ValueError('Unbound output cannot be recovered')
        else: atomic_json(output / 'binding.json', {'identity': binding})
        runs = output / 'runs'
        if runs.exists() and set(p.name for p in runs.iterdir()) - {r['id'] for r in plan}:
            raise ValueError('Unexpected matched run inventory')
        save_ledger(output, plan, config, data, binding)
        for run in plan:
            folder = runs / run['id']; token = identity({'binding': binding, 'run': run})
            attempts = common.verify_journal(folder, token)
            if (folder / 'completion.json').exists():
                fold, a = training.refit.load_training_fold(data, run)
                test = inference.load_test(data, run, fold)
                inference.verify_completed(folder, run, fold, a, test, token)
                if attempts[-1]['status'] == 'STARTED':
                    attempts[-1].update(status='CACHE_VERIFIED_AFTER_INTERRUPTION', completion_sha256=sha(folder / 'completion.json'), finished_utc=common.now())
                    atomic_json(folder / 'attempts.json', attempts)
                elif attempts[-1]['status'] == 'FAILED':
                    attempts.append({'attempt': len(attempts) + 1, 'identity': token, 'started_utc': common.now(),
                        'status': 'CACHE_VERIFIED_AFTER_FAILURE', 'completion_sha256': sha(folder / 'completion.json'), 'finished_utc': common.now()})
                    atomic_json(folder / 'attempts.json', attempts)
                common.verify_journal(folder, token, True); done += 1; continue
            if executed >= cap: continue
            if (folder / 'inference_started.json').exists(): raise RuntimeError('Ambiguous outer inference cannot be retried')
            if attempts and attempts[-1]['status'] == 'FAILED' and not retry_failed:
                raise RuntimeError('Inspect failed training before explicit --retry-failed')
            if attempts and attempts[-1]['status'] not in ('FAILED', 'STARTED'):
                raise ValueError('Terminal journal without completion')
            folder.mkdir(parents=True, exist_ok=True)
            if attempts and attempts[-1]['status'] == 'STARTED': attempts[-1]['status'] = 'INTERRUPTED'
            attempts.append({'attempt': len(attempts) + 1, 'identity': token, 'status': 'STARTED', 'started_utc': common.now()})
            atomic_json(folder / 'attempts.json', attempts)
            try:
                fold, a = training.refit.load_training_fold(data, run); context = inference.context_for(fold, run)
                if (folder / 'partition_context.json').exists() and read(folder / 'partition_context.json') != context:
                    raise ValueError('Recovered context differs')
                atomic_json(folder / 'partition_context.json', context)
                training.fit(folder, run, config, fold, a['x_train'], a['y_train'], token)
                save_ledger(output, plan, config, data, binding)
                tests = inference.load_test(data, run, fold)
                inference.evaluate_once(folder, run, fold, a, tests, token)
                attempts[-1].update(status='COMPLETE', completion_sha256=sha(folder / 'completion.json'), finished_utc=common.now())
                atomic_json(folder / 'attempts.json', attempts); common.verify_journal(folder, token, True)
                done += 1; executed += 1
            except Exception:
                attempts[-1].update(status='FAILED', traceback=traceback.format_exc(), finished_utc=common.now())
                atomic_json(folder / 'attempts.json', attempts)
                atomic_json(output / 'progress.json', {'status': 'FAILED', 'failed_run': run['id'], 'completed': done, 'planned': len(plan)})
                # Retain the primary error. Checkpoint events are the authoritative recovery source.
                try: save_ledger(output, plan, config, data, binding)
                except Exception: atomic_json(output / 'ledger_failure.json', {'traceback': traceback.format_exc()})
                raise
        events = save_ledger(output, plan, config, data, binding)
        result = {'status': 'COMPLETE' if done == len(plan) else 'PARTIAL', 'completed': done, 'planned': len(plan),
                  'pending': len(plan) - done, 'executed': executed, 'noise_events_including_history': events,
                  'application_privacy_certified': False}
        atomic_json(output / 'progress.json', result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--package', type=Path, default=EXECUTION)
    p.add_argument('--output', type=Path, default=STUDY); p.add_argument('--freeze', action='store_true')
    p.add_argument('--execute', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-failed', action='store_true'); p.add_argument('--cap', type=int, default=1); args = p.parse_args()
    if args.freeze and (args.execute or args.resume or args.retry_failed): p.error('Freeze and execution are separate')
    print(freeze(args.output) if args.freeze else execute(args.package, args.output, args.cap, args.resume, not args.execute, args.retry_failed))
