"""R14 prospective matched simulation plan; deliberately contains no trainer."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

from paper2.restart import public_refit_runner as refit
from paper2.restart import verify_public_refit as refit_audit
from paper2.restart import work_scaled_privacy_simulation_runner as old
from paper2.restart import verify_work_scaled_privacy_simulation as old_audit
from paper2.restart.privacy_accounting import gaussian_budget, replacement_sensitivity
from paper2.restart.validation_search import ROOT, read, sha, atomic_json, runtime, writer_lock

RESULTS = ROOT / 'paper2/results'
PROTOCOL = RESULTS / 'restart_matched_noise_protocol_v1'
OLD_PACKAGE = RESULTS / 'restart_privacy_execution_v3'
OLD_STUDY = RESULTS / 'restart_privacy_simulation_v3'
CODE = [Path(__file__), Path(__file__).with_name('privacy_accounting.py')]
ARTIFACTS = {'config.json', 'plan.json', 'references.json', 'prior_ledger.json',
             'hypothetical_accounting.json', 'parents.json'}
NAMESPACE = 'r14-matched-v1:'


def config():
    return {'version': 'r14-matched-v1', 'simulation_only': True,
            'application_privacy_certified': False, 'study_dp_claim_allowed': False,
            'nonprivate_selection_covered': False, 'unnoised_control_release_covered': False,
            'private_metric_release_covered': False, 'noise_seed_secrecy': False,
            'finite_precision_certified': False, 'sampling_amplification': False,
            'delta': 1e-5, 'participation': 1., 'seeds': refit.protocol.SEEDS,
            'optimizer': 'plain_SGD_no_momentum_no_weight_decay', 'hidden': 32,
            'batch_size': 32, 'local_steps': 4, 'gradient_clip_norm': 1.,
            'clip_rule': 'learning_rate*4', 'noise_multipliers': [2., 4., 8.],
            'initialization': 'fresh_same_fold_seed_as_R13_never_warm_start_trained_model',
            'noise_stream_namespace': NAMESPACE,
            'checkpoint': 'last_fixed_inner_selected_round',
            'inference': 'once_per_run_intent_before_forward_ambiguity_requires_stop',
            'prediction_clipping': False, 'execution_ready': False,
            'scope': 'public_benchmark_development_prior_outer_exposure',
            'primary': 'equal_group_mean_RMSE_per_seed',
            'secondary': ['pooled_RMSE', 'MAE', 'R2', 'species_and_composition_support'],
            'limitations': ['80/224 inner curves hit cap240; all choices at grid edges',
                            '11/56 selected inner fits worse than matched constant',
                            'seeds are not independent regions; tiny unequal groups',
                            '265 unseen systems are Auzeville only',
                            'nonprivate tuning, controls, metrics and public RNG excluded from accounting',
                            'conditional Gaussian bounds are not pipeline privacy guarantees']}


def require_simulation(settings):
    if settings != config():
        raise ValueError('Only the exact public simulation contract is supported')


def make_plan(parents, slots):
    expected = {(f'outer_{i}', s) for i in range(8) for s in refit.protocol.SEEDS}
    if len(parents) != 40 or {(r['fold'], r['seed']) for r in parents} != expected or len({r['id'] for r in parents}) != 40:
        raise ValueError('Complete unique 40-reference matrix required')
    if set(slots) != {f'outer_{i}' for i in range(8)}:
        raise ValueError('Eight fold rosters required')
    universe = set().union(*(set(v) for v in slots.values()))
    if len(universe) != 8 or any(len(v) != 7 or len(set(v)) != 7 or not all(isinstance(s, str) and s for s in v) for v in slots.values()):
        raise ValueError('Eight public groups and seven unique slots per fold required')
    if len({tuple(sorted(universe - set(v))) for v in slots.values()}) != 8:
        raise ValueError('Distinct leave-one-group-out rosters required')
    plan = []
    for r in parents:
        refit.training.validate_run(r)
        if r['method'] != 'fixed_public_equal_client_control' or r['target_scale'] != 1. or r['learning_rate'] != .01:
            raise ValueError('R14 must carry the verified R13 scale1/LR.01 settings')
        for name, multiplier in [('clipped_average', None), ('noisy_sigma2', 2.), ('noisy_sigma4', 4.), ('noisy_sigma8', 8.)]:
            plan.append({**r, 'id': f"{r['fold']}_{name}_seed{r['seed']}", 'reference_id': r['id'],
                         'method': 'fixed_public_matched_noise_simulation', 'variant': name,
                         'clip_norm': r['learning_rate'] * 4, 'noise_multiplier': multiplier,
                         'public_slots': list(slots[r['fold']])})
    return plan


def per_slot(events):
    by_slot = {}
    seen = set()
    for row in events:
        if set(row) != {'event_id', 'slots', 'event'} or not isinstance(row['event_id'], str) or not row['event_id'] or row['event_id'] in seen:
            raise ValueError('Malformed or duplicate ledger event')
        seen.add(row['event_id'])
        if not isinstance(row['slots'], list) or len(row['slots']) != 7 or len(set(row['slots'])) != 7 or not all(isinstance(s, str) and s for s in row['slots']):
            raise ValueError('Seven unique slots per event required')
        event = row['event']
        gaussian_budget([event], 1e-5)
        if type(event['count']) is not int or event['count'] != 1 or event['sensitivity'] <= 0:
            raise ValueError('One positive-sensitivity release per ledger row required')
        for slot in row['slots']:
            by_slot.setdefault(slot, []).append(event)
    return [{'slot': s, **gaussian_budget(e, 1e-5)} for s, e in sorted(by_slot.items())]


def check_prior(ledger, expected_count=8564):
    rows = ledger['events']
    if len(rows) != expected_count or type(ledger['event_count']) is not int or ledger['event_count'] != len(rows):
        raise ValueError('Historical ledger count differs')
    budgets = per_slot(rows)
    if ledger['per_slot'] != budgets or len(budgets) != 8 or ledger['application_privacy_certified'] is not False or ledger['excludes_unnoised_controls'] is not True or ledger['mode'] != 'SIMULATION_ACCOUNTING':
        raise ValueError('Historical ledger accounting or scope differs')
    if any(r['event_id'].startswith(NAMESPACE) for r in rows):
        raise ValueError('New event namespace collides with history')
    return copy.deepcopy(rows)


def planned_events(run):
    if run['noise_multiplier'] is None:
        return []
    sensitivity = replacement_sensitivity(run['clip_norm'], len(run['public_slots']))
    event = {'sensitivity': sensitivity, 'noise_std': sensitivity * run['noise_multiplier'], 'count': 1}
    return [{'event_id': f"{NAMESPACE}{run['id']}:{i}", 'slots': list(run['public_slots']), 'event': dict(event)}
            for i in range(1, run['rounds'] + 1)]


def accounting(plan, prior):
    history = check_prior(prior)
    future = [row for run in plan for row in planned_events(run)]
    combined = per_slot(history + future)
    return {'status': 'HYPOTHETICAL_NOT_EXECUTED', 'prior_events': len(history), 'new_events': len(future),
            'combined_events': len(history) + len(future), 'per_slot': combined,
            'worst_slot_epsilon_upper_bound': max(r['epsilon_upper_bound'] for r in combined),
            'per_run': [{'id': r['id'], 'conditional_budget': gaussian_budget([e['event'] for e in planned_events(r)], 1e-5)
                         if r['noise_multiplier'] is not None else None} for r in plan],
            'application_privacy_certified': False, 'study_dp_claim_allowed': False,
            'qualification': 'Conditional ideal Gaussian noisy-only transcript, fixed public settings and hidden independent noise; actual simulation violates privacy premises'}


def collect_parents():
    """Fresh cached audits; no model inference or training. Old ledger read is locked."""
    ledger_hash = sha(OLD_STUDY / 'shared_ledger.json')
    with tempfile.TemporaryDirectory(prefix='r14_parent_audit_', dir=RESULTS) as tmp:
        tmp = Path(tmp)
        refit_audit.audit(refit.EXECUTION, refit.STUDY, tmp / 'refit')
        old_audit.audit(OLD_PACKAGE, OLD_STUDY, tmp / 'noise')
        ra, oa = read(tmp / 'refit/verification.json'), read(tmp / 'noise/verification.json')
    if ra['status'] != 'PASS_COMPLETE_REFIT' or ra['verified'] != 40 or ra['failed_attempts'] or oa['status'] != 'PASS_COMPLETE' or oa['verified'] != 200 or oa['noisy_events'] != 8564:
        raise ValueError('Complete verified parent studies required')
    _, _, parent_plan, data = refit.verify_execution(refit.EXECUTION)
    slots = {}
    for name in sorted({r['fold'] for r in parent_plan}):
        partition = read(data / 'folds' / name / 'partition.json')
        slots[name] = sorted(partition['clients'])
    with writer_lock(OLD_STUDY):
        prior = read(OLD_STUDY / 'shared_ledger.json')
        if sha(OLD_STUDY / 'shared_ledger.json') != ledger_hash:
            raise ValueError('Prior ledger changed during audit')
    if prior['per_slot'] != oa['per_slot_noisy_only_budgets']:
        raise ValueError('Prior ledger changed after audit')
    check_prior(prior)
    refs = []
    audited = {r['id']: r for r in ra['rows']}
    with writer_lock(refit.STUDY):
        for r in parent_plan:
            folder = refit.STUDY / 'runs' / r['id']; row = audited[r['id']]
            assets = {n: sha(folder / n) for n in ('completion.json', 'attempts.json', 'test_cache.npz', 'fit.json')}
            if assets['completion.json'] != row['completion_sha256'] or assets['attempts.json'] != row['journal_sha256']:
                raise ValueError('Reference changed after audit')
            refs.append({'run': r, 'folder': str(folder), 'assets': assets})
    parents = {'refit_audit': ra, 'prior_audit': oa, 'data': str(data),
               'bindings': {str(p): sha(p) for p in (refit.EXECUTION / 'manifest.json', refit.PROTOCOL / 'manifest.json',
                            OLD_PACKAGE / 'manifest.json', old.PROTOCOL / 'manifest.json', data / 'manifest.json', OLD_STUDY / 'shared_ledger.json')}}
    return parent_plan, slots, prior, refs, parents


def payloads():
    parents, slots, prior, refs, provenance = collect_parents()
    plan = make_plan(parents, slots)
    return {'config.json': config(), 'plan.json': plan, 'references.json': refs,
            'prior_ledger.json': prior, 'hypothetical_accounting.json': accounting(plan, prior), 'parents.json': provenance}


def freeze(output):
    output = refit.safe_output(output, (refit.STUDY, refit.EXECUTION, refit.PROTOCOL, OLD_STUDY, OLD_PACKAGE, old.PROTOCOL,
                                       RESULTS / 'restart_public_data_v1', refit.protocol.integrity.STUDY))
    if output.exists(): raise FileExistsError(output)
    values = payloads()
    output.mkdir(parents=True); (output / 'source_snapshot').mkdir()
    for source in CODE: shutil.copyfile(source, output / 'source_snapshot' / source.name)
    for n, value in values.items(): atomic_json(output / n, value)
    atomic_json(output / 'manifest.json', {'status': 'FROZEN_PROTOCOL_NO_TRAINING', 'planned': 160, 'reused_controls': 40,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'runtime': runtime(),
        'source_sha256': {p.name: sha(p) for p in CODE}, 'artifacts': {n: sha(output / n) for n in ARTIFACTS}})
    return {'status': 'FROZEN_PROTOCOL_NO_TRAINING', 'planned': 160, 'reused_controls': 40}


def verify(package):
    package = Path(package); m = read(package / 'manifest.json')
    if m['status'] != 'FROZEN_PROTOCOL_NO_TRAINING' or m['planned'] != 160 or m['reused_controls'] != 40 or m['runtime'] != runtime() or set(m['artifacts']) != ARTIFACTS:
        raise ValueError('Protocol manifest/runtime differs')
    if set(m['source_sha256']) != {p.name for p in CODE}:
        raise ValueError('Source inventory differs')
    for p in CODE:
        if sha(p) != m['source_sha256'][p.name] or sha(package / 'source_snapshot' / p.name) != sha(p):
            raise ValueError('Protocol source differs')
    for n, h in m['artifacts'].items():
        if sha(package / n) != h: raise ValueError('Protocol artifact differs')
    require_simulation(read(package / 'config.json'))
    for n, value in payloads().items():
        if read(package / n) != value: raise ValueError('Protocol parent/semantic binding differs: ' + n)
    return {'status': 'PASS_FROZEN_MATCHED_PROTOCOL', 'planned': 160, 'reused_controls': 40,
            'training_started': False, 'application_privacy_certified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'verify'))
    parser.add_argument('--output', type=Path, default=PROTOCOL); args = parser.parse_args()
    print(freeze(args.output) if args.action == 'freeze' else verify(args.output))
