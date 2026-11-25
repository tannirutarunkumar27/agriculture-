"""Prospective unnoised fixed-public refit plan; no training/test inference."""
import argparse
from pathlib import Path
import shutil

from paper2.restart import control_search_integrity as integrity
from paper2.restart import control_search_runner as search
from paper2.restart.public_refit_training import SETTINGS, validate_run
from paper2.restart.validation_search import ROOT, read, sha, atomic_json, runtime

SEEDS = [123, 2024, 3407, 9999, 2718]
CODE = [Path(__file__), Path(__file__).with_name('public_refit_training.py'), Path(integrity.__file__)]


def make_plan(selected, prepared, seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 or s == 42 for s in seeds):
        raise ValueError('Distinct seeds excluding inner-tuning seed42 required')
    if selected['status'] != 'COMPLETE_INNER_SELECTION' or selected['candidates'] != 224 or selected['outer_model_inference'] is not False or selected['privacy_claim'] is not False:
        raise ValueError('Complete development inner selection required')
    rows = selected['selections']
    if {r['outer_fold'] for r in rows} != {f'outer_{i}' for i in range(8)} or len(rows) != 8:
        raise ValueError('Eight unique outer selections required')
    plan = []
    for row in sorted(rows, key=lambda r: r['outer_fold']):
        if row['outer_fold'] not in prepared['assets']:
            raise ValueError('Missing refit data fold')
        for seed in seeds:
            run = {'id': f"{row['outer_fold']}_equal_public_seed{seed}", 'fold': row['outer_fold'],
                   'method': 'fixed_public_equal_client_control', 'seed': seed,
                   'target_scale': row['selected']['target_scale'], 'learning_rate': row['selected']['learning_rate'],
                   'rounds': row['refit_rounds']}
            validate_run(run)
            plan.append(run)
    return plan


def make_config():
    return {'version': 'fixed-public-unnoised-refit-v1', 'seeds': SEEDS, 'neural_settings': SETTINGS,
              'task': 'retrospective conditional harvested seed dry matter t/ha',
              'primary': 'equal_geographic_group_mean_RMSE_per_seed',
              'secondary': ['pooled_RMSE', 'MAE', 'R2_null_when_undefined', 'seen_unseen_species_composition_panels'],
              'checkpoint': 'last_fixed_inner_selected_refit_round_no_test_selection',
              'inference': 'once_per_run_with_identity_bound_receipts_and_fail_closed_ambiguity',
              'prediction_clipping': False, 'privacy_claim': False,
              'scope': 'development_public_benchmark_not_independent_confirmation',
              'prior_exposure': 'All earlier baseline/guard/privacy outer outcomes exposed; same five seeds reused, not new regions',
              'limitations': ['80/224 inner candidates hit240', 'all choices at grid edges',
                              '11/56 selected inner training fits worse than matched constant',
                              'no noise/SecAgg/production privacy in this plan'],
              'execution_runner': 'Separate bounded once-only inference coordinator still required before execution'}


def freeze(output):
    verified, selected, data = integrity.verify_search()
    output = search.safe_output(output, (search.EXECUTION, search.PROTOCOL, integrity.STUDY, data))
    if output.exists():
        raise FileExistsError(output)
    plan = make_plan(selected, read(data / 'manifest.json'), SEEDS)
    config = make_config()
    output.mkdir(parents=True)
    (output / 'source_snapshot').mkdir()
    for p in CODE:
        shutil.copyfile(p, output / 'source_snapshot' / p.name)
    for name, value in (('config.json', config), ('plan.json', plan), ('selected.json', selected),
                        ('search_integrity.json', verified)):
        atomic_json(output / name, value)
    manifest = {'status': 'FROZEN_PLAN_AND_TRAINING_CORE_NO_OUTER_INFERENCE', 'planned': len(plan),
                'data': str(data), 'data_manifest_sha256': sha(data / 'manifest.json'), 'runtime': runtime(),
                'parent_execution_manifest_sha256': verified['execution_manifest_sha256'],
                'metric_audit': str(integrity.METRIC_AUDIT), 'metric_audit_sha256': verified['metric_audit_sha256'],
                'source_sha256': {p.name: sha(p) for p in CODE},
                'artifacts': {n: sha(output / n) for n in ('config.json', 'plan.json', 'selected.json', 'search_integrity.json')}}
    atomic_json(output / 'manifest.json', manifest)
    return {'status': manifest['status'], 'planned': len(plan)}


def verify(package):
    package = Path(package)
    m = read(package / 'manifest.json')
    if runtime() != m['runtime'] or set(m['artifacts']) != {'config.json', 'plan.json', 'selected.json', 'search_integrity.json'} or any(sha(package / n) != h for n, h in m['artifacts'].items()):
        raise ValueError('Refit package/runtime changed')
    for p in CODE:
        if sha(p) != m['source_sha256'][p.name] or sha(package / 'source_snapshot' / p.name) != sha(p):
            raise ValueError('Refit source changed')
    verified, selected, data = integrity.verify_search(metric_audit=m['metric_audit'])
    if verified != read(package / 'search_integrity.json') or selected != read(package / 'selected.json') or sha(data / 'manifest.json') != m['data_manifest_sha256'] or str(data) != m['data'] or verified['metric_audit_sha256'] != m['metric_audit_sha256'] or verified['execution_manifest_sha256'] != m['parent_execution_manifest_sha256']:
        raise ValueError('Refit search/data binding changed')
    config, plan = read(package / 'config.json'), read(package / 'plan.json')
    if config != make_config() or plan != make_plan(selected, read(data / 'manifest.json'), config['seeds']) or m['planned'] != 40:
        raise ValueError('Refit configuration/plan changed')
    return {'status': 'PASS_FROZEN_REFIT_PLAN_AND_CORE', 'planned': len(plan), 'outer_model_inference': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('freeze', 'verify'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(freeze(args.output) if args.action == 'freeze' else verify(args.output))
