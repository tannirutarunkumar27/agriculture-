"""Complete matched R14 analysis using the unchanged tested aggregation."""
import argparse
from pathlib import Path
import numpy as np
import torch

from paper2.restart import matched_noise_runner as runner
from paper2.restart import verify_matched_noise as auditor
from paper2.restart import summarize_public_refit as existing
from paper2.restart.validation_search import read, sha, atomic_json, writer_lock

REFERENCE = 'equal_average_reference'
VARIANTS = (REFERENCE, 'clipped_average', 'noisy_sigma2', 'noisy_sigma4', 'noisy_sigma8')


def summarize_variants(records, seeds, groups, expected_ids):
    if {r['variant'] for r in records} != set(VARIANTS): raise ValueError('All five matched variants required')
    results = {v: existing.summarize_records([r for r in records if r['variant'] == v], seeds, groups, expected_ids) for v in VARIANTS}
    references = {(r['seed'], r['group']): r for r in records if r['variant'] == REFERENCE}
    for row in records:
        reference = references[row['seed'], row['group']]
        if row['support'] != reference['support'] or not np.array_equal(row['ids'], reference['ids']) or not np.array_equal(row['y'], reference['y']):
            raise ValueError('Matched targets/IDs/support differ')
    paired = []
    baseline = {(r['seed'], r['panel']): r['metrics'] for r in results[REFERENCE]['per_seed']}
    for variant in VARIANTS[1:]:
        differences = []
        for row in results[variant]['per_seed']:
            reference = baseline[row['seed'], row['panel']]
            if (row['metrics'] is None) != (reference is None): raise ValueError('Matched panel availability differs')
            if reference is not None:
                differences.append({'seed': row['seed'], 'panel': row['panel'],
                    'equal_group_rmse_difference': row['metrics']['equal_group_rmse'] - reference['equal_group_rmse'],
                    'pooled_rmse_difference': row['metrics']['pooled']['rmse'] - reference['pooled']['rmse']})
        paired.append({'variant': variant, 'per_seed': differences,
            'all_equal_group_rmse_difference': existing.mean_sd([r['equal_group_rmse_difference'] for r in differences if r['panel'] == 'all']),
            'all_pooled_rmse_difference': existing.mean_sd([r['pooled_rmse_difference'] for r in differences if r['panel'] == 'all'])})
    return {'variants': results, 'paired_differences_from_cached_reference': paired,
            'difference_sign': 'positive means larger error than unnoised reference; descriptive only'}


def load_record(folder, run, variant, completion_hash, journal_hash):
    if sha(folder / 'completion.json') != completion_hash or sha(folder / 'attempts.json') != journal_hash:
        raise ValueError('Audited run changed')
    receipt = read(folder / 'completion.json')
    if any(sha(folder / n) != h for n, h in receipt['assets'].items()): raise ValueError('Audited asset changed')
    context = read(folder / 'partition_context.json')
    with np.load(folder / 'test_cache.npz', allow_pickle=False) as cache:
        arrays = {n: cache[n].copy() for n in ('y', 'prediction', 'ids')}
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
    return {'variant': variant, 'seed': run['seed'], 'group': context['test_group'], 'ids': arrays['ids'],
            'y': arrays['y'], 'p': arrays['prediction'], 'support': context['support'],
            'training_fit': checkpoint['history'][-1]['training_equal_client_fit']}


def summarize(output, package=runner.EXECUTION, study=runner.STUDY):
    _, config, plan, data = runner.verify_execution(package)
    output = runner.safe_output(output, package); study = Path(study).resolve()
    if output.exists(): raise FileExistsError(output)
    if output == study or output in study.parents or study in output.parents: raise ValueError('Separate analysis output required')
    report = auditor.audit(package, study, output / 'audit')
    if report['status'] != 'PASS_COMPLETE_MATCHED_STUDY' or report['verified'] != 160:
        raise ValueError('Complete160-run audit required for analysis')
    rows = {r['id']: r for r in read(output / 'audit/verification.json')['rows']}
    records = []
    with writer_lock(study):
        for run in plan:
            row = rows[run['id']]
            records.append(load_record(study / 'runs' / run['id'], run, run['variant'], row['completion_sha256'], row['journal_sha256']))
        ledger = read(study / 'shared_ledger.json')
    # The execution verifier has freshly audited and bound each referenced R13 asset.
    references = read(runner.protocol.PROTOCOL / 'references.json')
    with writer_lock(runner.common.STUDY):
        for row in references:
            records.append(load_record(Path(row['folder']), row['run'], REFERENCE,
                                       row['assets']['completion.json'], row['assets']['attempts.json']))
    groups = sorted({r['group'] for r in records})
    if len(groups) != 8 or len(references) != 40: raise ValueError('Complete reference geography matrix required')
    result = summarize_variants(records, config['seeds'], groups, range(read(data / 'manifest.json')['rows']))
    hypothetical = read(runner.protocol.PROTOCOL / 'hypothetical_accounting.json')
    if ledger['event_count'] != hypothetical['combined_events'] or ledger['per_slot'] != hypothetical['per_slot']:
        raise ValueError('Complete observed ledger differs from prospective budget')
    result.update(status='VERIFIED_COMPLETE_MATCHED_ANALYSIS', model_reinference=False,
        application_privacy_certified=False, execution_manifest_sha256=sha(Path(package) / 'manifest.json'),
        source_sha256={Path(p).name: sha(p) for p in (__file__, existing.__file__, existing.existing_summary.__file__)},
        ledger={'events': ledger['event_count'], 'prior_events': ledger['prior_event_count'], 'per_slot': ledger['per_slot']},
        scope='Public benchmark development; seeds are not independent regions; no inferential superiority or application privacy claim',
        interpretation='Equal-group primary, pooled secondary; raw t/ha; all support panels and adverse findings retained')
    atomic_json(output / 'analysis.json', result)
    return {'status': result['status'], 'new_runs': len(plan), 'cached_references': len(references),
            'noise_events': ledger['event_count'], 'model_reinference': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    print(summarize(args.output))
