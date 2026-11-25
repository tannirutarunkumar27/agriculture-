"""Complete fixed-public unnoised refit analysis; reuse tested aggregation unchanged."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paper2.restart import public_refit_runner as runner
from paper2.restart import verify_public_refit as audit_module
from paper2.restart import summarize_outer as existing_summary
from paper2.restart.validation_search import read, sha, atomic_json, writer_lock

PANELS = ('all', 'species_seen', 'species_unseen', 'composition_seen', 'composition_unseen')


def mean_sd(values):
    if any(v is None for v in values): return None
    return {'mean': float(np.mean(values)), 'seed_sd': float(np.std(values, ddof=1))}


def summarize_records(records, seeds, groups, expected_ids):
    """Complete coverage per seed; repeated seed predictions are not new observations."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or not groups or len(set(groups)) != len(groups):
        raise ValueError('Distinct seeds and geographic groups required')
    expected = {(seed, group) for seed in seeds for group in groups}
    actual = [(r['seed'], r['group']) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Complete unique seed/group matrix required')
    expected_ids = sorted(expected_ids)
    if len(expected_ids) != len(set(expected_ids)): raise ValueError('Unique expected row coverage required')
    buckets = {}; regional = []; fits = []
    for seed in seeds:
        rows = [r for r in records if r['seed'] == seed]
        ids = np.concatenate([r['ids'] for r in rows])
        if sorted(ids.tolist()) != expected_ids: raise ValueError('Seed held-out row coverage differs')
        for r in rows:
            y, p = r['y'], r['p']
            if len(r['ids']) != len(y) or len(r['support']) != len(y): raise ValueError('Cached support/row alignment differs')
            if any(type(v[n]) is not bool for v in r['support'] for n in ('all_species_seen', 'composition_seen')):
                raise ValueError('Boolean support metadata required')
            species = np.array([v['all_species_seen'] for v in r['support']], bool)
            composition = np.array([v['composition_seen'] for v in r['support']], bool)
            masks = (np.ones(len(y), bool), species, ~species, composition, ~composition)
            for panel, mask in zip(PANELS, masks):
                if mask.any():
                    selected = {'ids': r['ids'][mask], 'y': y[mask], 'p': p[mask]}
                    buckets.setdefault((seed, panel), []).append(selected)
                    regional.append({'seed': seed, 'group': r['group'], 'panel': panel,
                                     'metrics': existing_summary.scores(selected['y'], selected['p'])})
            fits.append({'seed': seed, 'group': r['group'], 'saved_training_fit': r['training_fit']})
    per_seed = []; summaries = []
    for panel in PANELS:
        values = []
        for seed in seeds:
            value = existing_summary.aggregate(buckets.get((seed, panel), []))
            per_seed.append({'seed': seed, 'panel': panel, 'metrics': value})
            if value is not None: values.append(value)
        if values and len(values) != len(seeds): raise ValueError('Panel availability differs across seeds')
        row = {'panel': panel, 'available_seeds': len(values), 'metrics': None}
        if values:
            counts = {(v['groups'], v['pooled']['samples']) for v in values}
            if len(counts) != 1: raise ValueError('Panel support counts differ across seeds')
            row.update(groups_per_seed=values[0]['groups'], samples_per_seed=values[0]['pooled']['samples'])
            row['metrics'] = {name: mean_sd([v[name] if name.startswith('equal_group') else v['pooled'][name.removeprefix('pooled_')] for v in values])
                              for name in ('equal_group_rmse', 'equal_group_mae', 'pooled_rmse', 'pooled_mae', 'pooled_r2')}
        summaries.append(row)
    by_group = []
    for group in groups:
        measured = [r['metrics'] for r in regional if r['group'] == group and r['panel'] == 'all']
        by_group.append({'group': group, 'samples_per_seed': measured[0]['samples'], 'seeds': len(measured),
                         'metrics': {n: mean_sd([m[n] for m in measured]) for n in ('rmse', 'mae', 'r2')}})
    return {'per_seed': per_seed, 'summary': summaries, 'regional_panels': regional, 'by_group': by_group,
            'saved_training_fits': fits,
            'worse_than_matched_training_constant': sum(r['saved_training_fit']['worse_than_matched_constant'] for r in fits),
            'unique_systems_per_seed': len(expected_ids), 'geographic_groups': len(groups), 'study_seeds': len(seeds)}


def summarize(output, package=runner.EXECUTION, study=runner.STUDY):
    package, study = Path(package).resolve(), Path(study).resolve()
    m, config, plan, data = runner.verify_execution(package)
    output = runner.safe_output(output, (package, runner.PROTOCOL, data, study, runner.protocol.integrity.STUDY))
    if output.exists(): raise FileExistsError(output)
    audited = audit_module.audit(package, study, output / 'audit')
    if audited['status'] != 'PASS_COMPLETE_REFIT': raise ValueError('Complete40-run audit required before analysis')
    verified = read(output / 'audit/verification.json'); receipts = {r['id']: r for r in verified['rows']}
    records = []
    with writer_lock(study):
        for run in plan:
            folder = study / 'runs' / run['id']; row = receipts[run['id']]
            if sha(folder / 'completion.json') != row['completion_sha256'] or sha(folder / 'attempts.json') != row['journal_sha256']:
                raise ValueError('Verified run changed after audit')
            receipt = read(folder / 'completion.json')
            if any(sha(folder / n) != h for n, h in receipt['assets'].items()): raise ValueError('Verified assets changed')
            context = read(folder / 'partition_context.json')
            with np.load(folder / 'test_cache.npz', allow_pickle=False) as cache:
                arrays = {n: cache[n].copy() for n in ('y', 'prediction', 'ids')}
            checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            records.append({'seed': run['seed'], 'group': context['test_group'], 'ids': arrays['ids'], 'y': arrays['y'],
                            'p': arrays['prediction'], 'support': context['support'],
                            'training_fit': checkpoint['history'][-1]['training_equal_client_fit']})
        groups = sorted({r['group'] for r in verified['rows']})
        if len(groups) != 8 or len(plan) != 40: raise ValueError('Frozen complete study dimensions differ')
        result = summarize_records(records, config['seeds'], groups, range(read(data / 'manifest.json')['rows']))
    result.update(status='VERIFIED_COMPLETE_REFIT_ANALYSIS', model_reinference=False, privacy_claim=False,
                  execution_manifest_sha256=sha(package / 'manifest.json'),
                  source_sha256={Path(__file__).name: sha(__file__), 'summarize_outer.py': sha(existing_summary.__file__)},
                  scope='Development descriptive summary of one unnoised control; seeds are not independent regional replications',
                  training_diagnostic='Saved final-training metrics, not newly inferred training predictions',
                  interpretation='Equal-group primary and pooled secondary; no test-based tuning, inferential superiority or privacy claim')
    atomic_json(output / 'analysis.json', result)
    return {k: result[k] for k in ('status', 'unique_systems_per_seed', 'geographic_groups', 'study_seeds', 'worse_than_matched_training_constant', 'model_reinference')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(summarize(args.output))
