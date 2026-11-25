"""Scientific plots sourced from fresh complete cached analyses, never partial runs."""
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper2.restart import summarize_public_refit as refit
from paper2.restart import summarize_matched_noise as matched
from paper2.restart.validation_search import ROOT, read, sha, atomic_json

LABELS = {'equal_average_reference': 'Unnoised reference', 'clipped_average': 'Clipped control',
          'noisy_sigma2': 'Noise multiplier 2', 'noisy_sigma4': 'Noise multiplier 4', 'noisy_sigma8': 'Noise multiplier 8'}
COLORS = ['#253858', '#17806D', '#BB6029', '#8261A8', '#C44357']


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative RMSE/SD required')
    return value


def extract(analysis, kind):
    if kind == 'refit':
        if analysis['status'] != 'VERIFIED_COMPLETE_REFIT_ANALYSIS' or analysis['privacy_claim'] is not False:
            raise ValueError('Complete unnoised analysis required')
        variants = {'equal_average_reference': analysis}
    elif kind == 'matched':
        if analysis['status'] != 'VERIFIED_COMPLETE_MATCHED_ANALYSIS' or analysis['application_privacy_certified'] is not False:
            raise ValueError('Complete simulation analysis required')
        variants = analysis['variants']
        if set(variants) != set(matched.VARIANTS): raise ValueError('All matched variants required')
    else: raise ValueError('Unknown study kind')
    result = {'kind': kind, 'variants': [], 'error_bar': 'descriptive standard deviation across five seeds, not confidence interval',
              'scope': 'Public benchmark development; no independent field or privacy validation'}
    for name in LABELS:
        if name not in variants: continue
        value = variants[name]
        if (value['unique_systems_per_seed'], value['geographic_groups'], value['study_seeds']) != (607, 8, 5):
            raise ValueError('Complete607-system eight-group five-seed study required')
        panels = {row['panel']: row for row in value['summary']}
        if len(panels) != 5: raise ValueError('Complete support summaries required')
        overview = []; support = []
        for panel in ('all', 'species_seen', 'species_unseen', 'composition_seen', 'composition_unseen'):
            row = panels[panel]
            if row['metrics'] is None or row['available_seeds'] != 5: raise ValueError('Complete support panel required')
            for metric in ('equal_group_rmse', 'pooled_rmse'):
                measured = row['metrics'][metric]
                means = [r['metrics']['equal_group_rmse'] if metric == 'equal_group_rmse' else r['metrics']['pooled']['rmse']
                         for r in value['per_seed'] if r['panel'] == panel]
                if len(means) != 5: raise ValueError('Five seed measurements required')
                for score in means: finite(score)
                if not np.isclose(measured['mean'], np.mean(means), rtol=1e-12, atol=1e-12) or not np.isclose(measured['seed_sd'], np.std(means, ddof=1), rtol=1e-12, atol=1e-12):
                    raise ValueError('Plot summary differs from seed values')
                record = {'panel': panel, 'metric': metric, 'mean': finite(measured['mean']),
                          'seed_sd': finite(measured['seed_sd']), 'seed_values': means,
                          'samples': row['samples_per_seed'], 'groups': row['groups_per_seed']}
                (overview if panel == 'all' else support).append(record)
        groups = value['by_group']
        if len(groups) != 8 or len({r['group'] for r in groups}) != 8 or sum(r['samples_per_seed'] for r in groups) != 607:
            raise ValueError('Geographic support differs')
        geographic = []
        for row in groups:
            measured = row['metrics']['rmse']
            if row['seeds'] != 5: raise ValueError('Incomplete geographic seeds')
            geographic.append({'group': row['group'], 'samples': row['samples_per_seed'],
                               'mean': finite(measured['mean']), 'seed_sd': finite(measured['seed_sd'])})
        result['variants'].append({'name': name, 'label': LABELS[name], 'overview': overview, 'geographic': geographic, 'support': support})
    reference = result['variants'][0]
    for variant in result['variants'][1:]:
        if [(r['group'], r['samples']) for r in variant['geographic']] != [(r['group'], r['samples']) for r in reference['geographic']]:
            raise ValueError('Matched geographic labels/counts differ')
        if [(r['panel'], r['samples'], r['groups']) for r in variant['support']] != [(r['panel'], r['samples'], r['groups']) for r in reference['support']]:
            raise ValueError('Matched support counts differ')
    return result


def draw(inputs, output):
    output = Path(output); variants = inputs['variants']; count = len(variants)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    artifacts = []
    def save(fig, name, caption):
        for extension in ('png', 'pdf', 'svg'):
            path = output / f'{name}.{extension}'
            fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
            artifacts.append(path.name)
        plt.close(fig)
        return {'name': name, 'caption': caption}
    captions = []
    overview_max = max(max(r['seed_values'] + [r['mean'] + r['seed_sd']]) for v in variants for r in v['overview'])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, max(2.8, count * .55 + 1.2)), layout='constrained', sharey=True)
    for ax, metric, title in zip(axes, ('equal_group_rmse', 'pooled_rmse'), ('Equal geographic groups', 'Pooled systems')):
        for i, variant in enumerate(variants):
            row = next(r for r in variant['overview'] if r['metric'] == metric)
            ax.scatter(row['seed_values'], np.linspace(i - .10, i + .10, 5), color=COLORS[i], alpha=.45, s=15)
            ax.errorbar(row['mean'], i, xerr=row['seed_sd'], fmt='D', color=COLORS[i], capsize=4, markersize=5)
        ax.set_title(title); ax.set_xlabel('RMSE (t/ha)'); ax.set_xlim(0, max(.1, overview_max * 1.08)); ax.grid(axis='x', alpha=.18)
        ax.set_yticks(range(count), [v['label'] for v in variants])
    axes[0].set_ylim(count - .5, -.5)
    fig.suptitle('Held-out prediction error | 607 systems, 8 groups, 5 seeds', fontsize=11)
    captions.append(save(fig, 'rmse_overview', 'Dots show the five seed estimates; diamonds and bars show mean and descriptive seed SD. Equal-group RMSE weights geographic groups equally; pooled RMSE weights held-out systems equally. Seed SD is not a regional confidence interval.'))

    fig, ax = plt.subplots(figsize=(8.2, 5.2), layout='constrained')
    geography = variants[0]['geographic']
    for i, variant in enumerate(variants):
        offset = (i - (count - 1) / 2) * (.65 / max(count, 1))
        rows = variant['geographic']
        ax.errorbar([r['mean'] for r in rows], np.arange(8) + offset, xerr=[r['seed_sd'] for r in rows],
                    fmt='o', color=COLORS[i], capsize=2, markersize=4, label=variant['label'])
    ax.set_yticks(range(8), [f"{r['group']}  (n={r['samples']})" for r in geography]); ax.invert_yaxis()
    ax.set_xlabel('Held-out RMSE (t/ha), mean ± seed SD'); ax.set_xlim(left=0); ax.grid(axis='x', alpha=.18)
    ax.set_title('Geographic performance and unequal sample support')
    if count > 1: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    captions.append(save(fig, 'geographic_rmse', 'Each geographic holdout is shown with its system count. Error bars are descriptive SD over five seeds; small holdouts remain visible and are not evidence of precise regional generalization.'))

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.5), layout='constrained')
    names = ('species_seen', 'species_unseen', 'composition_seen', 'composition_unseen')
    support_max = max(r['mean'] + r['seed_sd'] for v in variants for r in v['support'] if r['metric'] == 'equal_group_rmse')
    for ax, panel in zip(axes.flat, names):
        row0 = next(r for r in variants[0]['support'] if r['panel'] == panel and r['metric'] == 'equal_group_rmse')
        for i, variant in enumerate(variants):
            row = next(r for r in variant['support'] if r['panel'] == panel and r['metric'] == 'equal_group_rmse')
            ax.errorbar(i, row['mean'], yerr=row['seed_sd'], fmt='o', color=COLORS[i], capsize=3)
        ax.set_title(f"{panel.replace('_', ' ').capitalize()}\nn={row0['samples']}, groups={row0['groups']}")
        ax.set_xticks(range(count), [str(i + 1) for i in range(count)]); ax.set_ylim(0, max(.1, support_max * 1.1))
        ax.set_ylabel('Equal-group RMSE (t/ha)'); ax.grid(axis='y', alpha=.18)
    fig.supxlabel('   |   '.join(f"{i + 1}: {v['label']}" for i, v in enumerate(variants)), fontsize=8)
    captions.append(save(fig, 'support_rmse', 'Support is defined using each fold\'s training data. Means and bars show descriptive seed SD. Unseen species/composition systems are confined to Auzeville (one group); this is not unseen-system validation across eight regions.'))
    return artifacts, captions


def generate(output, kind):
    output = matched.runner.safe_output(output)
    if output.exists(): raise FileExistsError(output)
    # Each summary performs a new cached audit. Training and inference are never invoked.
    if kind == 'refit': refit.summarize(output / 'verified_analysis')
    elif kind == 'matched': matched.summarize(output / 'verified_analysis')
    else: raise ValueError('Unknown study kind')
    source = output / 'verified_analysis/analysis.json'; inputs = extract(read(source), kind)
    atomic_json(output / 'plot_values.json', inputs)
    artifacts, captions = draw(inputs, output)
    atomic_json(output / 'captions.json', captions)
    atomic_json(output / 'manifest.json', {'status': 'GENERATED_FROM_COMPLETE_AUDITED_ANALYSIS', 'study': kind,
        'analysis_sha256': sha(source), 'code_sha256': sha(__file__), 'matplotlib': matplotlib.__version__,
        'assets': {n: sha(output / n) for n in artifacts + ['plot_values.json', 'captions.json']},
        'visual_review': 'pending', 'model_reinference': False})
    return {'status': 'GENERATED_FROM_COMPLETE_AUDITED_ANALYSIS', 'plots': 3, 'formats': ['png', 'pdf', 'svg']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--study', choices=('refit', 'matched'), required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args(); print(generate(args.output, args.study))
