"""Complete matched-seed/group descriptive analysis of verified cached predictions."""
import argparse
from pathlib import Path
import itertools
import numpy as np
from paper2.restart.validation_search import ROOT, read, sha, atomic_json, writer_lock
from paper2.restart.verify_outer_study import audit, scores


def aggregate(groups):
    if not groups:
        return None
    ids=np.concatenate([g['ids'] for g in groups])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Overlapping held-out rows')
    measured=[scores(g['y'],g['p']) for g in groups]
    return {'groups':len(groups),'equal_group_rmse':float(np.mean([m['rmse'] for m in measured])),
            'equal_group_mae':float(np.mean([m['mae'] for m in measured])),
            'pooled':scores(np.concatenate([g['y'] for g in groups]),np.concatenate([g['p'] for g in groups]))}


def summarize(output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if ROOT/'paper2/results' not in output.parents:
        raise ValueError('Use a new results directory')
    study=ROOT/'paper2/results/restart_outer_study_v1'
    audit_output=output/'audit'
    result=audit(ROOT/'paper2/results/restart_outer_execution_v1',study,audit_output)
    if result['status']!='PASS_COMPLETE':
        raise ValueError('Complete study required')
    verified=read(audit_output/'verification.json')
    plan=read(ROOT/'paper2/results/restart_outer_protocol_v1/plan.json')
    seeds=read(ROOT/'paper2/results/restart_outer_protocol_v1/config.json')['seeds']
    algorithms=sorted({r['algorithm'] for r in plan})
    panels=('all','species_seen','species_unseen','composition_seen','composition_unseen')
    buckets={}; regions=[]
    with writer_lock(study):
        for row in verified['rows']:
            folder=study/'runs'/row['id']
            if sha(folder/'completion.json')!=row['completion_sha256']:
                raise ValueError('Completion changed after audit')
            receipt=read(folder/'completion.json')
            if any(sha(folder/n)!=h for n,h in receipt['assets'].items()):
                raise ValueError('Cached asset changed after audit')
            context=read(folder/'partition_context.json')
            run=read(folder/'result.json')['run']
            fold=read(ROOT/'paper2/results/restart_regression_data_v1/folds'/run['fold']/'partition.json')
            if context['support']!=fold['support']['test']:
                raise ValueError('Support changed')
            with np.load(folder/'test_cache.npz',allow_pickle=False) as c:
                species=np.array([v['all_species_seen'] for v in context['support']],bool)
                composition=np.array([v['composition_seen'] for v in context['support']],bool)
                masks=(np.ones(len(c['y']),bool),species,~species,composition,~composition)
                for panel,mask in zip(panels,masks):
                    if mask.any():
                        g={'ids':c['ids'][mask].copy(),'y':c['y'][mask].copy(),'p':c['prediction'][mask].copy()}
                        buckets.setdefault((row['algorithm'],row['seed'],panel),[]).append(g)
                        regions.append({'algorithm':row['algorithm'],'seed':row['seed'],'group':row['group'],'panel':panel,'metrics':scores(g['y'],g['p'])})
    per_seed=[]; summary=[]
    for algorithm in algorithms:
        for panel in panels:
            values=[]
            for seed in seeds:
                groups=buckets.get((algorithm,seed,panel),[])
                if panel=='all' and len(groups)!=8:
                    raise ValueError('Missing geographic group')
                value=aggregate(groups)
                per_seed.append({'algorithm':algorithm,'seed':seed,'panel':panel,'metrics':value})
                if value is not None:
                    values.append(value)
            if values and len(values)!=len(seeds):
                raise ValueError('Incomplete seed panel')
            row={'algorithm':algorithm,'panel':panel,'seeds':len(values)}
            if values:
                row.update(samples_per_seed=values[0]['pooled']['samples'],groups_per_seed=values[0]['groups'])
                for metric in ('equal_group_rmse','equal_group_mae','pooled_rmse','pooled_mae','pooled_r2'):
                    nums=[v[metric] if metric.startswith('equal') else v['pooled'][metric.removeprefix('pooled_')] for v in values]
                    row[metric]=None if any(v is None for v in nums) else {'mean':float(np.mean(nums)),'seed_sd':float(np.std(nums,ddof=1))}
            summary.append(row)
    paired=[]
    for a,b in itertools.combinations(algorithms,2):
        differences=[next(v['metrics']['equal_group_rmse'] for v in per_seed if v['algorithm']==a and v['seed']==s and v['panel']=='all')-
                     next(v['metrics']['equal_group_rmse'] for v in per_seed if v['algorithm']==b and v['seed']==s and v['panel']=='all') for s in seeds]
        paired.append({'a':a,'b':b,'a_minus_b_rmse_by_seed':differences,'mean_difference':float(np.mean(differences))})
    atomic_json(output/'analysis.json',{'status':'COMPLETE_DESCRIPTIVE_ANALYSIS','verified_runs':320,'summary':summary,
        'per_seed':per_seed,'per_region':regions,'paired_comparisons':paired,
        'uncertainty':'Seed SD measures initialization variability, not geographic confidence. No p-values or independence claim.',
        'panel_policy':'Equal weight among nonempty geographic groups within each panel; empty panels null.',
        'audit_sha256':sha(audit_output/'verification.json'),'analysis_code_sha256':sha(__file__)})
    return [r for r in summary if r['panel']=='all']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    import json
    print(json.dumps(summarize(p.parse_args().output),indent=2))
