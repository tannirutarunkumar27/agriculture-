"""Versioned post-hoc sensitivity using fixed models and a training-only gate."""
import argparse
import io
from pathlib import Path
import shutil
import unittest
import numpy as np
import torch
from paper2.restart.validation_search import ROOT, read, sha, identity, atomic_json, writer_lock, runtime
from paper2.restart.outer_runner import verify_runner, load_fold
from paper2.restart.outer_training import evaluate_once, verify_evaluation
from paper2.restart.verify_outer_study import scores
from paper2.restart.summarize_outer import aggregate
from paper2.restart.support_guard import fit_gate, apply_gate

BASE = ROOT/'paper2/results/restart_outer_execution_v1'
STUDY = ROOT/'paper2/results/restart_outer_study_v1'
CODE = [Path(__file__), Path(__file__).with_name('support_guard.py'),
        Path(__file__).with_name('verify_outer_study.py'), Path(__file__).with_name('summarize_outer.py')]


def safe_output(output, protected):
    output=Path(output).resolve()
    if ROOT/'paper2/results' not in output.parents or any(output==p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Use a separate results directory')
    return output


def freeze(output):
    output=safe_output(output,(BASE,STUDY))
    record,manifest,config,plan=verify_runner(BASE)
    if output.exists(): raise FileExistsError(output)
    data=Path(manifest['data']); gates={}; completions={}
    with writer_lock(STUDY):
        for fold in sorted({r['fold'] for r in plan}):
            _,a=load_fold(data,{'fold':fold})
            mask=fit_gate(a['x_train'])
            np.testing.assert_array_equal(apply_gate(a['x_train'],mask),a['x_train'])
            gates[fold]=mask.tolist()
        for run in plan:
            folder=STUDY/'runs'/run['id']
            verify_evaluation(folder,identity({'binding':identity(record),'run':run}))
            completions[run['id']]=sha(folder/'completion.json')
        output.mkdir(parents=True)
        (output/'source_snapshot').mkdir()
        for p in CODE: shutil.copyfile(p,output/'source_snapshot'/p.name)
        atomic_json(output/'gates.json',gates)
        atomic_json(output/'plan.json',plan)
        atomic_json(output/'manifest.json',{'version':'zero-support-posthoc-v1','parent':str(BASE),
            'parent_manifest_sha256':sha(BASE/'manifest.json'),'runtime':runtime(),
            'code_sha256':{p.name:sha(p) for p in CODE},'gates_sha256':sha(output/'gates.json'),
            'plan_sha256':sha(output/'plan.json'),'parent_completions':completions,
            'policy':'Gate all identically-zero training columns to zero for every inference input.',
            'training':'Reuse locally verified fixed models; training inputs unchanged exactly.',
            'scope':'POST_HOC_SENSITIVITY; informed by exposed outer errors, no independent confirmation.',
            'hyperparameters':'unchanged; no retuning; no outcome-based gate selection'})
    return {'status':'FROZEN_POST_HOC_SENSITIVITY','planned':len(plan)}


def verify(package):
    package=Path(package).resolve(); m=read(package/'manifest.json')
    if m['runtime']!=runtime() or sha(BASE/'manifest.json')!=m['parent_manifest_sha256']:
        raise ValueError('Sensitivity parent/runtime changed')
    for p in CODE:
        if sha(p)!=m['code_sha256'][p.name] or sha(package/'source_snapshot'/p.name)!=sha(p):
            raise ValueError('Sensitivity code changed')
    if sha(package/'gates.json')!=m['gates_sha256'] or sha(package/'plan.json')!=m['plan_sha256']:
        raise ValueError('Sensitivity artifacts changed')
    record,manifest,config,plan=verify_runner(BASE)
    if plan!=read(package/'plan.json'): raise ValueError('Sensitivity plan changed')
    data=Path(manifest['data']); gates=read(package/'gates.json')
    for name,stored in gates.items():
        _,a=load_fold(data,{'fold':name})
        if fit_gate(a['x_train']).tolist()!=stored: raise ValueError('Gate not training-derived')
    return m,record,config,plan,data,gates


def execute(package,output,cap=1,resume=False,dry_run=True):
    package=Path(package).resolve(); output=safe_output(output,(BASE,STUDY,package))
    if type(cap) is not int or cap<1: raise ValueError('Positive cap required')
    m,record,config,plan,data,gates=verify(package)
    if dry_run: return {'status':'DRY_RUN','planned':len(plan)}
    if output.exists() and not resume: raise ValueError('Explicit resume required')
    if not output.exists(): output.mkdir(parents=True)
    binding=identity(m); done=0; executed=0
    torch.set_num_threads(1)
    with writer_lock(output),writer_lock(STUDY):
        if (output/'binding.json').exists():
            if read(output/'binding.json')['identity']!=binding: raise ValueError('Sensitivity binding changed')
        elif resume or any(p.name!='writer.lock' for p in output.iterdir()):
            raise ValueError('Unbound sensitivity output')
        else: atomic_json(output/'binding.json',{'identity':binding})
        for run in plan:
            folder=output/'runs'/run['id']; source=STUDY/'runs'/run['id']
            token=identity({'binding':binding,'run':run})
            if (folder/'completion.json').exists():
                verify_evaluation(folder,token); done+=1; continue
            if executed>=cap: continue
            if (folder/'inference_started.json').exists() or (folder/'failure.json').exists():
                raise RuntimeError('Inspect ambiguous or failed sensitivity run before proceeding')
            if sha(source/'completion.json')!=m['parent_completions'][run['id']]: raise ValueError('Parent result changed')
            verify_evaluation(source,identity({'binding':identity(record),'run':run}))
            fold,a=load_fold(data,run); mask=np.array(gates[run['fold']],bool)
            np.testing.assert_array_equal(apply_gate(a['x_train'],mask),a['x_train'])
            folder.mkdir(parents=True,exist_ok=True)
            fitted=read(source/'fit.json'); model=fitted['model']
            shutil.copyfile(source/model,folder/model)
            atomic_json(folder/'fit.json',dict(fitted,identity=token))
            atomic_json(folder/'provenance.json',{'identity':token,'parent_completion_sha256':m['parent_completions'][run['id']],
                'copied_model_sha256':sha(folder/model),'training_unchanged':True,'gate':gates[run['fold']]})
            try:
                evaluate_once(folder,run,config['neural_settings'],apply_gate(a['x_test'],mask),a['y_test'],a['ids_test'],token)
            except Exception:
                import traceback
                atomic_json(folder/'failure.json',{'traceback':traceback.format_exc()}); raise
            done+=1; executed+=1
        result={'status':'COMPLETE' if done==len(plan) else 'PARTIAL','completed':done,'planned':len(plan),
                'pending':len(plan)-done,'executed':executed,'training_performed':False}
        atomic_json(output/'progress.json',result)
    return result


def analyze(package,study,output):
    package,study=Path(package).resolve(),Path(study).resolve()
    output=safe_output(output,(BASE,STUDY,package,study))
    if output.exists(): raise FileExistsError(output)
    m,record,config,plan,data,gates=verify(package); binding=identity(m)
    if read(study/'binding.json')['identity']!=binding: raise ValueError('Study binding changed')
    buckets={}; region=[]
    with writer_lock(study):
        for run in plan:
            folder=study/'runs'/run['id']; token=identity({'binding':binding,'run':run})
            result=verify_evaluation(folder,token)
            if result['run']!=run: raise ValueError('Run mismatch')
            source=STUDY/'runs'/run['id']; fitted=read(folder/'fit.json')
            if sha(source/'completion.json')!=m['parent_completions'][run['id']]: raise ValueError('Parent completion changed')
            verify_evaluation(source,identity({'binding':identity(record),'run':run}))
            if sha(folder/fitted['model'])!=read(source/'fit.json')['model_sha256']: raise ValueError('Reused model mismatch')
            if read(folder/'provenance.json') != {'identity':token,'parent_completion_sha256':m['parent_completions'][run['id']],
                'copied_model_sha256':sha(folder/fitted['model']),'training_unchanged':True,'gate':gates[run['fold']]}:
                raise ValueError('Guarded model provenance changed')
            fold,a=load_fold(data,run)
            with np.load(folder/'test_cache.npz',allow_pickle=False) as c:
                np.testing.assert_array_equal(c['ids'],a['ids_test']);np.testing.assert_array_equal(c['y'],a['y_test'])
                for name,mask in [('all',np.ones(len(c['y']),bool)),
                    ('species_seen',np.array([s['all_species_seen'] for s in fold['support']['test']],bool)),
                    ('species_unseen',~np.array([s['all_species_seen'] for s in fold['support']['test']],bool))]:
                    if mask.any():
                        g={'ids':c['ids'][mask].copy(),'y':c['y'][mask].copy(),'p':c['prediction'][mask].copy()}
                        buckets.setdefault((run['algorithm'],run['seed'],name),[]).append(g)
                region.append({'id':run['id'],'group':fold['test_group'],'metrics':scores(c['y'],c['prediction'])})
        per_seed=[];summary=[]
        for algorithm in sorted({r['algorithm'] for r in plan}):
            for panel in ('all','species_seen','species_unseen'):
                values=[]
                for seed in config['seeds']:
                    groups=buckets.get((algorithm,seed,panel),[])
                    if panel=='all' and len(groups)!=8: raise ValueError('Incomplete geographic coverage')
                    v=aggregate(groups);values.append(v);per_seed.append({'algorithm':algorithm,'seed':seed,'panel':panel,'metrics':v})
                if any(v is None for v in values): raise ValueError('Unexpected empty panel')
                summary.append({'algorithm':algorithm,'panel':panel,'samples':values[0]['pooled']['samples'],
                    'equal_group_rmse_mean':float(np.mean([v['equal_group_rmse'] for v in values])),
                    'equal_group_rmse_seed_sd':float(np.std([v['equal_group_rmse'] for v in values],ddof=1)),
                    'pooled_rmse_mean':float(np.mean([v['pooled']['rmse'] for v in values]))})
        output.mkdir(parents=True)
        atomic_json(output/'analysis.json',{'status':'PASS_COMPLETE_POST_HOC_SENSITIVITY','verified':len(plan),
            'scope':m['scope'],'summary':summary,'per_seed':per_seed,'per_region':region,
            'inference_repeated_during_audit':False,'training_performed':False})
    return [r for r in summary if r['panel']=='all']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','execute','analyze','tests']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path,default=ROOT/'paper2/results/restart_guard_protocol_v1')
    p.add_argument('--study',type=Path,default=ROOT/'paper2/results/restart_guard_study_v1')
    p.add_argument('--cap',type=int,default=1);p.add_argument('--resume',action='store_true');p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.action=='freeze': print(freeze(args.output))
    elif args.action=='execute': print(execute(args.package,args.output,args.cap,args.resume,not args.execute))
    elif args.action=='analyze': print(analyze(args.package,args.study,args.output))
    else:
        if args.output.exists(): raise FileExistsError(args.output)
        args.output.mkdir(parents=True);stream=io.StringIO()
        suite=unittest.defaultTestLoader.loadTestsFromName('paper2.restart.test_support_guard')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (args.output/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
        atomic_json(args.output/'verification.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
            'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)})
        print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
