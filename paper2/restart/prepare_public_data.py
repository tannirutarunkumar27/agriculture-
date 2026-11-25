"""Versioned fixed-transform tensors on existing audited geographic splits."""
import argparse
import csv
from pathlib import Path
import shutil
import numpy as np
from paper2.restart.public_preprocessing import schema,transform_records,SCALES,SPECIES
from paper2.restart.validation_search import ROOT,read,sha,identity,atomic_json,runtime
from paper2.restart.outer_runner import verify_runner

BASE=ROOT/'paper2/results/restart_outer_execution_v1'


def load_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for name in [*SCALES,'target_seed_dm_t_ha',*['contains_'+s for s in SPECIES],'is_intercrop']:
            row[name]=None if row[name]=='' else float(row[name])
        for name in ('previous_crop','crop_season','pest_control'):
            if row[name]=='':row[name]=None
    if not rows or len({r['row_id'] for r in rows})!=len(rows):
        raise ValueError('Unique nonempty regression rows required')
    return rows


def new_output(output,parent):
    output=Path(output).resolve()
    if ROOT/'paper2/results' not in output.parents or output==parent or parent in output.parents or output in parent.parents:
        raise ValueError('Use a separate results directory')
    if output.exists():raise FileExistsError(output)
    return output


def prepare(output):
    _,parent,_,_=verify_runner(BASE)
    source=Path(parent['data']);output=new_output(output,source)
    rows=load_rows(source/'regression_rows.csv');original=read(source/'manifest.json')
    definition=schema();state={**definition,'identity':identity(definition)}
    output.mkdir(parents=True)
    shutil.copyfile(source/'regression_rows.csv',output/'regression_rows.csv')
    assets={};summary=[]
    for fold_id in original['assets']:
        parent_fold=source/'folds'/fold_id;folder=output/'folds'/fold_id;folder.mkdir(parents=True)
        fold=read(parent_fold/'partition.json');fold['preprocessing_identity']=state['identity']
        names=('train','validation') if fold['kind']=='inner_selection' else ('train','test')
        arrays={}
        for split in names:
            ids=fold[split];selected=[rows[i] for i in ids]
            arrays['x_'+split]=transform_records(selected)
            arrays['y_'+split]=np.array([r['target_seed_dm_t_ha'] for r in selected],np.float32)
            arrays['ids_'+split]=np.array(ids,np.int64)
        with np.load(parent_fold/'arrays.npz',allow_pickle=False) as old:
            for split in names:
                np.testing.assert_array_equal(arrays['y_'+split],old['y_'+split])
                np.testing.assert_array_equal(arrays['ids_'+split],old['ids_'+split])
        np.savez_compressed(folder/'arrays.npz',**arrays)
        atomic_json(folder/'partition.json',fold);atomic_json(folder/'preprocessing.json',state)
        assets[fold_id]={n:sha(folder/n) for n in ('arrays.npz','partition.json','preprocessing.json')}
        summary.append({'fold':fold_id,'kind':fold['kind'],'features':54,'rows':{n:len(fold[n]) for n in names}})
    atomic_json(output/'manifest.json',{'version':'fixed-public-data-v1','status':'PREPARED_NO_TRAINING',
        'parent_data':str(source),'parent_manifest_sha256':sha(source/'manifest.json'),
        'rows_sha256':sha(output/'regression_rows.csv'),'schema_identity':state['identity'],
        'rows':len(rows),'folds':len(assets),'assets':assets,'summary':summary,'runtime':runtime(),
        'source_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('public_preprocessing.py'))},
        'public_transform':True,'application_privacy_certified':False})
    return {'status':'PREPARED_NO_TRAINING','rows':len(rows),'folds':len(assets),'features':54}


def verify(data):
    data=Path(data).resolve();m=read(data/'manifest.json');_,parent,_,_=verify_runner(BASE)
    source=Path(parent['data']);orig=read(source/'manifest.json')
    if m['parent_data']!=str(source) or sha(source/'manifest.json')!=m['parent_manifest_sha256'] or runtime()!=m['runtime']:
        raise ValueError('Public-data parent/runtime changed')
    for name,expected in m['source_sha256'].items():
        if sha(Path(__file__).with_name(name))!=expected:raise ValueError('Public preparation code changed')
    if sha(data/'regression_rows.csv')!=m['rows_sha256'] or m['rows_sha256']!=orig['rows_sha256']:
        raise ValueError('Row table changed')
    if set(m['assets'])!=set(orig['assets']):raise ValueError('Fold set changed')
    rows=load_rows(data/'regression_rows.csv');state={**schema(),'identity':identity(schema())};outer=[]
    if m['schema_identity']!=state['identity']:raise ValueError('Public schema changed')
    for fold_id,hashes in m['assets'].items():
        folder=data/'folds'/fold_id
        if any(sha(folder/n)!=expected for n,expected in hashes.items()):raise ValueError('Public fold changed')
        fold=read(folder/'partition.json');parent_fold=read(source/'folds'/fold_id/'partition.json')
        parent_fold['preprocessing_identity']=state['identity']
        if fold!=parent_fold or read(folder/'preprocessing.json')!=state:raise ValueError('Fold/schema mismatch')
        names=('train','validation') if fold['kind']=='inner_selection' else ('train','test')
        with np.load(folder/'arrays.npz',allow_pickle=False) as a:
            if set(a.files)!={prefix+n for prefix in ('x_','y_','ids_') for n in names}:raise ValueError('Unexpected arrays/test exposure')
            for split in names:
                np.testing.assert_array_equal(a['ids_'+split],fold[split])
                np.testing.assert_array_equal(a['x_'+split],transform_records([rows[i] for i in fold[split]]))
                np.testing.assert_array_equal(a['y_'+split],np.array([rows[i]['target_seed_dm_t_ha'] for i in fold[split]],np.float32))
        if fold['kind']=='outer_refit':outer.extend(fold['test'])
    if sorted(outer)!=list(range(len(rows))):raise ValueError('Outer coverage mismatch')
    return {'status':'PASS','verified_folds':len(m['assets']),'rows':len(rows),'features':54,
        'no_train_statistics':True,'manifest_sha256':sha(data/'manifest.json'),'training_performed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','verify']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'paper2/results/restart_public_data_v1');args=p.parse_args()
    if args.action=='prepare':print(prepare(args.output))
    else:
        result=verify(args.data)
        if args.output.exists():raise FileExistsError(args.output)
        args.output.mkdir(parents=True);atomic_json(args.output/'verification.json',result);print(result)
