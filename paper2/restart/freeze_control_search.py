"""Prospectively freeze expanded R12 search; no training or outer inference."""
import argparse
from pathlib import Path
import shutil
from paper2.restart.validation_search import ROOT,read,sha,atomic_json,runtime
from paper2.restart.prepare_public_data import verify,new_output
from paper2.restart.control_convergence import CODE


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=ROOT/'paper2/results/restart_public_data_v1';verified=verify(data);output=new_output(args.output,data)
    config_path=ROOT/'paper2/configs/restart_control_search_v1.json';config=read(config_path)
    plan=[]
    for fold in sorted(n for n in read(data/'manifest.json')['assets'] if '_inner_' in n):
        for scale in config['target_scales']:
            for lr in config['learning_rates']:
                plan.append({'id':fold+'_scale'+str(scale)+'_lr'+str(lr),'fold':fold,'target_scale':scale,
                    'learning_rate':lr,'rounds':config['rounds'],'report_rounds':config['report_rounds'],'seed':config['seed']})
    if len(plan)!=224 or len({r['id'] for r in plan})!=224:raise ValueError('Expected 224 unique candidates')
    output.mkdir(parents=True);(output/'source_snapshot').mkdir()
    sources=[*CODE,Path(__file__)]
    for source in sources:shutil.copyfile(source,output/'source_snapshot'/source.name)
    atomic_json(output/'config.json',config);atomic_json(output/'plan.json',plan)
    atomic_json(output/'manifest.json',{'status':'FROZEN_EXPANDED_SEARCH_NO_TRAINING','planned':len(plan),
        'data':str(data),'data_manifest_sha256':verified['manifest_sha256'],'runtime':runtime(),
        'config_sha256':sha(output/'config.json'),'plan_sha256':sha(output/'plan.json'),
        'source_sha256':{p.name:sha(p) for p in sources},
        'execution_runner':'Separate capped coordinator and final selection tests still required before execution'})
    print({'status':'FROZEN_EXPANDED_SEARCH_NO_TRAINING','planned':len(plan)})
