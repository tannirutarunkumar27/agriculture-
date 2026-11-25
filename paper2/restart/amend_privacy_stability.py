"""Record failed-v1 evidence and freeze matched local-gradient stabilization."""
import argparse
from pathlib import Path
import shutil
from paper2.restart.privacy_simulation_runner import verify_protocol,PROTOCOL
from paper2.restart.validation_search import ROOT,read,sha,atomic_json
from paper2.restart.prepare_public_data import new_output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    old,config,plan,data=verify_protocol();output=new_output(args.output,data)
    failed=ROOT/'paper2/results/restart_privacy_simulation_v1/runs/outer_0_noisy_sigma4_seed123/failure.json'
    if not failed.exists():raise ValueError('Amendment requires preserved observed failure')
    config.update(version='public-transform-utility-simulation-v2',
        local_gradient_policy='Fixed L2 gradient clip 1 per local SGD step, same on all variants',
        prior_ledger_file='prior_ledger.json',
        noise_stream='stable-v2 run/round stream; distinct from v1 and local shuffling',
        amendment='Observed nonfinite local loss after noisy-v1 round47; no LR/budget/noise/metric changes')
    output.mkdir(parents=True);(output/'source_snapshot').mkdir()
    for name in old['source_sha256']:
        shutil.copyfile(PROTOCOL/'source_snapshot'/name,output/'source_snapshot'/name)
    source=dict(old['source_sha256'])
    for name in ('stable_regression_training.py','stable_simulation_training.py','stable_privacy_simulation_runner.py','amend_privacy_stability.py'):
        path=Path(__file__).with_name(name);source[name]=sha(path);shutil.copyfile(path,output/'source_snapshot'/name)
    atomic_json(output/'config.json',config);atomic_json(output/'plan.json',plan)
    shutil.copyfile(ROOT/'paper2/results/restart_privacy_simulation_v1/shared_ledger.json',output/'prior_ledger.json')
    shutil.copyfile(PROTOCOL/'hypothetical_noisy_transcript_budgets.json',output/'hypothetical_noisy_transcript_budgets.json')
    manifest=dict(old,status='FROZEN_V2_STABILITY_AMENDMENT_NO_TRAINING',source_sha256=source,
        parent_v1_protocol_sha256=sha(PROTOCOL/'manifest.json'),observed_failure_sha256=sha(failed),
        artifacts={n:sha(output/n) for n in ('config.json','plan.json','hypothetical_noisy_transcript_budgets.json','prior_ledger.json')})
    atomic_json(output/'manifest.json',manifest)
    print({'status':manifest['status'],'planned':len(plan),'legacy_noisy_events':read(output/'prior_ledger.json')['event_count']})
