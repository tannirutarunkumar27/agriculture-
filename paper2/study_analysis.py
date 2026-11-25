"""Descriptive matched-seed analysis of verified cached study results.

No model inference, no p-values, no test-based model/configuration selection.
Incomplete seed groups are never presented as complete comparisons.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from paper2.data import file_hash, object_hash
from paper2.study_protocol import load_json, verify_frozen_protocol
from paper2.study_runner import Execution, runner_identity, single_writer

PREDICTIVE = ["accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","nll"]
RESOURCE = ["total_local_sgd_steps","training_seconds","aggregation_seconds","upload_tensor_bytes","download_tensor_bytes"]


def mean_sd(values):
    if not values:
        raise ValueError("Empty statistical group")
    return float(statistics.mean(values)),float(statistics.stdev(values)) if len(values)>1 else None


def aggregate_complete_groups(results,protocol):
    """One independent RNG trial per seed, conditional on these reused datasets."""
    expected_seeds = set(protocol["seeds"])
    datasets = {dataset for scenario in protocol["scenarios"] for dataset in scenario["datasets"]}
    expected_groups = {(dataset,"centralized_reference",model) for dataset in datasets for model in protocol["reference_models"]}
    expected_groups |= {(dataset,scenario["id"],algorithm) for scenario in protocol["scenarios"]
                        for dataset in scenario["datasets"] for algorithm in protocol["neural_algorithms"]}
    by_group = defaultdict(dict)
    for result in results:
        key = (result["dataset"],result["scenario"],result["algorithm"])
        if key not in expected_groups:
            raise ValueError("Result group is not in the frozen protocol")
        seed = result["seed"]
        if seed not in expected_seeds or seed in by_group[key]:
            raise ValueError("Unexpected or duplicate result seed")
        by_group[key][seed] = result
    complete,excluded = [],[]
    for key in sorted(expected_groups):
        values = by_group.get(key,{})
        if set(values)!=expected_seeds:
            excluded.append({"dataset":key[0],"scenario":key[1],"algorithm":key[2],
                             "present_seeds":sorted(values),"missing_seeds":sorted(expected_seeds-set(values))})
            continue
        row = {"dataset":key[0],"scenario":key[1],"algorithm":key[2],"seeds":len(values)}
        ordered = [values[seed] for seed in protocol["seeds"]]
        for metric in PREDICTIVE:
            mean,sd = mean_sd([r["test"][metric] for r in ordered])
            row[metric+"_mean"] = mean
            row[metric+"_sample_sd"] = sd
        for metric in RESOURCE:
            measurements = [r["resources"][metric] for r in ordered]
            if all(value is None for value in measurements):
                row[metric+"_mean"] = row[metric+"_sample_sd"] = None
            elif any(value is None for value in measurements):
                raise ValueError("Resource field inconsistently missing across seeds")
            else:
                row[metric+"_mean"],row[metric+"_sample_sd"] = mean_sd(measurements)
        complete.append(row)
    paired_rows,paired_summary = [],[]
    for scenario in protocol["scenarios"]:
        for dataset in scenario["datasets"]:
            comparisons = [(name,"fedavg",scenario["id"]) for name in ["fedprox","fednova"]]
            comparisons += [(algorithm,reference,"centralized_reference")
                            for algorithm in protocol["neural_algorithms"] for reference in protocol["reference_models"]]
            for target,reference,reference_scenario in comparisons:
                a = by_group.get((dataset,scenario["id"],target),{})
                b = by_group.get((dataset,reference_scenario,reference),{})
                if set(a)!=expected_seeds or set(b)!=expected_seeds:
                    continue
                differences = []
                for seed in protocol["seeds"]:
                    difference = 100*(a[seed]["test"]["macro_f1"]-b[seed]["test"]["macro_f1"])
                    differences.append(difference)
                    paired_rows.append({"dataset":dataset,"scenario":scenario["id"],"target":target,
                                        "reference":reference,"seed":seed,"macro_f1_difference_pp":difference})
                mean,sd = mean_sd(differences)
                paired_summary.append({"dataset":dataset,"scenario":scenario["id"],"target":target,"reference":reference,
                                       "seeds":len(differences),"macro_f1_difference_mean_pp":mean,
                                       "macro_f1_difference_sample_sd_pp":sd})
    return complete,excluded,paired_rows,paired_summary


def write_csv(path,rows,empty_fields):
    with Path(path).open("x",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_study(frozen,study,output,allow_partial=False):
    frozen,study,output = Path(frozen).resolve(),Path(study).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output==frozen or frozen in output.parents or output in frozen.parents or output==study or study in output.parents or output in study.parents:
        raise ValueError("Analysis must be separate from frozen package and execution output")
    verified = verify_frozen_protocol(frozen)
    plan,protocol = load_json(frozen/"run_plan.json"),load_json(frozen/"protocol.json")
    identity = runner_identity(frozen)
    if load_json(study/"study_binding.json")["identity"]!=identity:
        raise ValueError("Study binding/implementation differs")
    results,pending,hashes = [],[],{}
    with single_writer(study):
        execution = Execution(frozen,study,identity)
        for run in plan:
            folder = execution.folder(run)
            if (folder/"completion.json").exists():
                result = execution.completed(run,folder)  # cached metrics only, never model inference
                result["partition_sha256"] = file_hash(frozen/run["partition"])
                result["split_hash"] = object_hash(load_json(frozen/run["partition"])["splits"])
                results.append(result)
                hashes[run["id"]] = {"completion":file_hash(folder/"completion.json"),"summary":file_hash(folder/"summary.json")}
            else:
                status = load_json(folder/"run_state.json")["status"] if (folder/"run_state.json").exists() else "pending"
                pending.append({"run_id":run["id"],"status":status})
        if pending and not allow_partial:
            raise ValueError(f"Study incomplete: {len(results)}/{len(plan)} verified. Partial analysis requires explicit opt-in.")
        # Pairing uses identical held-out rows, not just a matching seed label.
        split_identity = {}
        for result in results:
            key = (result["dataset"],result["seed"])
            if key in split_identity and split_identity[key]!=result["split_hash"]:
                raise ValueError("Results have different test/validation rows despite same dataset/seed")
            split_identity[key] = result["split_hash"]
    aggregates,excluded,pairs,paired_summary = aggregate_complete_groups(results,protocol)
    rows = [{"run_id":r["run_id"],"dataset":r["dataset"],"scenario":r["scenario"],"algorithm":r["algorithm"],
             "seed":r["seed"],"best_round":r["best_round"],"test_samples":r["test"]["samples"],
             **{k:r["test"][k] for k in PREDICTIVE},**r["resources"]} for r in results]
    regional = []
    per_class = []
    for result in results:
        for region,value in result["regional_test"].items():
            regional.append({"run_id":result["run_id"],"dataset":result["dataset"],"scenario":result["scenario"],
                             "algorithm":result["algorithm"],"seed":result["seed"],"region":region,
                             "samples":value["metrics"]["samples"],"present_true_crop_classes":value["present_true_crop_classes"],
                             "accuracy":value["metrics"]["accuracy"],"macro_f1_global_schema":value["metrics"]["macro_f1"],
                             "nll":value["metrics"]["nll"]})
        for name,value in result["test"]["per_class"].items():
            if isinstance(value,dict) and name not in {"macro avg","weighted avg","micro avg"}:
                per_class.append({"run_id":result["run_id"],"dataset":result["dataset"],"seed":result["seed"],
                                  "scenario":result["scenario"],"algorithm":result["algorithm"],"crop":name,**value})
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/"per_run_metrics.csv",rows,["run_id"])
    write_csv(output/"aggregate_metrics.csv",aggregates,["dataset","scenario","algorithm","seeds"])
    write_csv(output/"paired_seed_differences.csv",pairs,["dataset","scenario","target","reference","seed","macro_f1_difference_pp"])
    write_csv(output/"paired_difference_summary.csv",paired_summary,["dataset","scenario","target","reference","seeds"])
    write_csv(output/"regional_metrics.csv",regional,["run_id","region"])
    write_csv(output/"per_class_metrics.csv",per_class,["run_id","crop"])
    manifest = {"status":"complete" if not pending else "partial","verified_runs":len(results),"planned_runs":len(plan),
                "complete_groups":len(aggregates),"excluded_incomplete_groups":excluded,"pending_or_failed_runs":pending,
                "frozen_verification":verified,"source_receipt_hashes":hashes,"analysis_code_sha256":file_hash(Path(__file__)),
                "test_inference_performed":False,"statistics":protocol["statistics"],
                "units":"predictive metrics in [0,1], NLL dimensionless; paired F1 differences in percentage points",
                "limitations":"development holdouts, conditional seed variability, reused overlapping splits; no confirmatory p-values, privacy or field validation"}
    with (output/"verification.json").open("x",encoding="utf-8") as stream:
        json.dump(manifest,stream,indent=2,allow_nan=False)
    return {"status":manifest["status"],"verified_runs":len(results),"complete_groups":len(aggregates),
            "paired_comparisons":len(paired_summary),"output":str(output)}
