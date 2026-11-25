"""Freeze and verify a prospective development study without starting training.

Local SHA-256 checks detect inconsistent artifacts, not a cryptographic signature
or tamper-proof registry. All model rules remain the existing tested baselines.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil

from paper2.data import ROOT, SCHEMAS, file_hash, load_dataset, make_partition, object_hash, validate_partition
from paper2.train import checked_config

BASELINE_SOURCES = ["train.py", "data.py", "federated.py", "centralized.py", "metrics.py"]
POLICIES = {
    "selection": "minimum_global_validation_nll_first_round_wins_ties",
    "evaluation": "development_test_once_after_validation_selection_no_test_tuning",
    "preprocessing": "public_simulation_global_train_only_mean_std_and_categorical_vocab",
    "statistics": "descriptive_mean_sample_sd_and_paired_seed_differences_no_confirmatory_p_values",
}
ALGORITHMS = ["fedavg", "fedprox", "fednova"]
REFERENCES = ["centralized", "logistic_regression", "random_forest", "xgboost"]


def validate_protocol(protocol):
    expected = {"version", "purpose", "seeds", "tuning_seed", "clients", "min_client_rows", "datasets",
                "scenarios", "neural_algorithms", "reference_models", "metrics", "safety", "tuning_evidence",
                "tuning_verification"} | set(POLICIES)
    if set(protocol) != expected:
        raise ValueError("Protocol has missing or unknown fields")
    if not protocol["version"] or not protocol["purpose"]:
        raise ValueError("Protocol version/purpose required")
    seeds = protocol["seeds"]
    if not isinstance(seeds,list) or len(seeds)<3 or any(type(s) is not int or s<0 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("Need at least three distinct nonnegative integer study seeds")
    if type(protocol["tuning_seed"]) is not int or protocol["tuning_seed"]<0 or protocol["tuning_seed"] in seeds:
        raise ValueError("Study seeds must exclude the tuning seed")
    if type(protocol["clients"]) is not int or protocol["clients"]<2:
        raise ValueError("Need at least two clients")
    if type(protocol["min_client_rows"]) is not int or protocol["min_client_rows"]<1:
        raise ValueError("Invalid minimum client size")
    if protocol["neural_algorithms"] != ALGORITHMS or protocol["reference_models"] != REFERENCES:
        raise ValueError("This protocol supports exactly the tested baseline algorithms and references")
    if any(type(value) is not bool for value in protocol["safety"].values()) or protocol["safety"] != {"knowledge_enabled":False, "formal_privacy":False,
                              "automatic_hyperparameter_changes":False, "skip_failed_clients":False}:
        raise ValueError("Unsupported or misleading knowledge/privacy/failure policy")
    for key,value in POLICIES.items():
        if protocol[key] != value:
            raise ValueError(f"Unsupported {key} policy")
    if not protocol["datasets"] or set(protocol["datasets"])-{"canonical","maharashtra"}:
        raise ValueError("This approved baseline pilot covers canonical/Maharashtra only")
    for name,settings in protocol["datasets"].items():
        if set(settings)!={"neural","conventional"}:
            raise ValueError("Invalid dataset settings")
        if set(settings["neural"]) != {"rounds","local_epochs","batch_size","learning_rate","mu","hidden","participation","threads"}:
            raise ValueError("Neural settings must be complete and seed-independent")
        checked_config(settings["neural"])
        if settings["neural"]["participation"] != 1.0:
            raise ValueError("Base participation must be one; scenarios control partial participation")
        if set(settings["conventional"]) != set(REFERENCES[1:]):
            raise ValueError("Missing/unknown conventional reference")
    ids = []
    covered = set()
    for scenario in protocol["scenarios"]:
        if set(scenario)!={"id","strategy","alpha","participation","datasets"}:
            raise ValueError("Invalid scenario fields")
        identifier = scenario["id"]
        if not isinstance(identifier,str) or not identifier or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in identifier):
            raise ValueError("Unsafe scenario identifier")
        ids.append(identifier)
        if scenario["strategy"] not in {"iid","dirichlet","unequal","district"}:
            raise ValueError("Unsupported client partition strategy")
        for key in ["alpha","participation"]:
            value = scenario[key]
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f"Invalid scenario {key}")
        if scenario["participation"]>1:
            raise ValueError("Participation cannot exceed one")
        if not scenario["datasets"] or len(set(scenario["datasets"]))!=len(scenario["datasets"]):
            raise ValueError("Missing or duplicated scenario datasets")
        if set(scenario["datasets"])-set(protocol["datasets"]):
            raise ValueError("Scenario dataset not configured")
        if scenario["strategy"]=="district" and any("region" not in SCHEMAS[name] for name in scenario["datasets"]):
            raise ValueError("Cannot invent district clients for a dataset without regions")
        covered.update(scenario["datasets"])
    if len(ids)!=len(set(ids)) or covered!=set(protocol["datasets"]):
        raise ValueError("Duplicated scenario or uncovered dataset")
    for name in protocol["datasets"]:
        if not any(s["strategy"]=="iid" and s["participation"]==1 and name in s["datasets"] for s in protocol["scenarios"]):
            raise ValueError("Each dataset needs a full-participation IID reference")
    if protocol["metrics"] != {"primary":"macro_f1", "secondary":["accuracy","macro_precision","macro_recall","weighted_f1","nll"],
                                "resources":["local_sgd_steps","round_seconds","aggregation_seconds","upload_tensor_bytes","download_tensor_bytes"]}:
        raise ValueError("Unsupported metric policy")


def partition_key(dataset,seed,strategy,alpha):
    suffix = str(float(alpha)).replace(".","p")
    return f"{dataset}_{strategy}_{suffix}_seed{seed}.json"


def build_run_plan(protocol):
    validate_protocol(protocol)
    runs = []
    for dataset,settings in sorted(protocol["datasets"].items()):
        for seed in protocol["seeds"]:
            # A centralized reference is fit once per dataset/seed, not repeated
            # under every scenario when its source rows are identical.
            path = partition_key(dataset,seed,"iid",0.5)
            for model in REFERENCES:
                runs.append({"id":f"{dataset}_seed{seed}_reference_{model}", "dataset":dataset,
                             "seed":seed,"scenario":"centralized_reference","algorithm":model,
                             "partition":f"partitions/{path}",
                             "config":settings["neural"] | {"seed":seed} if model=="centralized" else
                                      {"seed":seed,"parameters":settings["conventional"][model]},
                             "evaluate_test":True, "test_scope":"development_holdout"})
            for scenario in protocol["scenarios"]:
                if dataset not in scenario["datasets"]:
                    continue
                path = partition_key(dataset,seed,scenario["strategy"],scenario["alpha"])
                for algorithm in ALGORITHMS:
                    runs.append({"id":f"{dataset}_seed{seed}_{scenario['id']}_{algorithm}","dataset":dataset,
                                 "seed":seed,"scenario":scenario["id"],"algorithm":algorithm,
                                 "partition":f"partitions/{path}",
                                 "config":settings["neural"] | {"seed":seed,"participation":scenario["participation"]},
                                 "evaluate_test":True,"test_scope":"development_holdout"})
    return runs


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_path(root,relative):
    root = Path(root).resolve()
    candidate = (root/relative).resolve()
    if candidate==root or root not in candidate.parents:
        raise ValueError("Artifact path escapes expected root")
    return candidate


def _write_new_json(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


def freeze_protocol(config_path,output):
    protocol = load_json(config_path)
    validate_protocol(protocol)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    evidence_path = _safe_path(ROOT,protocol["tuning_evidence"])
    verification_path = _safe_path(ROOT,protocol["tuning_verification"])
    evidence, verification = load_json(evidence_path),load_json(verification_path)
    if evidence["test_evaluated"] or verification["test_evaluated"] or verification["status"]!="PASS":
        raise ValueError("Tuning evidence must be independently verified and validation-only")
    for dataset,settings in protocol["datasets"].items():
        selected = evidence["selections"][dataset]
        if settings["neural"] | {"seed":protocol["tuning_seed"]} != selected["common_neural_config"]:
            raise ValueError("Frozen neural parameters differ from verified tuning selection")
        for model,parameters in settings["conventional"].items():
            if parameters != selected["conventional"][model]["parameters"]:
                raise ValueError("Frozen conventional parameters differ from tuning selection")
    runs = build_run_plan(protocol)
    prepared,source_hashes,split_hashes = {},{},{}
    for dataset in sorted(protocol["datasets"]):
        frame,schema,source = load_dataset(dataset)
        digest = file_hash(source)
        source_hashes[dataset] = {"relative_path":str(source.relative_to(ROOT)),"sha256":digest}
        for seed in protocol["seeds"]:
            specifications = {("iid",0.5)} | {(s["strategy"],float(s["alpha"])) for s in protocol["scenarios"] if dataset in s["datasets"]}
            for strategy,alpha in sorted(specifications):
                key = partition_key(dataset,seed,strategy,alpha)
                partition = make_partition(frame,schema,digest,dataset,seed,strategy,protocol["clients"],alpha,
                                           protocol["min_client_rows"])
                split_key = f"{dataset}:{seed}"
                split_digest = object_hash(partition["splits"])
                if split_key in split_hashes and split_hashes[split_key] != split_digest:
                    raise ValueError("Holdout rows differ across client scenarios")
                split_hashes[split_key] = split_digest
                prepared[key] = partition
    output.mkdir(parents=True,exist_ok=False)
    (output/"partitions").mkdir()
    (output/"source_snapshot").mkdir()
    _write_new_json(output/"protocol.json",protocol)
    _write_new_json(output/"run_plan.json",runs)
    _write_new_json(output/"tuning_evidence.json",evidence)
    _write_new_json(output/"tuning_verification.json",verification)
    for key,partition in prepared.items():
        _write_new_json(output/"partitions"/key,partition)
    sources = Path(__file__).parent
    code_hashes = {}
    for name in BASELINE_SOURCES:
        code_hashes[name] = file_hash(sources/name)
        shutil.copyfile(sources/name,output/"source_snapshot"/name)
    artifacts = {str(path.relative_to(output)):file_hash(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"version":protocol["version"],"protocol_hash":object_hash(protocol),"run_plan_hash":object_hash(runs),
                "planned_runs":len(runs),"partition_count":len(prepared),"source_data":source_hashes,
                "split_hashes":split_hashes,"code_hashes":code_hashes,"artifact_sha256":artifacts,
                "python":platform.python_version(),
                "dependencies":{name:importlib.metadata.version(name) for name in
                                ["torch","numpy","pandas","scikit-learn","joblib","xgboost"]},
                "training_started":False,"test_evaluated":False,
                "integrity_scope":"local consistency checks, not cryptographically signed or immutable"}
    _write_new_json(output/"manifest.json",manifest)
    verify_frozen_protocol(output)
    return manifest


def verify_frozen_protocol(folder,verify_environment=True):
    folder = Path(folder)
    manifest = load_json(folder/"manifest.json")
    for relative,digest in manifest["artifact_sha256"].items():
        if file_hash(_safe_path(folder,relative)) != digest:
            raise ValueError(f"Frozen artifact changed: {relative}")
    protocol,runs = load_json(folder/"protocol.json"),load_json(folder/"run_plan.json")
    validate_protocol(protocol)
    if object_hash(protocol)!=manifest["protocol_hash"] or object_hash(runs)!=manifest["run_plan_hash"]:
        raise ValueError("Frozen protocol/run-plan identity mismatch")
    if build_run_plan(protocol)!=runs or len(runs)!=manifest["planned_runs"]:
        raise ValueError("Run plan differs from protocol")
    if len({r["id"] for r in runs})!=len(runs):
        raise ValueError("Duplicate run identifiers")
    sources = Path(__file__).parent
    if set(manifest["code_hashes"])!=set(BASELINE_SOURCES):
        raise ValueError("Unknown baseline code schema")
    for name,digest in manifest["code_hashes"].items():
        if file_hash(sources/name)!=digest:
            raise ValueError(f"Baseline implementation changed: {name}; freeze an amended version")
    frames = {}
    split_hashes = {}
    paths = sorted(set(run["partition"] for run in runs))
    if len(paths)!=manifest["partition_count"]:
        raise ValueError("Partition count differs from planned references")
    for relative in paths:
        partition = load_json(_safe_path(folder,relative))
        dataset = partition["dataset"]
        if dataset not in frames:
            frame,schema,source = load_dataset(dataset)
            if file_hash(source)!=manifest["source_data"][dataset]["sha256"]:
                raise ValueError("Source dataset changed")
            frames[dataset] = (frame,schema)
        frame,schema = frames[dataset]
        if partition["source_sha256"]!=manifest["source_data"][dataset]["sha256"]:
            raise ValueError("Partition source hash differs")
        validate_partition(frame,schema,partition)
        key = f"{dataset}:{partition['seed']}"
        digest = object_hash(partition["splits"])
        if key in split_hashes and digest!=split_hashes[key]:
            raise ValueError("Holdout differs between scenarios")
        split_hashes[key] = digest
    if split_hashes!=manifest["split_hashes"]:
        raise ValueError("Split identities changed")
    if verify_environment:
        if platform.python_version()!=manifest["python"]:
            raise ValueError("Python changed; record an amended environment")
        for name,version in manifest["dependencies"].items():
            if importlib.metadata.version(name)!=version:
                raise ValueError(f"Dependency changed: {name}")
    return {"status":"PASS","planned_runs":len(runs),"partition_count":len(paths),
            "training_started":False,"test_evaluated":False,"protocol_hash":manifest["protocol_hash"]}
