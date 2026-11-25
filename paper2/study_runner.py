"""Bounded frozen-plan execution, journaled evaluation and fail-closed recovery.

No changes to the five frozen baseline source files. Single local writer only.
Test evaluation is an at-most-one automatic attempt: an uncommitted ambiguous
evaluation requires human intervention, not an automatic second evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import joblib
import numpy as np
import torch

from paper2.centralized import make_estimator, run_reference, serializable_parameters
from paper2.data import file_hash, load_dataset, object_hash, prepare_arrays
from paper2.federated import make_model
from paper2.metrics import probability_metrics
from paper2.study_protocol import _safe_path, load_json, verify_frozen_protocol
from paper2.train import checked_config, evaluate, run as train_neural

NEURAL = {"centralized", "fedavg", "fedprox", "fednova"}


class AmbiguousEvaluationError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _replace_portably(temporary, path):
    """Prefer atomic replacement, with a bounded Windows-safe fallback."""
    temporary, path = Path(temporary), Path(path)
    for delay in (0.0, 0.02, 0.05, 0.1, 0.2):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            pass
    shutil.copyfile(temporary, path)
    temporary.unlink(missing_ok=True)


def atomic_json(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace_portably(temporary,path)


@contextmanager
def single_writer(output):
    """OS-held lock releases on process exit; no unsafe stale-PID deletion."""
    handle = (Path(output)/"runner.lock").open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0,2)
            if handle.tell()==0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as error:
                raise RuntimeError("Another runner holds this study output lock") from error
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another runner holds this study output lock") from error
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()


def runner_identity(frozen):
    module = Path(__file__)
    policy = module.with_name("study_protocol.py")
    return {"frozen_manifest_sha256":file_hash(Path(frozen)/"manifest.json"),
            "runner_sha256":file_hash(module),"protocol_verifier_sha256":file_hash(policy)}


def selected_runs(plan,run_ids=None):
    if run_ids is None:
        return plan
    if not run_ids or len(set(run_ids))!=len(run_ids):
        raise ValueError("Run selection must be nonempty with no duplicate identifiers")
    allowed = {run["id"] for run in plan}
    if set(run_ids)-allowed:
        raise ValueError(f"Unknown frozen run identifiers: {sorted(set(run_ids)-allowed)}")
    return [run for run in plan if run["id"] in set(run_ids)]


def run_identity(study_identity,run):
    return object_hash({"study":study_identity,"run":run})


def _require(condition,message):
    if not condition:
        raise ValueError(message)


class Execution:
    def __init__(self,frozen,output,identity):
        self.frozen,self.output = Path(frozen),Path(output)
        self.identity = identity
        self.frozen_manifest = load_json(self.frozen/"manifest.json")
        self.contexts = {}

    def context(self,run):
        key = run["partition"]
        if key not in self.contexts:
            path = _safe_path(self.frozen,key)
            partition = load_json(path)
            frame,schema,source = load_dataset(run["dataset"])
            _require(partition["dataset"]==run["dataset"] and partition["seed"]==run["seed"],"Run/partition mismatch")
            _require(file_hash(source)==partition["source_sha256"],"Source changed")
            x,y,preprocessing = prepare_arrays(frame,schema,partition)
            self.contexts[key] = (path,partition,x,y,preprocessing,frame,schema)
        return self.contexts[key]

    def folder(self,run):
        return _safe_path(self.output,"runs/"+run["id"])

    def state(self,folder,status,**extra):
        atomic_json(folder/"run_state.json",{"status":status,"updated_utc":utc_now(),**extra})

    def failure(self,folder,error):
        errors = folder/"errors"
        errors.mkdir(exist_ok=True)
        index = len(list(errors.glob("error_*.json")))+1
        payload = {"utc":utc_now(),"type":type(error).__name__,"message":str(error),
                   "traceback":traceback.format_exc()}
        atomic_json(errors/f"error_{index:04d}.json",payload)
        self.state(folder,"failed",error_type=type(error).__name__,message=str(error))

    def training_model(self,run,baseline):
        """Verify and reconstruct validation-selected model, never evaluate test."""
        path,partition,x,y,preprocessing,_,_ = self.context(run)
        manifest = load_json(baseline/"manifest.json")
        _require(manifest["partition_sha256"]==file_hash(path),"Training partition changed")
        _require(manifest["source_sha256"]==partition["source_sha256"],"Training source identity mismatch")
        _require(manifest["preprocessing"]==preprocessing,"Preprocessing differs from frozen training-only policy")
        _require(all(self.frozen_manifest["code_hashes"].get(name)==digest for name,digest in manifest["code_hashes"].items()),
                 "Training code differs from frozen implementation")
        hashes = {"manifest.json":file_hash(baseline/"manifest.json")}
        if run["algorithm"] in NEURAL:
            config = checked_config(run["config"])
            _require(manifest["config"]==config and manifest["algorithm"]==run["algorithm"],"Training config/algorithm mismatch")
            _require(manifest["test_evaluation_enabled"] is False,"Training stage must be validation-only")
            expected_identity = object_hash({"config":{k:v for k,v in config.items() if k!="rounds"},
                                            "algorithm":run["algorithm"],"partition":partition,
                                            "code_hashes":manifest["code_hashes"],"evaluate_test":False})
            _require(manifest["identity"]==expected_identity,"Training identity mismatch")
            checkpoint = torch.load(baseline/"checkpoint.pt",map_location="cpu",weights_only=True)
            _require(checkpoint["identity"]==expected_identity,"Checkpoint identity mismatch")
            _require(checkpoint["completed_round"]==config["rounds"],"Training is incomplete")
            records = checkpoint["records"]
            _require([r["round"] for r in records]==list(range(1,config["rounds"]+1)),"Round record coverage mismatch")
            best = min(records,key=lambda record:record["validation"]["nll"])
            _require(best["round"]==checkpoint["best_round"],"Validation checkpoint selection mismatch")
            model = make_model(x.shape[1],len(partition["classes"]),config["hidden"])
            model.load_state_dict(checkpoint["best_state"])
            model.eval()
            _require(all(torch.isfinite(value).all() for value in model.state_dict().values()),"Non-finite saved model")
            validation = partition["splits"]["validation"]
            metrics,_ = evaluate(model,torch.from_numpy(x[validation]),torch.from_numpy(y[validation]),partition["classes"],True)
            _require(abs(metrics["nll"]-checkpoint["best_nll"])<1e-6,"Recomputed checkpoint NLL differs")
            resources = {"total_local_sgd_steps":sum(c["actual_sgd_steps"] for r in records for c in r["client_metrics"]),
                         "training_seconds":sum(r["round_seconds"] for r in records),
                         "aggregation_seconds":sum(r["aggregation_seconds"] for r in records),
                         "upload_tensor_bytes":sum(r["upload_payload_bytes"] for r in records),
                         "download_tensor_bytes":sum(r["download_payload_bytes"] for r in records)}
            details = {"validation":metrics,"best_round":checkpoint["best_round"],"completed_rounds":config["rounds"],
                       "resources":resources,"round_records":records}
            hashes["checkpoint.pt"] = file_hash(baseline/"checkpoint.pt")
        else:
            summary = load_json(baseline/"summary.json")
            _require(summary["test_evaluated"] is False and summary["test"] is None,"Reference trained with test evaluation")
            _require(summary["model"]==run["algorithm"] and summary["parameters"]==run["config"]["parameters"] and
                     summary["seed"]==run["seed"],"Reference settings mismatch")
            prototype = make_estimator(run["algorithm"],run["config"]["parameters"],run["seed"])
            _require(manifest["parameters"]==serializable_parameters(prototype.get_params()),"Estimator defaults/config mismatch")
            _require(file_hash(baseline/"model.joblib")==manifest["model_sha256"],"Saved reference model changed")
            model = joblib.load(baseline/"model.joblib")  # only locally generated, hash-checked artifacts
            _require(np.array_equal(model.classes_,np.arange(len(partition["classes"]))),"Class ordering differs")
            validation = partition["splits"]["validation"]
            metrics = probability_metrics(y[validation],model.predict_proba(x[validation]),partition["classes"])
            _require(abs(metrics["nll"]-summary["validation"]["nll"])<1e-6,"Reference validation NLL differs")
            details = {"validation":metrics,"best_round":None,"completed_rounds":None,
                       "resources":{"total_local_sgd_steps":None,"training_seconds":summary["fit_seconds"],
                                    "aggregation_seconds":0.0,"upload_tensor_bytes":0,"download_tensor_bytes":0},"round_records":[]}
            hashes |= {name:file_hash(baseline/name) for name in ["model.joblib","summary.json"]}
        _require(not (baseline/"test_predictions.npz").exists(),"Unexpected training-stage test artifacts")
        return model,details,hashes

    def ensure_training(self,run,folder,retry_failed):
        receipt_path = folder/"training_receipt.json"
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            self.check_training_receipt(run,folder,receipt)
            return receipt
        attempts = folder/"attempts"
        attempts.mkdir(exist_ok=True)
        previous = sorted(attempts.glob("attempt_*"))
        baseline = None
        if previous:
            latest = previous[-1]
            if run["algorithm"] in NEURAL and (latest/"checkpoint.pt").exists():
                checkpoint = torch.load(latest/"checkpoint.pt",map_location="cpu",weights_only=True)
                if checkpoint["completed_round"]<=run["config"]["rounds"]:
                    baseline = latest
            elif (latest/"manifest.json").exists() and (latest/"summary.json").exists():
                baseline = latest
            if baseline is None and not retry_failed:
                raise RuntimeError("Incomplete training attempt has no usable checkpoint; pass --retry-failed to preserve it and start a new attempt")
        if baseline is None:
            baseline = attempts/f"attempt_{len(previous)+1:04d}"
        path = self.context(run)[0]
        log = folder/"training.log"
        self.state(folder,"training",attempt=baseline.name)
        with log.open("a",encoding="utf-8") as stream,redirect_stdout(stream),redirect_stderr(stream):
            print(f"[{utc_now()}] {run['id']}: {baseline.name}",flush=True)
            if run["algorithm"] in NEURAL:
                completed = 0
                if (baseline/"checkpoint.pt").exists():
                    completed = torch.load(baseline/"checkpoint.pt",map_location="cpu",weights_only=True)["completed_round"]
                if completed<run["config"]["rounds"]:
                    train_neural(path,baseline,run["algorithm"],run["config"],resume=completed>0,evaluate_test=False)
            elif not (baseline/"manifest.json").exists():
                run_reference(path,baseline,run["algorithm"],run["config"]["parameters"],run["seed"],evaluate_test=False)
            _,details,hashes = self.training_model(run,baseline)
        records = details.pop("round_records")
        atomic_json(folder/"training_records.json",records)
        receipt = {"identity":run_identity(self.identity,run),"attempt":str(baseline.relative_to(folder)),
                   "artifact_hashes":hashes,"details":details,"records_sha256":file_hash(folder/"training_records.json"),
                   "test_evaluated":False}
        atomic_json(receipt_path,receipt)
        self.state(folder,"ready_for_evaluation")
        return receipt

    def check_training_receipt(self,run,folder,receipt):
        _require(receipt["identity"]==run_identity(self.identity,run) and receipt["test_evaluated"] is False,
                 "Training receipt identity/policy mismatch")
        baseline = _safe_path(folder,receipt["attempt"])
        for name,digest in receipt["artifact_hashes"].items():
            _require(file_hash(_safe_path(baseline,name))==digest,"Training artifact changed")
        _require(file_hash(folder/"training_records.json")==receipt["records_sha256"],"Training records changed")

    def predict_test(self,run,model):
        _,partition,x,_,_,_,_ = self.context(run)
        ids = partition["splits"]["test"]
        if run["algorithm"] in NEURAL:
            torch.set_num_threads(run["config"]["threads"])
            model.eval()
            with torch.no_grad():
                logits = torch.cat([model(batch) for batch in torch.from_numpy(x[ids]).split(1024)])
                _require(torch.isfinite(logits).all(),"Non-finite test logits")
                return logits.softmax(dim=1).numpy()
        return model.predict_proba(x[ids])

    def save_bundle(self,folder,probability,labels,ids,metadata):
        path = folder/"test_bundle.npz"
        temporary = folder/"test_bundle.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,probabilities=probability,labels=labels,row_ids=np.array(ids),
                                metadata=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False)))
            stream.flush()
            os.fsync(stream.fileno())
        _replace_portably(temporary,path)

    def cached_evaluation(self,run,folder,receipt):
        marker = load_json(folder/"evaluation_journal.json")
        digest = file_hash(folder/"training_receipt.json")
        _require(marker["identity"]==run_identity(self.identity,run) and marker["training_receipt_sha256"]==digest,
                 "Evaluation journal differs from training receipt")
        with np.load(folder/"test_bundle.npz",allow_pickle=False) as bundle:
            probability,labels,ids = bundle["probabilities"],bundle["labels"],bundle["row_ids"]
            metadata = json.loads(str(bundle["metadata"].item()))
        _require(metadata["identity"]==marker["identity"] and metadata["training_receipt_sha256"]==digest,
                 "Prediction bundle identity mismatch")
        _,partition,_,y,_,frame,schema = self.context(run)
        _require(np.array_equal(ids,np.array(partition["splits"]["test"])) and np.array_equal(labels,y[ids]),
                 "Prediction row IDs/labels differ from frozen holdout")
        metrics = probability_metrics(labels,probability,partition["classes"])
        _require(object_hash(metrics)==metadata["metrics_hash"],"Cached prediction metrics differ")
        regions = {}
        if "region" in schema:
            names = frame.iloc[ids][schema["region"]].astype(str).to_numpy()
            for name in sorted(set(names)):
                selected = np.flatnonzero(names==name)
                regions[name] = {"present_true_crop_classes":len(np.unique(labels[selected])),
                                 "metrics":probability_metrics(labels[selected],probability[selected],partition["classes"])}
        return {"run_id":run["id"],"dataset":run["dataset"],"seed":run["seed"],"scenario":run["scenario"],
                "algorithm":run["algorithm"],"config":run["config"],"scope":"development_holdout",
                "validation":receipt["details"]["validation"],"test":metrics,"regional_test":regions,
                "best_round":receipt["details"]["best_round"],"resources":receipt["details"]["resources"],
                "evaluation_seconds":metadata["evaluation_seconds"],"test_evaluation_attempts":1,
                "knowledge_enabled":False,"formal_privacy":False}

    def finalize(self,run,folder,receipt):
        journal,bundle = folder/"evaluation_journal.json",folder/"test_bundle.npz"
        if journal.exists() and not bundle.exists():
            raise AmbiguousEvaluationError("Evaluation started but no committed prediction bundle exists; refusing automatic re-evaluation")
        if bundle.exists() and not journal.exists():
            raise AmbiguousEvaluationError("Prediction bundle has no evaluation journal; manual recovery required")
        if not journal.exists():
            model,_,_ = self.training_model(run,_safe_path(folder,receipt["attempt"]))
            marker = {"status":"started","identity":run_identity(self.identity,run),"started_utc":utc_now(),
                      "training_receipt_sha256":file_hash(folder/"training_receipt.json")}
            atomic_json(journal,marker)  # before any model test inference
            self.state(folder,"evaluating")
            started = time.perf_counter()
            probability = self.predict_test(run,model)
            _,partition,_,y,_,_,_ = self.context(run)
            ids = partition["splits"]["test"]
            metrics = probability_metrics(y[ids],probability,partition["classes"])
            metadata = {"identity":marker["identity"],"training_receipt_sha256":marker["training_receipt_sha256"],
                        "metrics_hash":object_hash(metrics),"evaluation_seconds":time.perf_counter()-started}
            self.save_bundle(folder,probability,y[ids],ids,metadata)
        result = self.cached_evaluation(run,folder,receipt)
        atomic_json(folder/"summary.json",result)
        hashes = {name:file_hash(folder/name) for name in ["training_receipt.json","test_bundle.npz","summary.json"]}
        atomic_json(folder/"completion.json",{"identity":run_identity(self.identity,run),"artifact_hashes":hashes,
                                            "test_evaluation_attempts":1})
        marker = load_json(journal)
        marker["status"] = "committed"
        atomic_json(journal,marker)
        self.state(folder,"complete")
        return result

    def completed(self,run,folder):
        completion = load_json(folder/"completion.json")
        _require(completion["identity"]==run_identity(self.identity,run) and completion["test_evaluation_attempts"]==1,
                 "Completion identity mismatch")
        for name,digest in completion["artifact_hashes"].items():
            _require(file_hash(_safe_path(folder,name))==digest,"Completed run artifact changed; refusing re-execution")
        receipt = load_json(folder/"training_receipt.json")
        self.check_training_receipt(run,folder,receipt)
        result = self.cached_evaluation(run,folder,receipt)
        _require(object_hash(result)==object_hash(load_json(folder/"summary.json")),"Completed summary differs from cached evidence")
        return result

    def execute_one(self,run,retry_failed=False):
        folder = self.folder(run)
        folder.mkdir(parents=True,exist_ok=True)
        binding = folder/"run_binding.json"
        expected = {"identity":run_identity(self.identity,run),"run":run}
        if binding.exists():
            _require(load_json(binding)==expected,"Run output belongs to another plan")
        else:
            _require(not list(folder.iterdir()),"Unbound existing run directory must not be overwritten")
            atomic_json(binding,expected)
        try:
            if (folder/"completion.json").exists():
                return self.completed(run,folder)
            if (folder/"run_state.json").exists() and load_json(folder/"run_state.json")["status"]=="failed" and not retry_failed:
                raise RuntimeError("Run previously failed; inspect its error record and explicitly pass --retry-failed")
            # Ambiguous evaluation is checked before any potential re-training.
            if (folder/"evaluation_journal.json").exists() and not (folder/"test_bundle.npz").exists():
                raise AmbiguousEvaluationError("Uncommitted evaluation: automatic retry is prohibited")
            receipt = self.ensure_training(run,folder,retry_failed)
            return self.finalize(run,folder,receipt)
        except BaseException as error:
            self.failure(folder,error)
            raise

    def export(self,plan):
        rows,failed = [],[]
        for run in plan:
            folder = self.folder(run)
            if (folder/"completion.json").exists():
                result = self.completed(run,folder)
                rows.append({"run_id":run["id"],"dataset":run["dataset"],"seed":run["seed"],
                             "scenario":run["scenario"],"algorithm":run["algorithm"],
                             "test_accuracy":result["test"]["accuracy"],"test_macro_f1":result["test"]["macro_f1"],
                             "test_nll":result["test"]["nll"],"best_round":result["best_round"],**result["resources"]})
            elif (folder/"run_state.json").exists() and load_json(folder/"run_state.json")["status"]=="failed":
                failed.append(run["id"])
        fields = ["run_id","dataset","seed","scenario","algorithm","test_accuracy","test_macro_f1","test_nll",
                  "best_round","total_local_sgd_steps","training_seconds","aggregation_seconds","upload_tensor_bytes","download_tensor_bytes"]
        temporary = self.output/"completed_runs.csv.tmp"
        with temporary.open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        _replace_portably(temporary,self.output/"completed_runs.csv")
        status = "complete" if len(rows)==len(plan) else "partial"
        report = {"status":"failed" if failed else status,"planned_runs":len(plan),"completed_runs":len(rows),
                  "failed_runs":failed,"pending_runs":len(plan)-len(rows)-len(failed),
                  "test_scope":"development_holdout","knowledge_enabled":False,"formal_privacy":False}
        atomic_json(self.output/"study_progress.json",report)
        return report


def execute_study(frozen,output,run_ids=None,max_runs=1,resume=False,retry_failed=False,dry_run=False):
    if type(max_runs) is not int or max_runs<1:
        raise ValueError("Invocation run limit must be a positive integer")
    frozen,output = Path(frozen).resolve(),Path(output).resolve()
    if output==frozen or frozen in output.parents or output in frozen.parents:
        raise ValueError("Study output must not contain or overwrite the frozen protocol package")
    verified = verify_frozen_protocol(frozen)
    plan = load_json(frozen/"run_plan.json")
    chosen = selected_runs(plan,run_ids)
    if dry_run:
        return {"status":"dry_run","planned_runs":len(plan),"selected_runs":len(chosen),"max_new_runs":max_runs,
                "run_ids":[r["id"] for r in chosen],"training_started":False,"frozen_verification":verified}
    identity = runner_identity(frozen)
    if output.exists() and not resume:
        raise FileExistsError("Existing study output requires --resume; no overwrite")
    if not output.exists() and resume:
        raise FileNotFoundError("Cannot resume a study output that does not exist")
    output.mkdir(parents=True,exist_ok=True)
    with single_writer(output):
        ownership = output/"study_binding.json"
        if ownership.exists():
            _require(load_json(ownership)["identity"]==identity,"Study runner/protocol identity changed")
        else:
            _require(set(p.name for p in output.iterdir())=={"runner.lock"},"Existing output has no valid study binding")
            atomic_json(ownership,{"identity":identity,"created_utc":utc_now(),"frozen":str(frozen)})
            snapshot = output/"runner_snapshot"
            snapshot.mkdir()
            for path in [Path(__file__),Path(__file__).with_name("study_protocol.py")]:
                shutil.copyfile(path,snapshot/path.name)
        execution = Execution(frozen,output,identity)
        executed = 0
        try:
            execution.export(plan)  # verifies existing completed artifacts before proceeding
            for run in chosen:
                folder = execution.folder(run)
                if (folder/"completion.json").exists():
                    execution.completed(run,folder)
                    print(f"Verified completed run, not repeated: {run['id']}",flush=True)
                    continue
                if executed>=max_runs:
                    break
                print(f"Executing frozen run: {run['id']}",flush=True)
                execution.execute_one(run,retry_failed)
                executed += 1
                execution.export(plan)
        except BaseException as error:
            errors = output/"study_errors"
            errors.mkdir(exist_ok=True)
            index = len(list(errors.glob("error_*.json")))+1
            payload = {"utc":utc_now(),"type":type(error).__name__,"message":str(error),
                       "traceback":traceback.format_exc()}
            atomic_json(errors/f"error_{index:04d}.json",payload)
            try:
                execution.export(plan)
            except BaseException as export_error:
                payload["progress_export_error"] = {"type":type(export_error).__name__,"message":str(export_error)}
                atomic_json(errors/f"error_{index:04d}.json",payload)
                atomic_json(output/"study_progress.json",{"status":"blocked_integrity","planned_runs":len(plan),
                                                         "completed_runs":None,"failure_record":f"study_errors/error_{index:04d}.json"})
                raise error from export_error
            raise
        result = execution.export(plan)
        result["executed_this_invocation"] = executed
        return result
