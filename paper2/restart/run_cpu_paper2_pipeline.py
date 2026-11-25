"""Continue the authorized CPU-only RR06 -> RR07 study pipeline without overwriting evidence."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from paper2.restart import (finalize_matched_neural_selection as finalizer,
                            rr07_development_protocol as rr07_plan,
                            rr07_execution_gate as rr07_gate,
                            rr07_runner,
                            verify_matched_neural)
from paper2.restart.validation_search import ROOT, atomic_json, read


RESULTS = ROOT / "paper2/results"
RR06_PACKAGE = RESULTS / "restart_matched_neural_execution_v10"
RR06_STUDY = RESULTS / "restart_matched_neural_study_cpu_local_v1"
RR06_AUDIT = RESULTS / "restart_matched_neural_study_cpu_local_audit_v1"
FINALIZATION = RESULTS / "restart_matched_neural_cpu_local_finalization_v1"
RR07_PLAN = RESULTS / "restart_rr07_cpu_local_plan_v1"
RR07_EXECUTION = RESULTS / "restart_rr07_cpu_local_execution_v1"
RR07_STUDY = RESULTS / "restart_rr07_outer_cpu_local_v1"


def wait_for_rr06(poll_seconds=30):
    while True:
        progress_path = RR06_STUDY / "progress.json"
        if progress_path.exists():
            progress = read(progress_path)
            if progress.get("status") == "COMPLETE" and progress.get("completed") == 504 and progress.get("selection_available") is True:
                return progress
            if progress.get("status") == "FAILED":
                raise RuntimeError("RR06 stopped with a failure; inspect immutable attempt journals")
        time.sleep(poll_seconds)


def run(poll_seconds=30):
    if type(poll_seconds) is not int or poll_seconds < 1:
        raise ValueError("poll_seconds must be a positive integer")
    for path in (RR06_AUDIT, FINALIZATION, RR07_PLAN, RR07_EXECUTION, RR07_STUDY):
        if path.exists():
            raise FileExistsError(f"Pipeline output already exists: {path}")
    wait_for_rr06(poll_seconds)
    verify_matched_neural.audit(RR06_PACKAGE, RR06_STUDY, RR06_AUDIT)
    finalizer.finalize(RR06_PACKAGE, RR06_STUDY, RR06_AUDIT, FINALIZATION)
    rr07_plan.freeze(RR07_PLAN, FINALIZATION)
    rr07_gate.freeze(RR07_EXECUTION, RR07_PLAN)
    os.environ["PAPER2_DEVICE"] = "cpu"
    result = rr07_runner.execute(RR07_EXECUTION, RR07_STUDY, cap=160, allow_outer_inference=True)
    atomic_json(RR07_STUDY / "pipeline_receipt.json", {"status": "CPU_PIPELINE_COMPLETE", "rr06_study": str(RR06_STUDY),
                "rr06_audit": str(RR06_AUDIT), "finalization": str(FINALIZATION), "rr07_plan": str(RR07_PLAN),
                "rr07_execution": str(RR07_EXECUTION), "rr07_result": result, "cpu_only": True})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(); print(run(args.poll_seconds))
