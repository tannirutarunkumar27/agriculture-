"""
run_full_pipeline.py — Master Pipeline Runner
Executes all 14 steps of the KG-Hybrid research pipeline in sequence.
Tracks completion status in pipeline_status.json.

Usage:
    cd "D:\\Final Project\\implementation_for _crop recommendation\\KG-Hybrid"
    python run_full_pipeline.py
    python run_full_pipeline.py --from-step 9   # resume from step 9
    python run_full_pipeline.py --step 9        # run only step 9
"""
import sys, json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
STATUS_FILE  = PROJECT_ROOT / "pipeline_status.json"
PYTHON       = sys.executable

# ─── Pipeline step registry ───────────────────────────────────────────────────
STEPS = [
    {
        "id": 1,
        "name": "Multi-Source Agricultural Data Collection",
        "description": "Verify raw dataset exists (Crop_recommendation.csv)",
        "type": "check",
        "check_path": "data/raw/Crop_recommendation.csv",
    },
    {
        "id": 2,
        "name": "Data Preprocessing",
        "description": "Split, scale, and encode the dataset",
        "type": "check",
        "check_path": "data/processed/train.csv",
    },
    {
        "id": 3,
        "name": "Agricultural Knowledge Base Construction",
        "description": "Build structured knowledge base with NPK/temp/pH/rainfall/seasonal rules",
        "type": "script",
        "script": "src/knowledge/knowledge_feature_gate.py",
        "check_path": "data/processed/knowledge_base.json",
    },
    {
        "id": "4-5",
        "name": "Knowledge Representation + Knowledge-Guided Feature Gate",
        "description": "Compute feature importance from agronomic rules and save gate weights",
        "type": "script",
        "script": "src/knowledge/knowledge_feature_gate.py",
        "check_path": "outputs/tables/knowledge_feature_importance.csv",
    },
    {
        "id": "6A",
        "name": "XGBoost Branch",
        "description": "Verify XGBoost model trained",
        "type": "check",
        "check_path": "outputs/models/xgboost_model.pkl",
    },
    {
        "id": "6B",
        "name": "FT-Transformer Branch",
        "description": "Verify FT-Transformer model trained",
        "type": "check",
        "check_path": "outputs/models/best_model.pth",
    },
    {
        "id": 7,
        "name": "Knowledge-Guided Adaptive Fusion",
        "description": "Verify KG-Hybrid model trained",
        "type": "check",
        "check_path": "outputs/models/kg_hybrid_best.pth",
    },
    {
        "id": 8,
        "name": "Crop Recommendation Head",
        "description": "Verify prediction head output (part of KG-Hybrid model)",
        "type": "check",
        "check_path": "outputs/models/kg_hybrid_best.pth",
    },
    {
        "id": 9,
        "name": "Explainable AI Module (SHAP + Feature Importance)",
        "description": "Run SHAP analysis, feature importance, and decision interpretation",
        "type": "script",
        "script": "src/xai/shap_explainer.py",
        "check_path": "outputs/tables/shap_feature_importance.csv",
    },
    {
        "id": 10,
        "name": "Performance Evaluation",
        "description": "Verify evaluation metrics are saved",
        "type": "check",
        "check_path": "outputs/tables/model_comparison.csv",
    },
    {
        "id": 11,
        "name": "Comparative Analysis",
        "description": "Verify comparison table exists",
        "type": "check",
        "check_path": "outputs/tables/model_comparison.csv",
    },
    {
        "id": 12,
        "name": "Ablation Study",
        "description": "Run all ablation experiments (A1–A7)",
        "type": "check",
        "check_path": "outputs/ablation",
        "check_type": "dir",
    },
    {
        "id": 13,
        "name": "Robustness and Generalization Analysis",
        "description": "Run missing/noise/cross-dataset robustness tests",
        "type": "check",
        "check_path": "outputs/robustness",
        "check_type": "dir",
    },
    {
        "id": 14,
        "name": "Discussion and Conclusions",
        "description": "Auto-generate discussion document from all results",
        "type": "script",
        "script": "src/utils/generate_discussion.py",
        "check_path": "outputs/reports/discussion_and_conclusions.md",
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_status():
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"steps": {}, "last_run": None}


def save_status(status):
    status["last_run"] = datetime.now().isoformat()
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def check_step(step):
    """Return True if the step's output already exists."""
    cp = step.get("check_path")
    if not cp:
        return False
    p = PROJECT_ROOT / cp
    if step.get("check_type") == "dir":
        return p.exists() and any(p.glob("*.json"))
    return p.exists()


def run_script(script_rel, label):
    script_path = PROJECT_ROOT / script_rel
    if not script_path.exists():
        print(f"  [ERROR] Script not found: {script_path}")
        return False
    print(f"  Running: {script_rel}")
    result = subprocess.run(
        [PYTHON, "-X", "utf8", str(script_path)],
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] Script failed (exit code {result.returncode})")
        return False
    return True


def run_step(step, status):
    sid  = str(step["id"])
    name = step["name"]
    t0   = time.time()

    print(f"\n{'='*70}")
    print(f"  STEP {sid}: {name}")
    print(f"  {step['description']}")
    print(f"{'='*70}")

    # Already done?
    if status["steps"].get(sid, {}).get("status") == "done" and check_step(step):
        print(f"  [SKIP] Already completed.")
        return True

    ok = False
    if step["type"] == "check":
        ok = check_step(step)
        if ok:
            print(f"  [OK] Output verified: {step.get('check_path','')}")
        else:
            print(f"  [WARN] Output not found: {step.get('check_path','')}")
            print(f"         Run the corresponding training/preprocessing script first.")
    elif step["type"] == "script":
        ok = run_script(step["script"], name)
        if ok and check_step(step):
            print(f"  [OK] Output verified: {step.get('check_path','')}")

    elapsed = time.time() - t0
    status["steps"][sid] = {
        "status": "done" if ok else "failed",
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    save_status(status)

    if ok:
        print(f"  [OK] Completed in {elapsed:.1f}s")
    else:
        print(f"  [FAIL] Step {sid} incomplete.")
    return ok


def print_summary(status):
    print(f"\n{'='*70}")
    print("  PIPELINE STATUS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Step':<8} {'Status':<10} {'Time':>8}  Name")
    print(f"  {'-'*8} {'-'*10} {'-'*8}  {'-'*40}")
    for step in STEPS:
        sid = str(step["id"])
        info = status["steps"].get(sid, {})
        st = info.get("status", "pending")
        el = info.get("elapsed_s", "—")
        icon = "[OK]" if st == "done" else ("[FAIL]" if st == "failed" else "[..]")
        print(f"  {icon:<5} {sid:<7} {st:<10} {str(el):>7}s  {step['name']}")


    done   = sum(1 for s in STEPS if status["steps"].get(str(s["id"]),{}).get("status") == "done")
    failed = sum(1 for s in STEPS if status["steps"].get(str(s["id"]),{}).get("status") == "failed")
    total  = len(STEPS)
    print(f"\n  Completed: {done}/{total}  |  Failed: {failed}/{total}  |  Pending: {total-done-failed}/{total}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KG-Hybrid Full Pipeline Runner")
    parser.add_argument("--from-step", type=str, default=None,
                        help="Start from this step ID (e.g. 9 or 4-5)")
    parser.add_argument("--step",      type=str, default=None,
                        help="Run only this step ID")
    parser.add_argument("--force",     action="store_true",
                        help="Re-run even if step is marked done")
    args = parser.parse_args()

    status = load_status()

    print("\n" + "=" * 70)
    print("  KG-HYBRID CROP RECOMMENDATION — FULL PIPELINE RUNNER")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Determine which steps to run
    if args.step:
        steps_to_run = [s for s in STEPS if str(s["id"]) == args.step]
    elif args.from_step:
        ids = [str(s["id"]) for s in STEPS]
        start_idx = ids.index(args.from_step) if args.from_step in ids else 0
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = STEPS

    if args.force:
        for s in steps_to_run:
            status["steps"].pop(str(s["id"]), None)

    t_global = time.time()
    results = []
    for step in steps_to_run:
        ok = run_step(step, status)
        results.append(ok)

    total_time = time.time() - t_global
    print_summary(status)
    print(f"\n  Total runtime: {total_time:.1f}s")
    print(f"  Status file  : {STATUS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
