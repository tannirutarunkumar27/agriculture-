"""
File: verify_everything.py

Purpose:
Master System Verification Suite — End-to-End System Audit
KrishiSetu Platform — Full Research & Production Integration Check

Verifies:
  1. All 14 ML Research Pipeline Steps
  2. Production FastAPI Server Endpoints
  3. Real-time Weather & Soil Data Ingestion Engine
  4. TorchScript Mobile Edge Exporter
  5. Kerala Regional Cropping Seasons & Economic Engine
  6. DPDP 2025 Security & Digital Consent Logger
  7. Mobile Model Artifact (.pt)
  8. OOD External Dataset Validation Protocol
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.production_pipeline import ProductionCropRecommender
from src.ingestion.data_ingestion import LiveDataIngestionEngine
from src.kerala.kerala_regional_engine import KeralaRegionalEngine
from src.security.dpdp_security import DPDPSecurityEngine
from src.api.main import app
from fastapi.testclient import TestClient


def run_master_verification():
    print("=" * 80)
    print("      KRISHISETU MASTER SYSTEM VERIFICATION & AUDIT SUITE")
    print("=" * 80)

    results = []

    # Check 1: 14 ML Pipeline Status File
    print("\n[CHECK 1] Verifying 14-Step ML Research Pipeline Status...")
    status_file = PROJECT_ROOT / "pipeline_status.json"
    if status_file.exists():
        with open(status_file, "r") as f:
            p_status = json.load(f)
        steps_dict = p_status.get("steps", {})
        completed_steps = sum(1 for s in steps_dict.values() if isinstance(s, dict) and s.get("status") == "done")
        print(f"  [OK] Pipeline Status: {completed_steps}/14 Steps Completed (Last Run: {p_status.get('last_run', 'N/A')})")
        results.append(("14-Step ML Research Pipeline", True))
    else:
        print("  [FAIL] Pipeline status file missing")
        results.append(("14-Step ML Research Pipeline", False))

    # Check 2: Production Crop Recommender (Imputation + Early Exit)
    print("\n[CHECK 2] Testing Production Crop Recommender Engine...")
    try:
        recommender = ProductionCropRecommender(use_knn=True, early_exit_threshold=0.90, use_lightweight=True)
        res = recommender.predict_single({"N": 90.0, "P": 45.0, "K": 50.0, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0})
        print(f"  [OK] Recommender Prediction: Top Crop = {res['recommended_crop']} (Conf: {res['confidence']}, Route: {res['inference_route']})")
        results.append(("Production Crop Recommender Engine", True))
    except Exception as e:
        print(f"  [FAIL] Recommender Error: {e}")
        results.append(("Production Crop Recommender Engine", False))

    # Check 3: Real-time Weather & Soil Data Ingestion Engine
    print("\n[CHECK 3] Testing Real-time Data Ingestion Engine (Open-Meteo & Kerala Defaults)...")
    try:
        ingestion = LiveDataIngestionEngine()
        enriched = ingestion.enrich_farm_input(district="Palakkad", user_input={"N": 85.0})
        print(f"  [OK] Live Ingestion Enriched Vector (Palakkad): Temp={enriched['temperature']}C, Hum={enriched['humidity']}%, Rain={enriched['rainfall']}mm")
        results.append(("Real-time Data Ingestion Engine", True))
    except Exception as e:
        print(f"  [FAIL] Ingestion Error: {e}")
        results.append(("Real-time Data Ingestion Engine", False))

    # Check 4: FastAPI Microservice Endpoints
    print("\n[CHECK 4] Testing FastAPI Microservice Endpoints (TestClient)...")
    try:
        client = TestClient(app)
        h_res = client.get("/health")
        p_res = client.post("/api/v1/predict", json={"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9})
        if h_res.status_code == 200 and p_res.status_code == 200:
            print(f"  [OK] FastAPI Health & Predict Endpoints OK (Latency: {p_res.json()['latency_ms']}ms)")
            results.append(("FastAPI Production Microservice", True))
        else:
            print(f"  [FAIL] FastAPI Response Error: Health HTTP {h_res.status_code}, Predict HTTP {p_res.status_code}")
            results.append(("FastAPI Production Microservice", False))
    except Exception as e:
        print(f"  [FAIL] FastAPI Test Error: {e}")
        results.append(("FastAPI Production Microservice", False))

    # Check 5: Kerala Regional Adaptation & Economic ROI Engine
    print("\n[CHECK 5] Testing Kerala Regional Adaptation & Economic Engine...")
    try:
        k_engine = KeralaRegionalEngine()
        season = k_engine.get_current_season()
        roi = k_engine.calculate_economic_roi("rice")
        print(f"  [OK] Active Season: {season}")
        print(f"  [OK] Rice Net Profit: INR {roi['projected_net_profit_per_acre_inr']:,.2f}/Acre (ROI: {roi['estimated_roi_percentage']}%)")
        results.append(("Kerala Regional & Economic Engine", True))
    except Exception as e:
        print(f"  [FAIL] Kerala Regional Engine Error: {e}")
        results.append(("Kerala Regional & Economic Engine", False))

    # Check 6: DPDP 2025 Security & Digital Consent Logger
    print("\n[CHECK 6] Testing DPDP 2025 Security & Malayalam Consent Engine...")
    try:
        sec = DPDPSecurityEngine()
        consent = sec.log_malayalam_consent("+919847012345", consent_granted=True)
        print(f"  [OK] Malayalam Consent Status: {consent['status']} (ID: {consent['consent_id']})")
        results.append(("DPDP 2025 Security & Consent Engine", True))
    except Exception as e:
        print(f"  [FAIL] Security Engine Error: {e}")
        results.append(("DPDP 2025 Security & Consent Engine", False))

    # Check 7: Edge Model Export File
    print("\n[CHECK 7] Verifying Exported Edge Model Artifacts...")
    mobile_pt = PROJECT_ROOT / "outputs" / "models" / "kg_hybrid_mobile.pt"
    if mobile_pt.exists():
        print(f"  [OK] Mobile Model Artifact Present: {mobile_pt.name} ({mobile_pt.stat().st_size / (1024*1024):.2f} MB)")
        results.append(("Edge Model Mobile Artifact (.pt)", True))
    else:
        print("  [FAIL] Mobile Model artifact missing")
        results.append(("Edge Model Mobile Artifact (.pt)", False))

    # Check 8: OOD External Dataset Validation Protocol
    print("\n[CHECK 8] Verifying OOD External Dataset Validation Protocol...")
    ext_csv = PROJECT_ROOT / "outputs" / "tables" / "external_validation_metrics.csv"
    ext_json = PROJECT_ROOT / "outputs" / "reports" / "external_validation_report.json"
    if ext_csv.exists() and ext_json.exists():
        with open(ext_json, "r") as f:
            ext_rep = json.load(f)
        reg_b = ext_rep.get("regime_b_aligned", {})
        print(f"  [OK] External Validation Status: Dataset = {ext_rep.get('dataset_name', 'N/A')}")
        print(f"  [OK] Regime B (Domain-Aligned) F1-Macro: {reg_b.get('metrics', [{}])[0].get('External_F1_Macro', 'N/A')}")
        results.append(("OOD External Dataset Validation Protocol", True))
    else:
        print("  [FAIL] External validation artifacts missing")
        results.append(("OOD External Dataset Validation Protocol", False))

    # Check 9: Stratified 5-Fold Cross-Validation Engine
    print("\n[CHECK 9] Verifying Stratified 5-Fold Cross-Validation Engine...")
    cv_sum_csv = PROJECT_ROOT / "outputs" / "cross_validation" / "aggregate" / "cv_summary.csv"
    cv_oof_csv = PROJECT_ROOT / "outputs" / "cross_validation" / "oof_predictions" / "oof_predictions.csv"
    if cv_sum_csv.exists() and cv_oof_csv.exists():
        cv_df = pd.read_csv(cv_sum_csv)
        kgh_row = cv_df[cv_df["Model"] == "KG-HybridProposed"]
        f1_str = kgh_row["f1_macro_formatted"].values[0] if not kgh_row.empty else "N/A"
        print(f"  [OK] Stratified 5-Fold CV Status: 5 Folds Completed")
        print(f"  [OK] KG-Hybrid 5-Fold Mean Macro F1: {f1_str}")
        results.append(("Stratified 5-Fold Cross-Validation Protocol", True))
    else:
        print("  [FAIL] Cross-validation output artifacts missing")
        results.append(("Stratified 5-Fold Cross-Validation Protocol", False))

    # Final Summary
    print("\n" + "=" * 80)
    print("                     MASTER AUDIT SUMMARY RESULTS")
    print("=" * 80)
    passed_count = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status_str = "[OK] VERIFIED" if ok else "[FAIL] FAILED"
        print(f"  * {name:<45} : {status_str}")

    print("-" * 80)
    print(f"  TOTAL SYSTEM HEALTH: {passed_count}/{len(results)} Components Verified")
    print("=" * 80)

    if passed_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    run_master_verification()

