"""
FastAPI Automated Unit & Integration Test Suite
KrishiSetu Platform — Phase 1 (Milestone 1 API)
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test API root metadata endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "KrishiSetu Crop Recommendation API"
    assert data["status"] == "RUNNING"


def test_health_endpoint():
    """Test health and readiness endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert "v1.2.0" in data["model_version"]


def test_predict_single_success():
    """Test single crop recommendation with valid input."""
    payload = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
        "district": "Palakkad"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "recommended_crop" in data
    assert len(data["top_3_crops"]) == 3
    assert data["confidence"] > 0.0
    assert data["latency_ms"] > 0.0


def test_predict_single_missing_sensor():
    """Test crop recommendation with missing N and ph sensors (KNN Imputation)."""
    payload = {
        "N": None,  # Missing sensor
        "P": 45.0,
        "K": 40.0,
        "temperature": 24.5,
        "humidity": 80.0,
        "ph": None, # Missing sensor
        "rainfall": 210.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "IMPUTED" in data["data_quality"]
    assert "WARNING" in data["sensor_health"]


def test_predict_invalid_out_of_bounds():
    """Test Pydantic schema validation error on invalid input bounds."""
    payload = {
        "N": 999.0,  # Invalid (>300)
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_batch():
    """Test batch prediction with multiple farms."""
    payload = {
        "farms": [
            {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
            {"N": 20.0, "P": 120.0, "K": 200.0, "temperature": 27.0, "humidity": 80.0, "ph": 5.5, "rainfall": 100.0},
            {"N": 60.0, "P": 55.0, "K": 40.0, "temperature": 25.0, "humidity": 70.0, "ph": 6.0, "rainfall": 150.0}
        ]
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_farms"] == 3
    assert len(data["recommendations"]) == 3
    assert data["total_latency_ms"] > 0.0


def test_get_districts():
    """Test Kerala district profiles endpoint."""
    response = client.get("/api/v1/districts")
    assert response.status_code == 200
    districts = response.json()
    assert len(districts) >= 5
    district_names = [d["district_name"] for d in districts]
    assert any("Palakkad" in name for name in district_names)


def test_metrics_endpoint():
    """Test Prometheus plain-text metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "krishisetu_requests_total" in response.text
    assert "krishisetu_predictions_total" in response.text


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING KRISHISETU FASTAPI INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Root Metadata", test_root_endpoint),
        ("Health & Readiness", test_health_endpoint),
        ("Single Prediction", test_predict_single_success),
        ("Missing Sensor Imputation", test_predict_single_missing_sensor),
        ("Schema Out-of-Bounds Validation", test_predict_invalid_out_of_bounds),
        ("Batch Prediction", test_predict_batch),
        ("Kerala Districts Profiles", test_get_districts),
        ("Prometheus Metrics Export", test_metrics_endpoint)
    ]

    passed = 0
    for name, test_func in tests:
        try:
            test_func()
            print(f"  ✓ PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {name} — Error: {e}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed}/{len(tests)} Tests Passed Successfully!")
    print("=" * 70)
    if passed < len(tests):
        sys.exit(1)
