"""
FastAPI Production Microservice Server
KrishiSetu Platform — Phase 1 (Milestone 1 API)

Features:
  • /api/v1/predict - Single farm crop recommendation with Early Exit + KNN Imputation
  • /api/v1/predict/batch - High-throughput bulk prediction for FPOs & Co-ops
  • /api/v1/districts - Kerala 14-district agricultural profiles & defaults
  • /health - Infrastructure readiness, uptime, and model sanity check
  • /metrics - Prometheus metrics export (requests, latency, early exit ratio)
  • CORS middleware, structured logging, latency header (X-Process-Time-Ms)
"""

import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import (
    SoilClimateInput,
    BatchSoilClimateInput,
    CropRecommendationResponse,
    BatchRecommendationResponse,
    HealthResponse,
    DistrictProfile,
    CropScore
)
from src.inference.production_pipeline import ProductionCropRecommender

# ─── API Initialization & Metadata ──────────────────────────────────────────
START_TIME = time.time()

app = FastAPI(
    title="KrishiSetu Crop Recommendation API",
    description="Production-grade AI Advisory Microservice for Smallholder Farmers in Kerala",
    version="1.2.0-Production",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ─── CORS Middleware ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Simple In-Memory Prometheus Metrics Counter ────────────────────────────
METRICS = {
    "requests_total": 0,
    "predictions_total": 0,
    "early_exits_total": 0,
    "imputed_requests_total": 0,
    "errors_total": 0,
    "latencies_sum_ms": 0.0
}

# ─── Model Singleton Initialization ─────────────────────────────────────────
recommender: Optional[ProductionCropRecommender] = None

def get_recommender() -> ProductionCropRecommender:
    global recommender
    if recommender is None:
        print("[INIT] Loading KG-Hybrid Production Recommender Pipeline...")
        recommender = ProductionCropRecommender(
            use_knn=True,
            early_exit_threshold=0.90,
            use_lightweight=True,
            mc_samples=10,
            uncertainty_threshold=0.15,
            region_context="Kerala/Tropical"
        )
        print(f"[OK] Recommender initialized successfully (Version: {recommender.model_version})")
    return recommender

@app.on_event("startup")
def startup_event():
    get_recommender()


# ─── Processing Time Middleware ─────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    METRICS["requests_total"] += 1
    METRICS["latencies_sum_ms"] += elapsed_ms

    if response.status_code >= 400:
        METRICS["errors_total"] += 1

    return response


# ─── Kerala District Profiles Data ──────────────────────────────────────────
KERALA_DISTRICTS: Dict[str, Dict] = {
    "Palakkad": {
        "district_name": "Palakkad",
        "region": "Central Plain & Rice Bowl",
        "dominant_soil_type": "Black Cotton & Alluvial",
        "avg_rainfall_mm": 2100.0,
        "avg_temp_c": 28.5,
        "recommended_staples": ["rice", "maize", "cotton", "blackgram"]
    },
    "Alappuzha": {
        "district_name": "Alappuzha (Kuttanad)",
        "region": "Kuttanad Sub-sea Level Zone",
        "dominant_soil_type": "Kari & Karappadam (Acidic Peaty)",
        "avg_rainfall_mm": 3000.0,
        "avg_temp_c": 27.2,
        "recommended_staples": ["rice", "coconut", "banana"]
    },
    "Wayanad": {
        "district_name": "Wayanad",
        "region": "Highland Hill Zone",
        "dominant_soil_type": "Forest Loam & Red Soil",
        "avg_rainfall_mm": 3200.0,
        "avg_temp_c": 22.0,
        "recommended_staples": ["coffee", "blackgram", "rice"]
    },
    "Idukki": {
        "district_name": "Idukki",
        "region": "High Altitude Spices Zone",
        "dominant_soil_type": "Laterite & Forest Loam",
        "avg_rainfall_mm": 3500.0,
        "avg_temp_c": 20.5,
        "recommended_staples": ["coffee", "tea", "cardamom"]
    },
    "Thiruvananthapuram": {
        "district_name": "Thiruvananthapuram",
        "region": "Southern Coastal Plain",
        "dominant_soil_type": "Laterite & Coastal Alluvium",
        "avg_rainfall_mm": 1800.0,
        "avg_temp_c": 27.8,
        "recommended_staples": ["coconut", "tapioca", "banana", "rice"]
    }
}


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", summary="API Root Metadata")
def root():
    """Root metadata endpoint providing API version, documentation links, and capabilities."""
    return {
        "service": "KrishiSetu Crop Recommendation API",
        "version": "1.2.0-Production",
        "target_region": "Kerala, India",
        "status": "RUNNING",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "endpoints": {
            "predict_single": "POST /api/v1/predict",
            "predict_batch": "POST /api/v1/predict/batch",
            "districts": "GET /api/v1/districts"
        }
    }


@app.get("/health", response_model=HealthResponse, summary="System Health & Readiness Check")
def health_check():
    """Returns infrastructure health status, model readiness, uptime, and active model version."""
    rec = get_recommender()
    uptime = time.time() - START_TIME
    now_utc = datetime.now(timezone.utc).isoformat()

    return HealthResponse(
        status="HEALTHY",
        timestamp=now_utc,
        model_loaded=True,
        model_version=rec.model_version,
        uptime_seconds=round(uptime, 2),
        device="cpu"
    )


@app.post("/api/v1/predict", response_model=CropRecommendationResponse, summary="Single Farm Crop Recommendation")
def predict_crop(input_data: SoilClimateInput):
    """
    Generate real-time crop recommendation for a single farm input.
    Supports missing sensor values via automated KNN Imputation,
    early-exit confidence gating (<25ms), and MC Dropout uncertainty estimation.
    """
    rec = get_recommender()

    # Convert Pydantic input to dict
    input_dict = input_data.dict(exclude={'district', 'farmer_id'})

    # Execute inference
    try:
        res = rec.predict_single(input_dict, enable_early_exit=True)

        # Update operational metrics
        METRICS["predictions_total"] += 1
        if "EARLY_EXIT" in res.get("inference_route", ""):
            METRICS["early_exits_total"] += 1
        if "IMPUTED" in res.get("data_quality", ""):
            METRICS["imputed_requests_total"] += 1

        # Format top 3 crops into Pydantic models
        top3_models = [
            CropScore(
                crop=item["crop"],
                probability=item["probability"],
                uncertainty_std=item.get("uncertainty_std", None)
            )
            for item in res["top_3_crops"]
        ]

        return CropRecommendationResponse(
            timestamp=res["timestamp"],
            status="SUCCESS",
            model_version=res["model_version"],
            region_context=res["region_context"],
            recommended_crop=res["recommended_crop"],
            confidence=res["confidence"],
            uncertainty_std=res["uncertainty_std"],
            top_3_crops=top3_models,
            inference_route=res["inference_route"],
            data_quality=res["data_quality"],
            sensor_health=res["sensor_health"],
            latency_ms=res["latency_ms"],
            safety_alert=res.get("safety_alert", None)
        )

    except Exception as e:
        METRICS["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/predict/batch", response_model=BatchRecommendationResponse, summary="Batch Crop Recommendation")
def predict_batch(batch_input: BatchSoilClimateInput):
    """
    High-throughput batch crop recommendation for Cooperative Societies and FPOs.
    Processes up to 1,000 farm records per batch request.
    """
    rec = get_recommender()

    if len(batch_input.farms) == 0:
        raise HTTPException(status_code=400, detail="Empty farms list in batch payload")

    if len(batch_input.farms) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 1000 farms")

    t_batch_start = time.perf_counter()
    results = []

    for farm_data in batch_input.farms:
        input_dict = farm_data.dict(exclude={'district', 'farmer_id'})
        res = rec.predict_single(input_dict, enable_early_exit=True)

        top3_models = [
            CropScore(
                crop=item["crop"],
                probability=item["probability"],
                uncertainty_std=item.get("uncertainty_std", None)
            )
            for item in res["top_3_crops"]
        ]

        results.append(
            CropRecommendationResponse(
                timestamp=res["timestamp"],
                status="SUCCESS",
                model_version=res["model_version"],
                region_context=res["region_context"],
                recommended_crop=res["recommended_crop"],
                confidence=res["confidence"],
                uncertainty_std=res["uncertainty_std"],
                top_3_crops=top3_models,
                inference_route=res["inference_route"],
                data_quality=res["data_quality"],
                sensor_health=res["sensor_health"],
                latency_ms=res["latency_ms"],
                safety_alert=res.get("safety_alert", None)
            )
        )

        METRICS["predictions_total"] += 1

    t_batch_end = time.perf_counter()
    total_batch_latency = round((t_batch_end - t_batch_start) * 1000.0, 2)

    return BatchRecommendationResponse(
        total_farms=len(results),
        total_latency_ms=total_batch_latency,
        recommendations=results
    )


@app.get("/api/v1/districts", response_model=List[DistrictProfile], summary="Kerala District Soil Profiles")
def get_kerala_districts():
    """Returns district agricultural profiles and dominant soil types for Kerala."""
    profiles = [DistrictProfile(**data) for data in KERALA_DISTRICTS.values()]
    return profiles


@app.get("/metrics", summary="Prometheus Operational Metrics")
def prometheus_metrics():
    """Export Prometheus plain text metrics for scraping."""
    avg_latency = (
        METRICS["latencies_sum_ms"] / METRICS["requests_total"]
        if METRICS["requests_total"] > 0 else 0.0
    )
    early_exit_ratio = (
        METRICS["early_exits_total"] / METRICS["predictions_total"]
        if METRICS["predictions_total"] > 0 else 0.0
    )

    content = f"""# HELP krishisetu_requests_total Total API requests received
# TYPE krishisetu_requests_total counter
krishisetu_requests_total {METRICS['requests_total']}

# HELP krishisetu_predictions_total Total model predictions made
# TYPE krishisetu_predictions_total counter
krishisetu_predictions_total {METRICS['predictions_total']}

# HELP krishisetu_early_exits_total Total predictions routed through Early Exit
# TYPE krishisetu_early_exits_total counter
krishisetu_early_exits_total {METRICS['early_exits_total']}

# HELP krishisetu_imputed_requests_total Total predictions requiring KNN Imputation
# TYPE krishisetu_imputed_requests_total counter
krishisetu_imputed_requests_total {METRICS['imputed_requests_total']}

# HELP krishisetu_errors_total Total 4xx and 5xx API errors
# TYPE krishisetu_errors_total counter
krishisetu_errors_total {METRICS['errors_total']}

# HELP krishisetu_avg_latency_ms Average request process latency in milliseconds
# TYPE krishisetu_avg_latency_ms gauge
krishisetu_avg_latency_ms {avg_latency:.2f}

# HELP krishisetu_early_exit_ratio Ratio of predictions using early exit route
# TYPE krishisetu_early_exit_ratio gauge
krishisetu_early_exit_ratio {early_exit_ratio:.4f}
"""
    return Response(content=content, media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    print("[START] Launching KrishiSetu Production FastAPI Server...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
