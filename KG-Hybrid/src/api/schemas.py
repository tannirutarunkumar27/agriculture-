"""
FastAPI Request & Response Schemas (Pydantic Models)
KrishiSetu Production API — Phase 1 (Milestone 1)
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class SoilClimateInput(BaseModel):
    """Input payload for single farm crop recommendation."""
    N: Optional[float] = Field(None, description="Nitrogen content in soil (kg/ha). Range: 0–140", example=90.0)
    P: Optional[float] = Field(None, description="Phosphorus content in soil (kg/ha). Range: 0–145", example=42.0)
    K: Optional[float] = Field(None, description="Potassium content in soil (kg/ha). Range: 0–205", example=43.0)
    temperature: Optional[float] = Field(None, description="Temperature in Celsius (°C). Range: 8–45", example=20.8)
    humidity: Optional[float] = Field(None, description="Relative humidity (%). Range: 14–100", example=82.0)
    ph: Optional[float] = Field(None, description="Soil pH level. Range: 3.5–9.9", example=6.5)
    rainfall: Optional[float] = Field(None, description="Annual/Seasonal rainfall (mm). Range: 20–300", example=202.9)
    district: Optional[str] = Field(None, description="Kerala district name (optional context)", example="Palakkad")
    farmer_id: Optional[str] = Field(None, description="Optional anonymized farmer UUID", example="f83a21b4-92e1-4c12-b9e3")

    @validator('N')
    def validate_n(cls, v):
        if v is not None and (v < 0 or v > 300):
            raise ValueError('Nitrogen must be between 0 and 300 kg/ha')
        return v

    @validator('P')
    def validate_p(cls, v):
        if v is not None and (v < 0 or v > 300):
            raise ValueError('Phosphorus must be between 0 and 300 kg/ha')
        return v

    @validator('K')
    def validate_k(cls, v):
        if v is not None and (v < 0 or v > 300):
            raise ValueError('Potassium must be between 0 and 300 kg/ha')
        return v

    @validator('temperature')
    def validate_temp(cls, v):
        if v is not None and (v < 0 or v > 60):
            raise ValueError('Temperature must be between 0°C and 60°C')
        return v

    @validator('humidity')
    def validate_humidity(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Humidity must be between 0% and 100%')
        return v

    @validator('ph')
    def validate_ph(cls, v):
        if v is not None and (v < 0 or v > 14):
            raise ValueError('pH must be between 0 and 14')
        return v

    @validator('rainfall')
    def validate_rainfall(cls, v):
        if v is not None and (v < 0 or v > 1000):
            raise ValueError('Rainfall must be between 0 and 1000 mm')
        return v


class BatchSoilClimateInput(BaseModel):
    """Batch payload for cooperative societies and bulk queries."""
    farms: List[SoilClimateInput] = Field(..., description="List of farm input objects (max 1000 per request)")


class CropScore(BaseModel):
    """Individual crop score breakdown."""
    crop: str = Field(..., description="Recommended crop name", example="rice")
    probability: float = Field(..., description="Model confidence score (0.0 to 1.0)", example=0.9852)
    uncertainty_std: Optional[float] = Field(None, description="MC Dropout uncertainty standard deviation", example=0.012)


class CropRecommendationResponse(BaseModel):
    """Structured response output for crop recommendation."""
    timestamp: str = Field(..., description="UTC ISO timestamp of prediction")
    status: str = Field("SUCCESS", description="Execution status")
    model_version: str = Field(..., description="Model version tag", example="v1.2.0-ProductionLightweight")
    region_context: str = Field(..., description="Region context tag")
    recommended_crop: str = Field(..., description="Top recommended crop for cultivation", example="rice")
    confidence: float = Field(..., description="Top recommendation confidence score", example=0.9852)
    uncertainty_std: float = Field(..., description="Uncertainty standard deviation from MC Dropout", example=0.012)
    top_3_crops: List[CropScore] = Field(..., description="Top 3 recommended crops with probabilities")
    inference_route: str = Field(..., description="Inference pipeline path (e.g. Early Exit vs Full KG-Hybrid)")
    data_quality: str = Field(..., description="Data health status and imputation details")
    sensor_health: str = Field(..., description="Sensor fault diagnostic message")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds")
    safety_alert: Optional[str] = Field(None, description="Safety alert if boundary condition or high uncertainty detected")


class BatchRecommendationResponse(BaseModel):
    """Response wrapper for batch inference."""
    total_farms: int = Field(..., description="Number of farm predictions processed")
    total_latency_ms: float = Field(..., description="Total batch processing latency in milliseconds")
    recommendations: List[CropRecommendationResponse] = Field(..., description="List of individual farm recommendations")


class HealthResponse(BaseModel):
    """System health check status response."""
    status: str = Field("HEALTHY", example="HEALTHY")
    timestamp: str = Field(..., description="Current UTC timestamp")
    model_loaded: bool = Field(True, description="Model ready status")
    model_version: str = Field(..., example="v1.2.0-ProductionLightweight")
    uptime_seconds: float = Field(..., description="API server uptime in seconds")
    device: str = Field("cpu", example="cpu")


class DistrictProfile(BaseModel):
    """Kerala district agricultural profile."""
    district_name: str
    region: str
    dominant_soil_type: str
    avg_rainfall_mm: float
    avg_temp_c: float
    recommended_staples: List[str]
