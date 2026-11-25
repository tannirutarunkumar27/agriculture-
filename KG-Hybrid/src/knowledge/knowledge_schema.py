"""
Agronomic Knowledge Base Schema Definitions
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Defines fundamental type structures and dataclasses for storing:
  1. 4-Point Cardinal Parameter Bounds (min, opt_low, opt_high, max)
  2. Crop Knowledge Profiles (across 7 primary soil-climate dimensions)
  3. Institutional Source Provenance & Provenance Metadata
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional


class SourceType(Enum):
    """Institutional categories for agricultural knowledge sourcing."""
    GOVERNMENT_AGRI_DEPT = "Government Agricultural Department"
    UNIVERSITIES = "Agricultural University Research"
    INTERNATIONAL_ORG = "International Organization (FAO/CGIAR)"
    PEER_REVIEWED_JOURNAL = "Peer-Reviewed Agronomic Paper"


@dataclass(frozen=True)
class CardinalBounds:
    """
    4-Point Cardinal Parameter Threshold Tuple for physiological tolerance limits:
    [f_min, f_opt_low, f_opt_high, f_max]
    """
    min_val: float
    opt_low: float
    opt_high: float
    max_val: float

    def __post_init__(self):
        if not (self.min_val <= self.opt_low <= self.opt_high <= self.max_val):
            raise ValueError(
                f"Invalid physiological cardinal ordering: "
                f"min({self.min_val}) <= opt_low({self.opt_low}) <= opt_high({self.opt_high}) <= max({self.max_val}) violated!"
            )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_val, self.opt_low, self.opt_high, self.max_val)


@dataclass
class KnowledgeSourceRecord:
    """Provenential record documenting institutional origin of agronomic bounds."""
    source_name: str
    source_type: SourceType
    crop_name: str
    parameter_name: str
    reported_range: str
    unit: str
    confidence_score: int  # 1 (Low) to 5 (High)
    notes: Optional[str] = None


@dataclass
class CropKnowledgeProfile:
    """
    Complete Agronomic Knowledge Profile for a single crop class,
    covering the 7 core dataset input features + seasonal & soil metadata.
    """
    crop_name: str
    N_bounds: CardinalBounds
    P_bounds: CardinalBounds
    K_bounds: CardinalBounds
    temp_bounds: CardinalBounds
    humidity_bounds: CardinalBounds
    ph_bounds: CardinalBounds
    rainfall_bounds: CardinalBounds
    growing_seasons: List[str] = field(default_factory=list)
    preferred_soil_types: List[str] = field(default_factory=list)

    def get_bounds_for_feature(self, feature_name: str) -> CardinalBounds:
        mapping = {
            "N": self.N_bounds,
            "P": self.P_bounds,
            "K": self.K_bounds,
            "temperature": self.temp_bounds,
            "humidity": self.humidity_bounds,
            "ph": self.ph_bounds,
            "rainfall": self.rainfall_bounds
        }
        if feature_name not in mapping:
            raise KeyError(f"Feature '{feature_name}' not in profile for crop {self.crop_name}")
        return mapping[feature_name]
