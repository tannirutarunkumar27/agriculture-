"""
Knowledge Sources & Conflict Resolution Engine
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Manages provenance metadata and implements scientifically defensible resolution
strategies when reputable institutional sources report divergent agronomic bounds.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from src.knowledge.knowledge_schema import (
    CardinalBounds,
    KnowledgeSourceRecord,
    SourceType
)


class KnowledgeConflictResolver:
    """
    Scientifically resolves conflicting threshold ranges reported across
    multiple agricultural institutions (e.g. FAO vs ICAR vs KAU).
    """

    @staticmethod
    def resolve_bounds(
        sources: List[KnowledgeSourceRecord],
        raw_tuples: List[Tuple[float, float, float, float]]
    ) -> CardinalBounds:
        """
        Resolves conflicting range entries:
          1. f_min = min(f_min_sources)  [Outer Boundary Union]
          2. f_max = max(f_max_sources)  [Outer Boundary Union]
          3. f_opt_low = mean(f_opt_low_sources) [Optimal Consensus]
          4. f_opt_high = mean(f_opt_high_sources) [Optimal Consensus]
        """
        if not raw_tuples:
            raise ValueError("No raw parameter tuples provided for resolution!")

        mins = [t[0] for t in raw_tuples]
        opt_lows = [t[1] for t in raw_tuples]
        opt_highs = [t[2] for t in raw_tuples]
        maxs = [t[3] for t in raw_tuples]

        resolved_min = float(np.min(mins))
        resolved_opt_low = float(np.mean(opt_lows))
        resolved_opt_high = float(np.mean(opt_highs))
        resolved_max = float(np.max(maxs))

        # Enforce strict physiological ordering post-resolution
        if resolved_opt_low < resolved_min:
            resolved_opt_low = resolved_min
        if resolved_opt_high > resolved_max:
            resolved_opt_high = resolved_max
        if resolved_opt_low > resolved_opt_high:
            mid = (resolved_opt_low + resolved_opt_high) / 2.0
            resolved_opt_low = mid
            resolved_opt_high = mid

        return CardinalBounds(
            min_val=resolved_min,
            opt_low=resolved_opt_low,
            opt_high=resolved_opt_high,
            max_val=resolved_max
        )


# Global Sourcing Registry
PROVENANCE_REGISTRY: List[KnowledgeSourceRecord] = []


def register_source(
    crop_name: str,
    parameter_name: str,
    reported_range: str,
    unit: str,
    source_name: str = "FAO EcoCrop Database / ICAR Package of Practices",
    source_type: SourceType = SourceType.INTERNATIONAL_ORG,
    confidence_score: int = 5,
    notes: Optional[str] = None
) -> KnowledgeSourceRecord:
    record = KnowledgeSourceRecord(
        source_name=source_name,
        source_type=source_type,
        crop_name=crop_name,
        parameter_name=parameter_name,
        reported_range=reported_range,
        unit=unit,
        confidence_score=confidence_score,
        notes=notes
    )
    PROVENANCE_REGISTRY.append(record)
    return record
