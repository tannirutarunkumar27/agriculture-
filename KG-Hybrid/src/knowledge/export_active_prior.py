"""Export the exact active 22 x 7 prior with explicit provenance limits."""
import json
import hashlib
from pathlib import Path

import pandas as pd
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix


def export_active_prior():
    root = Path(__file__).resolve().parents[2]
    matrix, crops, features = load_authoritative_knowledge_matrix()
    tables = root / "outputs/tables"
    frame = pd.DataFrame(matrix, index=crops, columns=features)
    frame.index.name = "crop"
    frame["Source"] = "src/knowledge/build_knowledge_matrix.py: crop_knowledge explicit weights"
    frame["Source_Verification"] = "Institutional references are claimed guidance; cell-level source derivation unverified"
    frame.to_csv(tables / "active_knowledge_prior_with_provenance.csv")
    metadata = {
        "shape": list(matrix.shape), "classes": crops, "features": features,
        "matrix_sha256_float32": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "definition": "Manually specified crop-feature profile weights; exact values are the published prior",
        "range_to_weight_rule": None,
        "range_to_weight_rule_status": "No transformation from agronomic bounds to these weights exists in the implementation",
        "institutional_source_verification": "UNVERIFIED: no per-cell document/page/source-range records",
        "warning": "The separate cardinal-bound knowledge base is not the source of this matrix; do not conflate the two representations",
        "supported_policies": {"fixed": "K=K0", "anchored_trainable": "loss += lambda * mean((K-K0)^2), lambda explicitly recorded"},
        "learned_bounds": "L2 anchoring does not impose hard box constraints; report drift and extrema",
    }
    (tables / "active_knowledge_prior_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    print(export_active_prior().to_string())
