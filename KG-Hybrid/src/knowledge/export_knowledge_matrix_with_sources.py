"""
Agricultural Knowledge Matrix Export Script (Stage 7 - Provenance Export)
KrishiSetu Q1 ML Core — Knowledge-Guided Hybrid Crop Recommendation

Exports cardinal bounds with declared guidance and explicit verification status.
This is not the active fusion prior; export_active_prior exports that matrix.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.knowledge.knowledge_base import KNOWLEDGE_BASE


def export_knowledge_matrix_csv():
    """Queries KNOWLEDGE_BASE and exports provenance-annotated CSV table."""
    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    param_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # Historical declared guidance, not verified cell-level citations.
    def get_source_info(crop: str, param: str):
        if crop in ["rice", "maize", "chickpea", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"]:
            return "ICAR Package of Practices / FAO EcoCrop Database (2023)", "National Agricultural Research Institution"
        elif crop in ["pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya"]:
            return "NHB Indian Horticulture Database / ICAR-IIHR Guidelines (2023)", "Horticultural Research Board"
        elif crop in ["coconut", "coffee", "jute", "cotton"]:
            return "KAU Ad-hoc Package of Practices / CPCRI Advisory Manual (2024)", "State Agricultural University"
        else:
            return "FAO EcoCrop Global Agronomic Database (2023)", "International Organization"

    for crop_name, kb in KNOWLEDGE_BASE.items():
        bounds_map = {
            "N": kb.N_bounds,
            "P": kb.P_bounds,
            "K": kb.K_bounds,
            "temperature": kb.temp_bounds,
            "humidity": kb.humidity_bounds,
            "ph": kb.ph_bounds,
            "rainfall": kb.rainfall_bounds
        }

        for param in param_names:
            b = bounds_map[param]
            source_name, source_type = get_source_info(crop_name, param)
            records.append({
                "Crop_Name": crop_name,
                "Parameter": param,
                "Min_Boundary": b.min_val,
                "Optimum_Low": b.opt_low,
                "Optimum_High": b.opt_high,
                "Max_Boundary": b.max_val,
                "Institutional_Source": source_name,
                "Source_Type": source_type,
                "Source_Verification": "UNVERIFIED: source document, page and exact extracted range not retained",
                "Representation": "Cardinal-bound knowledge base; not the active fusion weight matrix",
            })

    df = pd.DataFrame(records)
    out_path = output_dir / "agricultural_knowledge_matrix_with_sources.csv"
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Exported Agricultural Knowledge Matrix with Sources to {out_path} ({len(df)} entries)")
    return df


if __name__ == "__main__":
    export_knowledge_matrix_csv()
