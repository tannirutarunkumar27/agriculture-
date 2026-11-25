"""Record actual source identities and blocked raw-feature approval contracts.

Does not infer units from column names or train/evaluate any model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper2.data import file_hash, load_dataset
from paper2.knowledge import KnowledgeGuidedMLP, RuleBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    registry_path = root / "paper2/configs/knowledge_registry_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name in ("canonical", "maharashtra"):
        frame, schema, source = load_dataset(name)
        contract = {"dataset": name, "source_sha256": file_hash(source),
                    "features": [{"name": feature, "unit": None, "measurement": None,
                                  "period": None, "verified": False, "evidence": None}
                                 for feature in schema["numeric"]]}
        classes = sorted(frame[schema["target"]].astype(str).unique().tolist())
        bank = RuleBank(registry, contract, classes)
        try:
            KnowledgeGuidedMLP(len(schema["numeric"]), 64, bank, alpha=1)
        except ValueError as exc:
            reason = str(exc)
        else:
            raise RuntimeError("Unapproved real-data guidance unexpectedly enabled")
        records[name] = {"classes": classes, "raw_contract": contract,
                         "compatible_approved_rules": bank.rule_count,
                         "guidance_enabled": False, "block_reason": reason,
                         "contract_status": "draft_not_measurement_certification"}
    report = {"status": "BLOCKED_FOR_REAL_DATA_GUIDANCE", "datasets": records,
              "registry_sha256": file_hash(registry_path),
              "code_sha256": {p.name: file_hash(p) for p in
                              [root / "paper2/knowledge.py", root / "paper2/knowledge_training.py", Path(__file__)]},
              "training_performed": False, "test_evaluation_performed": False,
              "privacy_implemented": False}
    with (args.output / "gate_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "datasets": list(records),
                      "approved_rules": 0, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
