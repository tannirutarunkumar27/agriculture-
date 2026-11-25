import copy
import json
from pathlib import Path
import tempfile
import unittest

from paper2.study_protocol import build_run_plan, freeze_protocol, validate_protocol, verify_frozen_protocol

TEMPLATE = Path(__file__).resolve().parents[1]/"configs/baseline_study_v1.json"


class TestStudyProtocol(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(TEMPLATE.read_text(encoding="utf-8"))

    def test_complete_unique_run_plan(self):
        runs = build_run_plan(self.protocol)
        self.assertEqual(len(runs),235)
        self.assertEqual(len({run["id"] for run in runs}),235)
        self.assertEqual(sum(run["scenario"]=="centralized_reference" for run in runs),40)

    def test_same_model_settings_across_algorithms(self):
        runs = build_run_plan(self.protocol)
        by_condition = {}
        for run in runs:
            if run["scenario"]=="centralized_reference":
                continue
            key = (run["dataset"],run["seed"],run["scenario"])
            if key in by_condition:
                self.assertEqual(by_condition[key],run["config"])
            by_condition[key] = run["config"]

    def test_partial_participation_reuses_iid_partition(self):
        runs = build_run_plan(self.protocol)
        candidates = [r for r in runs if r["dataset"]=="canonical" and r["seed"]==123 and r["algorithm"]=="fedavg" and r["scenario"].startswith("iid")]
        self.assertEqual(len({r["partition"] for r in candidates}),1)
        self.assertEqual({r["config"]["participation"] for r in candidates},{1.0,0.6,0.2})

    def test_tuning_seed_rejected(self):
        self.protocol["seeds"].append(42)
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol)

    def test_duplicate_seed_rejected(self):
        self.protocol["seeds"].append(123)
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol)

    def test_fake_regions_rejected(self):
        self.protocol["scenarios"][-1]["datasets"] = ["canonical"]
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol)

    def test_privacy_and_knowledge_claims_rejected(self):
        for key in ["formal_privacy","knowledge_enabled","automatic_hyperparameter_changes","skip_failed_clients"]:
            protocol = copy.deepcopy(self.protocol)
            protocol["safety"][key] = True
            with self.assertRaises(ValueError):
                validate_protocol(protocol)

    def test_unsafe_scenario_name_rejected(self):
        self.protocol["scenarios"][0]["id"] = "../overwrite"
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol)

    def test_test_tuning_policy_rejected(self):
        self.protocol["evaluation"] = "pick_best_test_accuracy"
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol)


class TestFrozenArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        protocol = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        protocol["datasets"] = {"canonical":protocol["datasets"]["canonical"]}
        protocol["seeds"] = [123,2024,3407]
        protocol["scenarios"] = [s for s in protocol["scenarios"] if "canonical" in s["datasets"]]
        for scenario in protocol["scenarios"]:
            scenario["datasets"] = ["canonical"]
        cls.config = cls.root/"config.json"
        cls.config.write_text(json.dumps(protocol),encoding="utf-8")
        cls.frozen = cls.root/"frozen"
        cls.manifest = freeze_protocol(cls.config,cls.frozen)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_freeze_without_training(self):
        report = verify_frozen_protocol(self.frozen)
        self.assertEqual(report["status"],"PASS")
        self.assertEqual(report["planned_runs"],66)
        self.assertEqual(report["partition_count"],12)
        self.assertFalse(report["training_started"])
        self.assertFalse(list(self.frozen.rglob("*.pt")))

    def test_existing_freeze_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            freeze_protocol(self.config,self.frozen)

    def test_protocol_tampering_detected(self):
        path = self.frozen/"protocol.json"
        original = path.read_bytes()
        try:
            protocol = json.loads(original)
            protocol["datasets"]["canonical"]["neural"]["learning_rate"] = 0.3
            path.write_text(json.dumps(protocol),encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_frozen_protocol(self.frozen)
        finally:
            path.write_bytes(original)

    def test_partition_tampering_detected(self):
        path = next((self.frozen/"partitions").glob("*.json"))
        original = path.read_bytes()
        try:
            path.write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_frozen_protocol(self.frozen)
        finally:
            path.write_bytes(original)

    def test_unapproved_hyperparameter_change_rejected(self):
        protocol = json.loads(self.config.read_text())
        protocol["datasets"]["canonical"]["neural"]["hidden"] = 128
        config = self.root/"changed.json"
        config.write_text(json.dumps(protocol),encoding="utf-8")
        with self.assertRaises(ValueError):
            freeze_protocol(config,self.root/"unapproved")
