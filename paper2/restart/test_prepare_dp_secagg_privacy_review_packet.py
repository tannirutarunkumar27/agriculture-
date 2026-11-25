import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper2.restart.prepare_dp_secagg_privacy_review_packet import prepare


class DpSecAggPrivacyReviewPacketTests(unittest.TestCase):
    def fixture(self, root: Path):
        protocol, audit = root / "protocol", root / "audit.json"; protocol.mkdir()
        (protocol / "privacy_contract.json").write_text(json.dumps({
            "participants": 2, "clip_norm": 1.0, "noise_multiplier": 4.0, "delta": 1e-5,
            "adaptive_releases": 2, "flower_noise_std": 2.0,
            "conditional_budget": {"rho": 0.25, "epsilon_upper_bound": 0.25 + 2 * (0.25 ** .5) * ((-__import__('math').log(1e-5)) ** .5), "delta": 1e-5, "releases": 2},
            "mechanism": "test", "trusted_server_required": True, "individual_updates_hidden_from_server": False,
            "known_security_issues_resolved": False, "independent_cryptographic_review_complete": False,
        }), encoding="utf-8")
        (protocol / "run_config.json").write_text(json.dumps({"num-server-rounds": 2, "central-dp": True, "differential-privacy-enabled": True}), encoding="utf-8")
        (protocol / "release_policy.json").write_text(json.dumps({
            "adaptive_model_releases": 2, "client_metrics_released": False, "private_validation_released": False,
            "intermediate_checkpoints_released": False, "noise_seeds_released": False, "raw_or_individual_updates_released": False,
        }), encoding="utf-8")
        audit.write_text(json.dumps({"status": "PASS_EXECUTED_DP_SECAGG_STUDY_NOT_CERTIFIED", "rounds": 2,
                                     "semantic_log_audit": {"passed": True}}), encoding="utf-8")
        return protocol, audit

    def test_recomputes_budget_and_keeps_claims_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, audit = self.fixture(root); output = root / "output"
            with patch("paper2.restart.prepare_dp_secagg_privacy_review_packet.RESULTS", root):
                packet = prepare(output, protocol, audit)
        self.assertEqual(packet["conditional_zcdp_budget"]["releases"], 2)
        self.assertFalse(packet["privacy_claim_enabled"])

    def test_rejects_mismatched_reported_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); protocol, audit = self.fixture(root); output = root / "output"
            contract = json.loads((protocol / "privacy_contract.json").read_text(encoding="utf-8"))
            contract["conditional_budget"]["rho"] = 0.2
            (protocol / "privacy_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            with patch("paper2.restart.prepare_dp_secagg_privacy_review_packet.RESULTS", root):
                with self.assertRaisesRegex(ValueError, "does not recompute"):
                    prepare(output, protocol, audit)


if __name__ == "__main__":
    unittest.main()
