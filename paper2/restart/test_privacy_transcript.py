"""Tests for transcript release boundaries; uses no agricultural data."""
import unittest

from paper2.restart.privacy_transcript import PrivacyTranscript


def metadata(round_number=1):
    return {"round": round_number, "public_slots": 2, "parameter_dimensions": 10,
            "weighting": "equal fixed slots", "mode": "SIMULATION_ONLY", "simulation_only": True}


def event():
    return {"sensitivity": 1.0, "noise_std": 2.0, "count": 1}


class PrivacyTranscriptTests(unittest.TestCase):
    def test_release_contains_only_aggregate_declarations(self):
        ledger = PrivacyTranscript(delta=1e-5, public_slots=2)
        ledger.append(event(), metadata())
        released = ledger.release()
        self.assertEqual(released["releases"], 1)
        self.assertFalse(released["secure_aggregation"])
        self.assertFalse(released["application_privacy_certified"])
        self.assertEqual(released["conditional_mathematical_budget"]["releases"], 1)
        self.assertNotIn("noise_seed", str(released))

    def test_forbidden_or_incomplete_metadata_is_rejected(self):
        ledger = PrivacyTranscript(delta=1e-5, public_slots=2)
        for key, value in (("client_updates", []), ("noise_seed", 1), ("per_client_norms", [])):
            bad = metadata(); bad[key] = value
            with self.assertRaises(ValueError): ledger.append(event(), bad)
        bad = metadata(); bad.pop("mode")
        with self.assertRaises(ValueError): ledger.append(event(), bad)

    def test_slot_and_round_changes_are_rejected(self):
        ledger = PrivacyTranscript(delta=1e-5, public_slots=2)
        ledger.append(event(), metadata(1))
        changed = metadata(2); changed["public_slots"] = 3
        with self.assertRaises(ValueError): ledger.append(event(), changed)
        with self.assertRaises(ValueError): ledger.append(event(), metadata(3))

    def test_bad_events_and_delta_are_rejected(self):
        with self.assertRaises(ValueError): PrivacyTranscript(delta=1.0, public_slots=2)
        ledger = PrivacyTranscript(delta=1e-5, public_slots=2)
        with self.assertRaises(ValueError): ledger.append({"noise_std": 1.0}, metadata())


if __name__ == "__main__":
    unittest.main()
