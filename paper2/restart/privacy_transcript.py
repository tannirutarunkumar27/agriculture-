"""Release-policy guard for noisy-aggregate simulation transcripts.

This is not secure aggregation or a production differential-privacy system.
It makes the simulation boundary executable: only declared aggregate events and
their public metadata can enter a released transcript; raw client updates,
per-client diagnostics, and noise seeds are rejected.
"""
from __future__ import annotations

from copy import deepcopy

from paper2.restart.privacy_accounting import gaussian_budget


FORBIDDEN_TERMS = frozenset({"client", "update", "gradient", "raw", "seed", "norm", "checkpoint", "parameter"})
ALLOWED_METADATA = frozenset({"round", "public_slots", "parameter_dimensions", "weighting", "mode", "simulation_only"})


def _safe_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) - ALLOWED_METADATA:
        raise ValueError("Only explicit public aggregate metadata may be released")
    if set(metadata) != ALLOWED_METADATA:
        raise ValueError("Complete public metadata required")
    if type(metadata["round"]) is not int or metadata["round"] < 1:
        raise ValueError("Positive round required")
    if type(metadata["public_slots"]) is not int or metadata["public_slots"] < 1:
        raise ValueError("Public slot count required")
    if type(metadata["parameter_dimensions"]) is not int or metadata["parameter_dimensions"] < 1:
        raise ValueError("Parameter dimension count required")
    if metadata["weighting"] != "equal fixed slots" or metadata["mode"] != "SIMULATION_ONLY" or metadata["simulation_only"] is not True:
        raise ValueError("Unsupported privacy simulation metadata")
    return deepcopy(metadata)


class PrivacyTranscript:
    """Append-only declaration ledger for one fixed simulation release stream."""

    def __init__(self, *, delta, public_slots):
        if type(public_slots) is not int or public_slots < 1:
            raise ValueError("Fixed positive public slot count required")
        self._delta = float(delta)
        # Validate delta before a first event is accepted.
        gaussian_budget([], self._delta)
        self._public_slots = public_slots
        self._records = []

    def append(self, event, metadata):
        metadata = _safe_metadata(metadata)
        if metadata["public_slots"] != self._public_slots:
            raise ValueError("Public slot count changed within transcript")
        if self._records and metadata["round"] != self._records[-1]["metadata"]["round"] + 1:
            raise ValueError("Released rounds must be contiguous")
        # The accountant strictly validates the event schema and values.
        gaussian_budget([event], self._delta)
        self._records.append({"event": deepcopy(event), "metadata": metadata})

    def release(self):
        events = [record["event"] for record in self._records]
        return {
            "mode": "SIMULATION_ONLY",
            "secure_aggregation": False,
            "application_privacy_certified": False,
            "public_slots": self._public_slots,
            "releases": len(events),
            "conditional_mathematical_budget": gaussian_budget(events, self._delta),
            "records": deepcopy(self._records),
            "qualification": (
                "Only a declaration ledger for noisy aggregate simulation. It does not hide "
                "raw updates from a trusted server and is not a production privacy certificate."
            ),
        }
