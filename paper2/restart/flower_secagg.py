"""Narrow compatibility gate for Flower SecAgg+; no custom cryptography.

This module only constructs a vetted-framework workflow after validating the
research protocol parameters.  It does not turn the current single-process
benchmark into a farm deployment, and it does not make a DP claim.
"""
from __future__ import annotations

from dataclasses import dataclass


FLOWER_VERSION = "1.42.0"


@dataclass(frozen=True)
class SecAggPilotConfig:
    participants: int = 7
    num_shares: int = 7
    reconstruction_threshold: int = 5
    max_weight: float = 1.0
    clipping_range: float = 8.0
    quantization_range: int = 2 ** 22
    modulus_range: int = 2 ** 32


def validate(config: SecAggPilotConfig) -> None:
    if not isinstance(config, SecAggPilotConfig):
        raise TypeError("SecAggPilotConfig required")
    if config.participants < 3:
        raise ValueError("At least three fixed clients are required")
    if not 2 <= config.num_shares <= config.participants:
        raise ValueError("num_shares must be within the fixed roster")
    if not 2 <= config.reconstruction_threshold <= config.num_shares:
        raise ValueError("Invalid reconstruction threshold")
    if config.reconstruction_threshold == config.participants:
        raise ValueError("No dropout tolerance; use an explicit lower threshold")
    if config.max_weight != 1.0:
        raise ValueError("Pilot uses equal fixed-client weights only")
    if config.clipping_range <= 0 or config.quantization_range <= 1:
        raise ValueError("Positive clipping and quantization ranges required")
    if config.modulus_range != 2 ** 32:
        raise ValueError("Use Flower's documented power-of-two uint32 modulus")


def build_workflow(config: SecAggPilotConfig = SecAggPilotConfig()):
    """Return Flower's workflow after version/protocol gates, not a deployment."""
    validate(config)
    try:
        import flwr
        from flwr.server.workflow import SecAggPlusWorkflow
    except ImportError as exc:
        raise RuntimeError("Install pinned flwr==1.42.0 before enabling SecAgg+") from exc
    if flwr.__version__ != FLOWER_VERSION:
        raise RuntimeError(f"Expected flwr=={FLOWER_VERSION}, found {flwr.__version__}")
    return SecAggPlusWorkflow(
        num_shares=config.num_shares,
        reconstruction_threshold=config.reconstruction_threshold,
        max_weight=config.max_weight,
        clipping_range=config.clipping_range,
        quantization_range=config.quantization_range,
        modulus_range=config.modulus_range,
    )


def protocol_record(config: SecAggPilotConfig = SecAggPilotConfig()):
    """Public record for a future frozen pilot; deliberately avoids client data."""
    validate(config)
    return {
        "framework": "Flower SecAgg+",
        "framework_version": FLOWER_VERSION,
        "participants": config.participants,
        "num_shares": config.num_shares,
        "reconstruction_threshold": config.reconstruction_threshold,
        "weighting": "equal_fixed_client",
        "secure_aggregation_requested": True,
        "differential_privacy_enabled": False,
        "production_deployment": False,
        "qualification": "Framework compatibility only; no client/server run or security audit completed.",
    }
