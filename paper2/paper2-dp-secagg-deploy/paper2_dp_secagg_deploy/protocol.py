"""Frozen framework-level SecAgg+ parameters for the deployment FAB."""
from __future__ import annotations

from dataclasses import dataclass


FLOWER_VERSION = "1.42.0"


@dataclass(frozen=True)
class SecAggProtocol:
    participants: int = 7
    num_shares: int = 7
    reconstruction_threshold: int = 5
    max_weight: float = 1.0
    clipping_range: float = 8.0
    quantization_range: int = 2 ** 22
    modulus_range: int = 2 ** 32


def workflow(config: SecAggProtocol = SecAggProtocol()):
    """Return Flower's vetted workflow after fail-closed version checks."""
    if not (config.participants == config.num_shares == 7):
        raise ValueError("The frozen deployment requires exactly seven shares")
    if config.reconstruction_threshold != 5 or config.max_weight != 1.0:
        raise ValueError("The frozen threshold or equal-weight policy changed")
    if config.clipping_range <= 0 or config.quantization_range != 2 ** 22 or config.modulus_range != 2 ** 32:
        raise ValueError("The frozen SecAgg+ numeric configuration changed")
    import flwr
    from flwr.server.workflow import SecAggPlusWorkflow
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
