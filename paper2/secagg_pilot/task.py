"""Client-local regression helpers for the Flower SecAgg+ pilot.

Only the training split is read.  This module never loads or evaluates outer
test targets, and the pilot must not be described as independent validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from paper2.restart.regression_training import make_regressor, train_regression_client


def load_client_training_data(data_root: str | Path, fold_id: str, client_id: str):
    root = Path(data_root)
    folder = root / "folds" / fold_id
    partition = json.loads((folder / "partition.json").read_text(encoding="utf-8"))
    if partition.get("kind") != "outer_refit" or client_id not in partition.get("clients", {}):
        raise ValueError("An outer-refit training client is required")
    lookup = {row_id: index for index, row_id in enumerate(partition["train"])}
    local_ids = partition["clients"][client_id]
    if not local_ids or any(row_id not in lookup for row_id in local_ids):
        raise ValueError("Client rows must be nonempty training rows")
    # Access only the two training arrays; x_test/y_test are intentionally never read.
    with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
        x_train = archive["x_train"].copy()
        y_train = archive["y_train"].copy()
    rows = np.asarray([lookup[row_id] for row_id in local_ids], dtype=np.int64)
    return x_train[rows], y_train[rows]


def initial_parameters(features: int, hidden: int = 32):
    if type(features) is not int or features < 1:
        raise ValueError("Positive feature dimension required")
    torch.manual_seed(20261025)
    return make_regressor(features, hidden)


def fit_config(server_round: int) -> dict[str, int]:
    """Send the deterministic local-training round explicitly to every client."""
    if type(server_round) is not int or server_round < 1:
        raise ValueError("Positive server round required")
    return {"server_round": server_round}


def state_to_ndarrays(model):
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_ndarrays(model, values):
    state = model.state_dict()
    if len(values) != len(state):
        raise ValueError("Model parameter count differs")
    converted = {}
    for (name, current), value in zip(state.items(), values):
        value = np.asarray(value)
        if value.shape != tuple(current.shape) or not np.isfinite(value).all():
            raise ValueError("Invalid federated parameter array")
        converted[name] = torch.from_numpy(value.copy()).to(dtype=current.dtype)
    model.load_state_dict(converted)


def local_fit(model, x, y, client_id: str, round_number: int, learning_rate: float = 0.01):
    """One local SGD pass; returns no per-client metric or model artifact."""
    if type(round_number) is not int or round_number < 1:
        raise ValueError("Positive server round required")
    update = train_regression_client(
        model, torch.from_numpy(x), torch.from_numpy(y), client_id, round_number,
        "paper2-secagg-pilot-v1", 1000003 + round_number, learning_rate=learning_rate,
        batch_size=32, mu=0.0,
    )
    model.load_state_dict(update.state)
    return state_to_ndarrays(model)
