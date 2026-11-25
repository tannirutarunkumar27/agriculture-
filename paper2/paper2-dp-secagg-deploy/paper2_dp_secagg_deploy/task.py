"""Client-local training only; no test labels, metrics, or updates are written."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _seed(client_id: str, round_number: int) -> int:
    digest = hashlib.sha256(f"paper2-dp-secagg:{client_id}:{round_number}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def make_regressor(features: int) -> torch.nn.Module:
    if type(features) is not int or features < 1:
        raise ValueError("Positive feature count required")
    torch.manual_seed(20261025)
    return torch.nn.Sequential(torch.nn.Linear(features, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))


def load_training_data(data_root: str, fold_id: str, client_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read only a named client's rows from outer-refit training arrays."""
    folder = Path(data_root) / "folds" / fold_id
    partition = json.loads((folder / "partition.json").read_text(encoding="utf-8"))
    if partition.get("kind") != "outer_refit":
        raise ValueError("Client data must be an outer-refit training partition")
    client_rows = partition.get("clients", {}).get(client_id)
    train_rows = partition.get("train", [])
    if not client_rows or not train_rows:
        raise ValueError("Client and training rows must be nonempty")
    index = {row_id: offset for offset, row_id in enumerate(train_rows)}
    if any(row_id not in index for row_id in client_rows):
        raise ValueError("Client partition contains a non-training row")
    with np.load(folder / "arrays.npz", allow_pickle=False) as values:
        x_train = values["x_train"].copy()
        y_train = values["y_train"].copy()
    rows = np.asarray([index[row_id] for row_id in client_rows], dtype=np.int64)
    return x_train[rows], y_train[rows]


def state_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    return [item.detach().cpu().numpy().copy() for item in model.state_dict().values()]


def set_ndarrays(model: torch.nn.Module, values: list[np.ndarray]) -> None:
    state = model.state_dict()
    if len(values) != len(state):
        raise ValueError("Federated parameter count differs")
    loaded = {}
    for (name, original), value in zip(state.items(), values):
        value = np.asarray(value)
        if value.shape != tuple(original.shape) or not np.isfinite(value).all():
            raise ValueError("Invalid federated parameter array")
        loaded[name] = torch.from_numpy(value.copy()).to(dtype=original.dtype)
    model.load_state_dict(loaded)


def train_one_round(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, client_id: str,
                    round_number: int, learning_rate: float) -> list[np.ndarray]:
    if x.dtype != np.float32 or y.dtype != np.float32 or x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("CPU float32 feature/target arrays with aligned rows are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not 0 < learning_rate < float("inf"):
        raise ValueError("Non-finite data or invalid learning rate")
    model.train(); optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(_seed(client_id, round_number)))
    features, targets = torch.from_numpy(x), torch.from_numpy(y)
    for start in range(0, len(y), 32):
        ids = order[start:start + 32]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features[ids])
        loss = torch.nn.functional.mse_loss(prediction[:, 0], targets[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite local loss")
        loss.backward(); optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Non-finite local parameters")
    return state_to_ndarrays(model)
