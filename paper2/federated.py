"""Plain-SGD FedAvg, sample-weighted FedProx, and restricted FedNova.

No networks, secure aggregation, differential privacy, or attack defense.
"""
from __future__ import annotations
import copy
import hashlib
from dataclasses import dataclass
import math
import time

import torch
from torch import nn


def make_model(inputs, classes, hidden=32):
    if min(inputs, classes, hidden) < 1:
        raise ValueError("Invalid model dimensions")
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, classes))


def clone_state(model):
    return {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_bytes(state):
    return sum(value.numel() * value.element_size() for value in state.values())


@dataclass
class Update:
    client: str
    round: int
    schema_hash: str
    base_hash: str
    samples: int
    steps: int
    state: dict
    supervised_loss: float = 0.0
    seconds: float = 0.0


def proximal_loss(model, base, mu):
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    return 0.5 * mu * sum((parameter - base[name]).square().sum()
                         for name, parameter in model.named_parameters())


def train_client(model, x, y, client, round_number, schema_hash, seed,
                 epochs=1, batch_size=32, learning_rate=0.05, mu=0.0):
    if len(x) != len(y) or len(x) == 0 or x.ndim != 2 or y.ndim != 1:
        raise ValueError("Invalid client arrays")
    if type(epochs) is not int or type(batch_size) is not int or epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid optimizer configuration")
    if x.dtype != torch.float32 or y.dtype != torch.int64 or x.device.type != "cpu" or y.device.type != "cpu":
        raise ValueError("Client tensors must be CPU float32 inputs and int64 labels")
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite inputs")
    local = copy.deepcopy(model).cpu().train()
    with torch.no_grad():
        classes = local(x[:1]).shape[1]
    if (y < 0).any() or (y >= classes).any():
        raise ValueError("Client label index outside global class schema")
    base = clone_state(model)
    optimizer = torch.optim.SGD(local.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    steps, total_loss, examples = 0, 0.0, 0
    started = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), batch_size):
            ids = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            supervised = nn.functional.cross_entropy(local(x[ids]), y[ids])
            loss = supervised + proximal_loss(local, base, mu)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in local.parameters()):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            total_loss += float(supervised.detach()) * len(ids)
            examples += len(ids)
            steps += 1
    return Update(client, round_number, schema_hash, state_hash(base), len(y), steps,
                  clone_state(local), total_loss/examples, time.perf_counter()-started)


def aggregate(base, updates, algorithm, round_number, schema_hash, expected_clients):
    """Server accepts only model payloads and coordinator-validated sample counts.

    expected_clients maps selected client identifiers to permitted training sizes.
    This integrity validation is NOT cryptographic authentication.
    """
    if algorithm not in {"fedavg", "fedprox", "fednova"} or not updates:
        raise ValueError("Unknown algorithm or no updates")
    ids = [u.client for u in updates]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_clients):
        raise ValueError("Duplicate, missing or unselected client")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in base.values()):
        raise ValueError("Baseline model supports finite floating tensors only")
    base_digest = state_hash(base)
    for update in updates:
        if update.round != round_number or update.schema_hash != schema_hash or update.base_hash != base_digest:
            raise ValueError("Stale or incompatible update")
        if type(update.samples) is not int or update.samples <= 0 or update.samples != expected_clients[update.client]:
            raise ValueError("Invalid sample count")
        if type(update.steps) is not int or update.steps < 1:
            raise ValueError("Invalid actual SGD step count")
        if not isinstance(update.state, dict) or set(update.state) != set(base):
            raise ValueError("State keys differ")
        for name, value in update.state.items():
            reference = base[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError("Update state values must be tensors")
            if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError("Tensor shape/dtype/device mismatch")
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite update")
    weights = [u.samples / sum(v.samples for v in updates) for u in updates]
    effective_steps = sum(p*u.steps for p,u in zip(weights, updates))
    result = {}
    for name, reference in base.items():
        if algorithm == "fednova":
            delta = sum(p * (reference-u.state[name]) / u.steps for p,u in zip(weights,updates))
            result[name] = reference - effective_steps * delta
        else:
            result[name] = sum(p*u.state[name] for p,u in zip(weights,updates))
        if not torch.isfinite(result[name]).all():
            raise FloatingPointError("Aggregation produced non-finite parameters")
    return result
