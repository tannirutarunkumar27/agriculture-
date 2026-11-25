"""Deterministic local-update primitives for the frozen RR06 protocol.

This module contains no data loading, selection, or outer-test inference.
"""
from __future__ import annotations

import copy
import math
import time

import torch

from paper2.federated import Update, aggregate, clone_state, proximal_loss, state_hash
from paper2.restart.work_scaled_simulation_training import control_average


def make_regressor(inputs, hidden):
    if type(inputs) is not int or type(hidden) is not int or inputs < 1 or hidden < 1:
        raise ValueError("Positive model dimensions required")
    return torch.nn.Sequential(torch.nn.Linear(inputs, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))


def learning_rate(run, round_number):
    if type(round_number) is not int or not 1 <= round_number <= run["rounds"]:
        raise ValueError("Round outside frozen budget")
    initial = run["learning_rate"]
    if run["schedule"] == "constant":
        return initial
    if run["schedule"] == "cosine":
        return initial * 0.5 * (1.0 + math.cos(math.pi * (round_number - 1) / max(1, run["rounds"] - 1)))
    raise ValueError("Unsupported frozen schedule")


def validate_run(run, *, fedprox_mu):
    required = {"id", "fold", "variant", "algorithm", "aggregation_objective", "outer_test_inference",
                "optimizer", "schedule", "hidden", "learning_rate", "weight_decay", "rounds", "local_steps"}
    if set(run) != required or run["outer_test_inference"] is not False:
        raise ValueError("Unexpected RR06 run schema")
    if run["optimizer"] not in {"sgd", "adamw"} or run["schedule"] not in {"constant", "cosine"}:
        raise ValueError("Unsupported optimizer setting")
    if run["algorithm"] not in {"centralized", "fedavg", "fedprox", "fednova_restricted"}:
        raise ValueError("Unsupported algorithm")
    if run["aggregation_objective"] not in {"sample_weighted", "equal_client"}:
        raise ValueError("Unsupported aggregation objective")
    if run["algorithm"] in {"centralized", "fedprox", "fednova_restricted"} and run["aggregation_objective"] != "sample_weighted":
        raise ValueError("Only FedAvg has an equal-client RR06 candidate")
    if run["algorithm"] == "fednova_restricted" and run["optimizer"] != "sgd":
        raise ValueError("Restricted FedNova only supports plain SGD")
    if run["optimizer"] == "adamw" and (run["weight_decay"] <= 0 or run["schedule"] != "cosine"):
        raise ValueError("AdamW candidate differs from frozen protocol")
    if run["optimizer"] == "sgd" and (run["weight_decay"] != 0 or run["schedule"] != "constant"):
        raise ValueError("SGD candidate differs from frozen protocol")
    if type(run["hidden"]) is not int or type(run["rounds"]) is not int or type(run["local_steps"]) is not int:
        raise ValueError("Integer work settings required")
    if run["hidden"] < 1 or run["rounds"] < 1 or run["local_steps"] != 4:
        raise ValueError("Frozen model/work settings differ")
    if not all(isinstance(run[key], (int, float)) and math.isfinite(run[key]) and run[key] > 0 for key in ("learning_rate",)):
        raise ValueError("Invalid learning rate")
    if not isinstance(fedprox_mu, (int, float)) or not math.isfinite(fedprox_mu) or fedprox_mu < 0:
        raise ValueError("Invalid FedProx coefficient")


def _optimizer(model, run, round_number):
    rate = learning_rate(run, round_number)
    if run["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=rate, momentum=0.0, weight_decay=0.0)
    return torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=run["weight_decay"])


def fixed_step_examples(samples, batch_size, steps):
    if any(type(value) is not int or value < 1 for value in (samples, batch_size, steps)):
        raise ValueError("Positive integer work inputs required")
    offset = total = 0
    for _ in range(steps):
        if offset >= samples: offset = 0
        taken = min(batch_size, samples - offset)
        total += taken; offset += taken
    return total


def train_client(model, x, y, client, round_number, schema_hash, seed, run, *, batch_size, fedprox_mu, steps=None, optimizer_state=None, example_budget=None, device="cpu"):
    validate_run(run, fedprox_mu=fedprox_mu)
    if x.dtype != torch.float32 or y.dtype != torch.float32 or x.device.type != "cpu" or y.device.type != "cpu" or x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("Finite CPU float32 regression arrays required")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Invalid client inputs")
    target_steps = run["local_steps"] if steps is None else steps
    if type(target_steps) is not int or target_steps < 1:
        raise ValueError("Positive fixed optimizer work required")
    if example_budget is not None and (type(example_budget) is not int or example_budget < target_steps):
        raise ValueError("Example budget must provide at least one example per step")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    local, base = copy.deepcopy(model).to(device).train(), clone_state(model)
    base_device = {name: value.to(device) for name, value in base.items()}
    x_device, y_device = x.to(device), y.to(device)
    optimizer, generator = _optimizer(local, run, round_number), torch.Generator().manual_seed(seed)
    if optimizer_state is not None:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(run, round_number)
            group["weight_decay"] = run["weight_decay"]
    total, examples, performed = 0.0, 0, 0
    order, offset = torch.randperm(len(y), generator=generator), 0
    started = time.perf_counter()
    while performed < target_steps:
        if example_budget is None:
            if offset >= len(order): order, offset = torch.randperm(len(y), generator=generator), 0
            ids = order[offset:min(offset + batch_size, len(order))]; offset += len(ids)
        else:
            remaining_steps, remaining_examples = target_steps - performed, example_budget - examples
            wanted = remaining_examples // remaining_steps + (1 if remaining_examples % remaining_steps else 0)
            pieces = []
            while wanted:
                if offset >= len(order): order, offset = torch.randperm(len(y), generator=generator), 0
                taken = min(wanted, len(order) - offset); pieces.append(order[offset:offset + taken]); offset += taken; wanted -= taken
            ids = torch.cat(pieces)
        optimizer.zero_grad(set_to_none=True)
        ids_device = ids.to(device)
        prediction = local(x_device[ids_device])
        if prediction.shape != (len(ids), 1):
            raise ValueError("Scalar regression head required")
        supervised = torch.nn.functional.mse_loss(prediction[:, 0], y_device[ids_device])
        loss = supervised + proximal_loss(local, base_device, fedprox_mu if run["algorithm"] == "fedprox" else 0.0)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite local loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in local.parameters()):
            raise FloatingPointError("Non-finite local gradient")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in local.parameters()):
            raise FloatingPointError("Non-finite local parameters")
        total += float(supervised.detach()) * len(ids)
        examples += len(ids); performed += 1
    update = Update(client, round_number, schema_hash, state_hash(base), len(y), performed, clone_state(local), total / examples, time.perf_counter() - started)
    update.examples_processed = examples
    update.optimizer_state = copy.deepcopy(optimizer.state_dict())
    return update


def aggregate_round(base, updates, run, round_number, schema_hash, expected_clients, expected_steps=None):
    validate_run(run, fedprox_mu=0.01)
    if run["algorithm"] == "centralized":
        if len(updates) != 1 or updates[0].client != "central" or set(expected_clients) != {"central"}:
            raise ValueError("Centralized candidate requires exactly one central update")
        update = updates[0]
        required_steps = run["local_steps"] if expected_steps is None else expected_steps
        if update.round != round_number or update.schema_hash != schema_hash or update.base_hash != state_hash(base) or update.samples != expected_clients["central"] or update.steps != required_steps:
            raise ValueError("Centralized update metadata differs")
        return updates[0].state
    if run["aggregation_objective"] == "equal_client":
        if run["algorithm"] != "fedavg":
            raise ValueError("Only FedAvg supports equal-client averaging")
        if len(updates) != len(expected_clients) or {item.client for item in updates} != set(expected_clients):
            raise ValueError("Equal-client roster differs")
        base_hash = state_hash(base)
        for update in updates:
            if update.round != round_number or update.schema_hash != schema_hash or update.base_hash != base_hash or update.samples != expected_clients[update.client] or update.steps != run["local_steps"]:
                raise ValueError("Equal-client update metadata differs")
        return control_average(base, {item.client: item.state for item in updates}, tuple(sorted(expected_clients)), None)
    algorithm = "fednova" if run["algorithm"] == "fednova_restricted" else run["algorithm"]
    return aggregate(base, updates, algorithm, round_number, schema_hash, expected_clients)
