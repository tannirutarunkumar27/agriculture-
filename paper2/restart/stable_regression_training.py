"""Versioned plain-SGD client with fixed local gradient clipping for noise stability."""
import copy
import time

import torch

from paper2.federated import Update, clone_state, proximal_loss, state_hash


def make_regressor(inputs, hidden=32):
    return torch.nn.Sequential(torch.nn.Linear(inputs, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))


def train_regression_client(model, x, y, client, round_number, schema, seed, learning_rate=.001, batch_size=32, mu=0.):
    if x.ndim != 2 or y.ndim != 1 or not len(y) or len(x) != len(y):
        raise ValueError("Invalid regression arrays")
    if x.dtype != torch.float32 or y.dtype != torch.float32 or x.device.type != "cpu" or y.device.type != "cpu":
        raise ValueError("CPU float32 tensors required")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite regression inputs")
    if not 0 < learning_rate < float("inf") or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Invalid SGD settings")
    local = copy.deepcopy(model).train()
    base = clone_state(model)
    optimizer = torch.optim.SGD(local.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    total, steps = 0., 0
    started = time.perf_counter()
    for start in range(0, len(y), batch_size):
        ids = order[start:start+batch_size]
        optimizer.zero_grad(set_to_none=True)
        prediction = local(x[ids])
        if prediction.shape != (len(ids), 1):
            raise ValueError("Scalar regression head required")
        supervised = torch.nn.functional.mse_loss(prediction[:, 0], y[ids])
        loss = supervised+proximal_loss(local, base, mu)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite regression loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in local.parameters()):
            raise FloatingPointError("Non-finite regression gradients")
        torch.nn.utils.clip_grad_norm_(local.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in local.parameters()):
            raise FloatingPointError("Non-finite regression parameters")
        total += float(supervised.detach())*len(ids)
        steps += 1
    return Update(client, round_number, schema, state_hash(base), len(y), steps,
                  clone_state(local), total/len(y), time.perf_counter()-started)
