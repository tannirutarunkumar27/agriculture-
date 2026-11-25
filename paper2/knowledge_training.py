"""Local multi-input training adapter; reuses existing aggregation unchanged.

No real-data KG study is launched until agronomic approvals and a new freeze.
"""
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from paper2.data import file_hash, object_hash
from paper2.federated import Update, clone_state, proximal_loss, state_hash
from paper2.knowledge import KnowledgeGuidedMLP


def guidance_schema(model, preprocessing_identity):
    if not isinstance(model, KnowledgeGuidedMLP) or not isinstance(preprocessing_identity, str) or not preprocessing_identity:
        raise ValueError("Explicit preprocessing identity required")
    return object_hash({"model": model.manifest(), "preprocessing": preprocessing_identity,
                        "state_layout": {k: [str(v.dtype), list(v.shape)] for k, v in model.state_dict().items()}})


def train_knowledge_client(model, x, raw, observed, contexts, y, client, round_number,
                           preprocessing_identity, seed, epochs=1, batch_size=32,
                           learning_rate=0.05, mu=0.0):
    if not isinstance(model, KnowledgeGuidedMLP):
        raise ValueError("KnowledgeGuidedMLP required")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y) or len(raw) != len(y):
        raise ValueError("Invalid/alignment-mismatched local arrays")
    if x.dtype != torch.float32 or raw.dtype != torch.float32 or y.dtype != torch.int64 or any(v.device.type != "cpu" for v in (x, raw, observed, y)):
        raise ValueError("CPU float32 model/raw inputs and int64 labels required")
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid epoch/batch/seed settings")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")
    if type(mu) not in (int, float) or not math.isfinite(mu) or mu < 0:
        raise ValueError("Invalid proximal coefficient")
    if not torch.isfinite(x).all() or (y < 0).any() or (y >= len(model.bank.classes)).any():
        raise ValueError("Invalid model inputs/labels")
    model.bank.suitability(raw, observed, contexts)  # validate every row before updates
    schema = guidance_schema(model, preprocessing_identity)
    local = copy.deepcopy(model).cpu().train()
    base = clone_state(model)
    optimizer = torch.optim.SGD(local.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    steps, total, examples = 0, 0.0, 0
    started = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), batch_size):
            ids = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = local(x[ids], raw[ids], observed[ids], [contexts[i] for i in ids.tolist()])
            supervised = nn.functional.cross_entropy(logits, y[ids])
            loss = supervised + proximal_loss(local, base, mu)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite guided loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in local.parameters()):
                raise FloatingPointError("Non-finite guided gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in local.parameters()):
                raise FloatingPointError("Non-finite guided parameter")
            steps += 1
            examples += len(ids)
            total += float(supervised.detach())*len(ids)
    return Update(client, round_number, schema, state_hash(base), len(y), steps,
                  clone_state(local), total/examples, time.perf_counter()-started)


def save_guided_weights(model, output, preprocessing_identity):
    """Selected weights only, NOT a resumable round/RNG/optimizer checkpoint."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "weights.pt"
    torch.save(clone_state(model), path)
    manifest = {"schema_hash": guidance_schema(model, preprocessing_identity),
                "model": model.manifest(), "weights_sha256": file_hash(path),
                "artifact_type": "selected_weights_not_training_resume"}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


def load_guided_weights(model, folder, preprocessing_identity):
    """Load locally produced artifacts only; metadata/hash are not signatures."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_hash"] != guidance_schema(model, preprocessing_identity) or manifest["model"] != model.manifest():
        raise ValueError("Knowledge/classes/alpha/preprocessing/model identity changed")
    if file_hash(folder / "weights.pt") != manifest["weights_sha256"]:
        raise ValueError("Saved guided weights changed")
    state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype
                                         or not torch.isfinite(state[k]).all() for k in expected):
        raise ValueError("Malformed guided weight state")
    model.load_state_dict(state, strict=True)
