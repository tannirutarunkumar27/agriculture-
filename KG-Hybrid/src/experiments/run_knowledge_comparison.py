"""Controlled five-seed study of knowledge and forward gating.

All hybrid variants share partitions, booster, pretrained FT weights, leaf
embedding initialization, head initialization, optimizer, and training budget.
Pure cross entropy isolates knowledge/gating from auxiliary-loss confounders.
The compact leaf vocabulary contains every terminal node in the trained
booster, including nodes not visited by the training observations.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.ablation.run_five_seed_ablation import load_raw_data, prepare_seed_partitions
from src.ablation.run_ablation import FusionWithoutGate
from src.experiments.seed_utils import set_all_seeds
from src.experiments.calibration import compute_ece, compute_brier_score, compute_nll
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.ft_transformer.model import FTTransformer
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix

VARIANTS = {
    "Hybrid_NoK": (None, False),
    "Hybrid_FixedK": ("fixed", False),
    "Hybrid_AnchoredK": ("anchored_trainable", False),
    "Hybrid_AnchoredK_Gate": ("anchored_trainable", True),
    "Hybrid_ShuffledK_Gate": ("fixed", True),
}


class PlainFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_total = 192
        self.layer_norm = nn.LayerNorm(192)

    def forward(self, xgb_features, ft_features):
        concatenated = torch.cat((xgb_features, ft_features), dim=1)
        return self.layer_norm(concatenated), torch.ones_like(concatenated)


def compact_leaf_vocabulary(booster):
    trees = booster.trees_to_dataframe()
    count = int(trees["Tree"].max()) + 1
    lookup = np.full((count, int(trees["Node"].max()) + 1), -1, dtype=np.int64)
    offset = 0
    for tree in range(count):
        nodes = np.sort(trees.loc[(trees["Tree"] == tree) & (trees["Feature"] == "Leaf"), "Node"].to_numpy(dtype=int))
        lookup[tree, nodes] = np.arange(offset, offset + len(nodes))
        offset += len(nodes)
    return lookup, offset


def encode_leaves(raw, lookup):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != lookup.shape[0] or not np.all(raw == np.floor(raw)):
        raise ValueError("Leaf dimensions or indices do not match the fitted booster")
    raw = raw.astype(np.int64)
    if (raw < 0).any() or (raw >= lookup.shape[1]).any():
        raise ValueError("Leaf node index is out of the fitted tree vocabulary")
    mapped = lookup[np.arange(lookup.shape[0]), raw]
    if (mapped < 0).any():
        raise ValueError("A nonterminal or unknown node was returned as a leaf")
    return mapped


def probabilities(model, features, leaves=None):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), 128):
            xb = torch.tensor(features[start:start + 128], dtype=torch.float32)
            logits = model(xb) if leaves is None else model(torch.tensor(leaves[start:start + 128]), xb)[0]
            chunks.append(logits.softmax(1).numpy())
    return np.concatenate(chunks)


def fit_neural(model, splits, epochs, seed, anchor_weight, leaves=None):
    X_train, y_train, X_val, y_val = splits
    set_all_seeds(seed)
    model.train()
    embedding = list(model.xgb_embedder.embeddings.parameters()) if leaves is not None else []
    embedding_ids = {id(p) for p in embedding}
    dense = [p for p in model.parameters() if p.requires_grad and id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(dense, lr=1e-3, weight_decay=1e-5)
    sparse_optimizer = torch.optim.SparseAdam(embedding, lr=1e-3) if embedding else None
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(X_train))
        for start in range(0, len(order), 32):
            ids = order[start:start + 32]
            xb = torch.tensor(X_train[ids], dtype=torch.float32)
            yb = torch.tensor(y_train[ids], dtype=torch.long)
            logits = model(xb) if leaves is None else model(torch.tensor(leaves[0][ids]), xb)[0]
            loss = nn.functional.cross_entropy(logits, yb)
            if leaves is not None and hasattr(model.fusion, "knowledge_anchor_loss"):
                loss = loss + anchor_weight * model.fusion.knowledge_anchor_loss()
            optimizer.zero_grad()
            if sparse_optimizer:
                sparse_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dense, 1)
            optimizer.step()
            if sparse_optimizer:
                sparse_optimizer.step()
        val = probabilities(model, X_val, None if leaves is None else leaves[1])
        val_loss = compute_nll(y_val, val)
        history.append({"epoch": epoch + 1, "val_nll": val_loss, "val_accuracy": float(np.mean(val.argmax(1) == y_val))})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if epoch + 1 - best_epoch >= 10:
            break
    model.load_state_dict(best_state)
    return time.perf_counter() - started, best_epoch, history


def save_result(directory, seed, name, y_test, ids, prob, params, elapsed, extra=None):
    np.save(directory / f"{name}_probabilities.npy", prob)
    pd.DataFrame({"sample_id": ids, "y_true": y_test, "y_pred": prob.argmax(1),
                  "confidence": prob.max(1)}).to_csv(directory / f"{name}_predictions.csv", index=False)
    record = {
        "seed": seed, "model": name, "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, prob.argmax(1)),
        "macro_f1": f1_score(y_test, prob.argmax(1), average="macro"),
        "ece": compute_ece(y_test, prob), "brier": compute_brier_score(y_test, prob),
        "nll": compute_nll(y_test, prob), "trainable_parameters": params,
        "training_seconds": elapsed, **(extra or {}),
    }
    (directory / f"{name}_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"seed={seed} {name}: accuracy={record['accuracy']:.4f} ECE={record['ece']:.6f} params={params}", flush=True)
    return record


def run(output_dir, seeds, epochs):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    df = load_raw_data()
    km, crops, features = load_authoritative_knowledge_matrix()
    manifest = {
        "status": "running", "seeds": seeds, "max_epochs": epochs, "variants": VARIANTS,
        "knowledge_anchor_weight": 1.0, "early_stopping": "validation NLL, patience=10",
        "hybrid_objective": "cross entropy + mean-squared knowledge anchor (lambda=1)",
        "embedding_optimizer": "SparseAdam, lr=0.001 (no embedding weight decay)",
        "dense_optimizer": "AdamW, lr=0.001, weight_decay=0.00001",
        "representation": "compact terminal-node vocabulary; every trained booster leaf included",
        "dataset_sha256": hashlib.sha256((ROOT / "data/raw/Crop_recommendation.csv").read_bytes()).hexdigest(),
        "knowledge_sha256": hashlib.sha256(km.tobytes()).hexdigest(),
        "knowledge_source_status": "Manually specified weights; exact institutional derivation unverified",
        "classes": crops, "features": features,
        "interpretation": "Repeated held-out splits, comparative evidence; no field validation or superiority assumed",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records = []
    for seed in seeds:
        destination = output_dir / f"seed_{seed}"
        destination.mkdir()
        parts = prepare_seed_partitions(df, seed)
        X_train, y_train, X_val, y_val, X_test, y_test, encoder, scaler = parts
        # Reproduce the exact sample IDs with the same partition function.
        from sklearn.model_selection import train_test_split
        labels = encoder.transform(df["label"])
        train_ids, temp_ids = train_test_split(np.arange(len(df)), test_size=.3, random_state=seed, stratify=labels)
        val_ids, test_ids = train_test_split(temp_ids, test_size=.5, random_state=seed, stratify=labels[temp_ids])
        np.savez(destination / "partitions.npz", train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        set_all_seeds(seed)
        started = time.perf_counter()
        booster_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=.1,
            subsample=.8, colsample_bytree=.8, tree_method="hist", random_state=seed, n_jobs=2)
        booster_model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        booster_model.save_model(destination / "xgboost.json")
        records.append(save_result(destination, seed, "XGBoost", y_test, test_ids,
            booster_model.predict_proba(X_test), 0, elapsed,
            {"parameter_unit": "non-neural; see tree count", "tree_count": len(booster_model.get_booster().get_dump())}))
        lookup, vocabulary_size = compact_leaf_vocabulary(booster_model.get_booster())
        np.save(destination / "leaf_lookup.npy", lookup)
        leaves = tuple(encode_leaves(booster_model.apply(X), lookup) for X in (X_train, X_val, X_test))
        set_all_seeds(seed)
        ft = FTTransformer()
        elapsed, best_epoch, history = fit_neural(ft, parts[:4], epochs, seed, 0)
        torch.save(ft.state_dict(), destination / "FTTransformer.pth")
        (destination / "FTTransformer_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        records.append(save_result(destination, seed, "FTTransformer", y_test, test_ids,
            probabilities(ft, X_test), sum(p.numel() for p in ft.parameters() if p.requires_grad), elapsed,
            {"best_epoch": best_epoch}))
        for name, (policy, gated) in VARIANTS.items():
            set_all_seeds(seed)
            embedder = XGBLeafEmbedder(vocabulary_size, lookup.shape[0], 128, sparse=True)
            backbone = copy.deepcopy(ft)
            matrix = km.copy()
            if name == "Hybrid_ShuffledK_Gate":
                matrix = matrix[np.random.default_rng(seed).permutation(len(km))]
            if policy is None:
                fusion = PlainFusion()
            elif gated:
                fusion = KnowledgeGuidedFusion(knowledge_matrix=matrix, knowledge_policy=policy)
            else:
                fusion = FusionWithoutGate(matrix, knowledge_policy=policy)
                # Remove unused gate parameters so reported counts describe active models.
                del fusion.gate
            set_all_seeds(seed + 100)
            model = KGHybrid(embedder, backbone, fusion)
            elapsed, best_epoch, history = fit_neural(model, parts[:4], epochs, seed, 1, leaves[:2])
            torch.save(model.state_dict(), destination / f"{name}.pth")
            (destination / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            drift = fusion.knowledge_drift_metrics() if hasattr(fusion, "knowledge_drift_metrics") else {}
            records.append(save_result(destination, seed, name, y_test, test_ids,
                probabilities(model, X_test, leaves[2]), sum(p.numel() for p in model.parameters() if p.requires_grad), elapsed,
                {"best_epoch": best_epoch, "leaf_vocabulary_size": vocabulary_size, **drift}))
            if policy:
                pd.DataFrame(fusion.knowledge_matrix.detach().numpy(), index=crops, columns=features).to_csv(destination / f"{name}_learned_matrix.csv")
            pd.DataFrame(records).to_csv(output_dir / "per_seed_metrics.csv", index=False)
    frame = pd.DataFrame(records)
    frame.groupby("model")[["accuracy", "macro_f1", "ece", "brier", "nll", "trainable_parameters"]].agg(["mean", "std"]).to_csv(output_dir / "summary.csv")
    paired = []
    full = frame[frame.model == "Hybrid_AnchoredK_Gate"].set_index("seed")
    for name in frame.model.unique():
        if name == "Hybrid_AnchoredK_Gate":
            continue
        comparator = frame[frame.model == name].set_index("seed")
        for metric in ("accuracy", "macro_f1", "ece", "brier", "nll"):
            difference = full[metric] - comparator.loc[full.index, metric]
            paired.append({"comparator": name, "metric": metric, "n_seeds": len(seeds),
                "mean_full_minus_comparator": float(difference.mean()),
                "paired_t_pvalue": float(stats.ttest_1samp(difference, 0).pvalue) if len(seeds) > 1 and difference.std() > 0 else None,
                "wilcoxon_pvalue": float(stats.wilcoxon(difference).pvalue) if len(seeds) > 1 and (difference != 0).any() else None})
    pd.DataFrame(paired).to_csv(output_dir / "paired_comparisons.csv", index=False)
    from src.experiments.validate_knowledge_comparison import add_adjusted_comparisons
    add_adjusted_comparisons(output_dir)
    manifest["status"] = "completed"
    manifest["completed_results"] = len(records)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/knowledge_comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 3407, 9999])
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Epochs must be positive and seeds unique")
    run(args.output_dir.resolve(), args.seeds, args.epochs)
