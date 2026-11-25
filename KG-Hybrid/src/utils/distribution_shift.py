"""
File: src/utils/distribution_shift.py

Purpose:
Quantify out-of-distribution (OOD) covariate shift between primary internal training 
data and external validation datasets using standard statistical metrics:
1. 1D Wasserstein Distance (Earth Mover's Distance)
2. Population Stability Index (PSI)
3. Jensen-Shannon Divergence (JSD)
4. Multivariate Mahalanobis Distance

Citations:
- Wasserstein, L. N. (1969). Markov processes over denumerable sets of states.
- Yurdakul, B. (2018). Statistical properties of Population Stability Index.
- Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import os
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns


def _save_figure_portably(fig, destination: str, **kwargs) -> None:
    """Publish a complete plot when Windows has the prior image open.

    Matplotlib can fail when it writes directly over a file currently previewed
    by Explorer or another renderer.  Render beside the destination and replace
    atomically when possible; retain a normal overwrite fallback for restricted
    Windows/sandbox filesystems.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        fig.savefig(temporary_path, **kwargs)
        for attempt in range(5):
            try:
                os.replace(temporary_path, path)
                break
            except OSError:
                if attempt == 4:
                    shutil.copy2(temporary_path, path)
                    break
                time.sleep(0.2)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def calculate_psi(
    reference: np.ndarray,
    target: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and target distributions.
    
    Interpretation:
    - PSI < 0.10: Minimal shift (In-Distribution)
    - 0.10 <= PSI < 0.25: Moderate shift (Noticeable drift)
    - PSI >= 0.25: Severe shift (Significant OOD drift)
    """
    reference = np.asarray(reference).ravel()
    target = np.asarray(target).ravel()

    # Determine quantile bins based on reference data
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] -= epsilon
    bin_edges[-1] += epsilon
    # Ensure strictly increasing bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        bin_edges = np.linspace(reference.min() - epsilon, reference.max() + epsilon, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (len(reference) + len(bin_edges) * epsilon)
    target_pct = (target_counts + epsilon) / (len(target) + len(bin_edges) * epsilon)

    psi_val = np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct))
    return float(psi_val)


def calculate_jsd(
    reference: np.ndarray,
    target: np.ndarray,
    bins: int = 20,
    epsilon: float = 1e-4
) -> float:
    """
    Calculate Jensen-Shannon Divergence (JSD) bounded in [0, 1].
    """
    reference = np.asarray(reference).ravel()
    target = np.asarray(target).ravel()

    min_val = min(reference.min(), target.min()) - epsilon
    max_val = max(reference.max(), target.max()) + epsilon
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)

    p = (ref_counts + epsilon) / (len(reference) + bins * epsilon)
    q = (target_counts + epsilon) / (len(target) + bins * epsilon)

    # jensenshannon returns square root of JS divergence, square it for JSD
    js_distance = jensenshannon(p, q)
    return float(js_distance ** 2)


def calculate_mahalanobis_shift(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame
) -> float:
    """
    Calculate mean Mahalanobis Distance of target samples relative to reference distribution.
    """
    ref_mat = reference_df.values
    target_mat = target_df.values

    mean_ref = np.mean(ref_mat, axis=0)
    cov_ref = np.cov(ref_mat, rowvar=False)

    # Regularize covariance matrix to ensure invertibility
    cov_inv = np.linalg.pinv(cov_ref + 1e-6 * np.eye(cov_ref.shape[0]))

    diff = target_mat - mean_ref
    # Vectorized Mahalanobis calculation
    mahalanobis_sq = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
    return float(np.mean(np.sqrt(np.maximum(0.0, mahalanobis_sq))))


def compute_distribution_shift_metrics(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Compute comprehensive feature-wise distribution shift metrics.
    """
    records = []
    for col in feature_cols:
        ref_vals = reference_df[col].values
        target_vals = target_df[col].values

        w_dist = float(wasserstein_distance(ref_vals, target_vals))
        psi_val = calculate_psi(ref_vals, target_vals)
        jsd_val = calculate_jsd(ref_vals, target_vals)

        shift_status = "Minimal"
        if psi_val >= 0.25:
            shift_status = "Severe (OOD)"
        elif psi_val >= 0.10:
            shift_status = "Moderate"

        records.append({
            "feature": col,
            "wasserstein_distance": round(w_dist, 4),
            "psi": round(psi_val, 4),
            "jsd": round(jsd_val, 4),
            "shift_status": shift_status
        })

    return pd.DataFrame(records)


def plot_distribution_shift(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list,
    save_path: str,
    target_dataset_name: str = "External Dataset"
) -> None:
    """
    Generate publication-quality density comparison plot for feature distributions.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        sns.kdeplot(reference_df[col], ax=ax, label="Internal Train", color="#1f77b4", linewidth=2, fill=True, alpha=0.25)
        sns.kdeplot(target_df[col], ax=ax, label=target_dataset_name, color="#ff7f0e", linewidth=2, fill=True, alpha=0.25)
        ax.set_title(f"Feature: {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    # Hide extra subplots if feature_cols < 9
    for idx in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Covariate Shift Analysis: Internal Train vs {target_dataset_name}", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    _save_figure_portably(fig, save_path, dpi=300, bbox_inches='tight')
    plt.close()
