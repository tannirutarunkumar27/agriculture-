"""Training-only zero-support gate; unsupported connections cannot activate."""
import numpy as np


def fit_gate(x):
    x = np.asarray(x)
    if x.ndim != 2 or not len(x) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Finite nonempty float32 training matrix required')
    return np.all(x == 0, axis=0)


def apply_gate(x, mask):
    x, mask = np.asarray(x), np.asarray(mask)
    if x.ndim != 2 or x.dtype != np.float32 or mask.dtype != np.bool_ or mask.shape != (x.shape[1],) or not np.isfinite(x).all():
        raise ValueError('Incompatible support gate')
    result = x.copy()
    result[:, mask] = 0
    return result
