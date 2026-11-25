"""Strict class-ordered probability evaluation shared by non-neural baselines."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.metrics import precision_recall_fscore_support


def probability_metrics(y, probabilities, classes, detailed=True):
    y = np.asarray(y)
    probability = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or probability.shape != (len(y), len(classes)):
        raise ValueError("Probability shape does not match labels/class schema")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Invalid probabilities")
    totals = probability.sum(axis=1,keepdims=True)
    if not np.allclose(totals, 1, atol=1e-5, rtol=0):
        raise ValueError("Probabilities must sum to one")
    # Float32 library outputs can differ from unity by ~1e-7 after float64
    # conversion. Normalize only after strict validation, not as calibration.
    probability = probability / totals
    if y.dtype.kind not in "iu" or (y < 0).any() or (y >= len(classes)).any():
        raise ValueError("Invalid label indices")
    prediction = probability.argmax(axis=1)
    indices = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=indices,
                                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y, prediction, labels=indices,
                                               average="weighted", zero_division=0)[2]
    result = {"samples":len(y), "accuracy":float(accuracy_score(y, prediction)),
              "macro_precision":float(precision), "macro_recall":float(recall),
              "macro_f1":float(f1), "weighted_f1":float(weighted),
              "nll":float(log_loss(y, probability, labels=indices))}
    if detailed:
        result["per_class"] = classification_report(y, prediction, labels=indices,
                                                    target_names=classes, output_dict=True, zero_division=0)
        result["confusion_matrix"] = confusion_matrix(y, prediction, labels=indices).tolist()
    return result
