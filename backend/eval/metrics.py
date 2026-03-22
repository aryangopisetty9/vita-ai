"""
Eval – Metrics
===============
Standard physiological signal evaluation metrics for rPPG systems.
All functions accept arrays of predicted and ground-truth BPM values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def mean_absolute_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """MAE = mean(|pred − gt|)."""
    return float(np.mean(np.abs(pred - gt)))


def root_mean_square_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """RMSE = sqrt(mean((pred − gt)²))."""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def pearson_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pearson r between predicted and ground-truth HR sequences.

    Returns 0.0 when the correlation is undefined (e.g. constant signal).
    """
    if len(pred) < 2:
        return 0.0
    std_p, std_g = np.std(pred), np.std(gt)
    if std_p < 1e-9 or std_g < 1e-9:
        return 0.0
    r = float(np.corrcoef(pred, gt)[0, 1])
    return r if np.isfinite(r) else 0.0


def std_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Standard deviation of the error distribution."""
    return float(np.std(pred - gt))


def bland_altman_bias(pred: np.ndarray, gt: np.ndarray) -> float:
    """Bland–Altman bias = mean(pred − gt)."""
    return float(np.mean(pred - gt))


def bland_altman_limits(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[float, float, float]:
    """Return (bias, lower_limit, upper_limit) for Bland–Altman analysis.

    Limits of agreement = bias ± 1.96 × SD(diff).
    """
    diff = pred - gt
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def confidence_coverage(
    pred: np.ndarray,
    gt: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.5,
) -> Optional[Dict[str, float]]:
    """Evaluate error metrics only for samples where confidence ≥ threshold.

    Returns a dict with count, MAE, and RMSE for the high-confidence
    subset, or None if no samples pass the threshold.
    """
    mask = confidences >= threshold
    if not np.any(mask):
        return None
    p, g = pred[mask], gt[mask]
    return {
        "count": int(np.sum(mask)),
        "MAE": mean_absolute_error(p, g),
        "RMSE": root_mean_square_error(p, g),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate helper
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    confidences: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute the full metric suite and return a JSON-serialisable dict.

    Parameters
    ----------
    pred : array-like
        Predicted BPM values (one per sample or per time-step).
    gt : array-like
        Ground-truth BPM values, same length as *pred*.
    confidences : array-like or None
        Per-sample confidence scores (0–1).

    Returns
    -------
    dict
        Keys: MAE, RMSE, Pearson_r, STD_error, bias,
              limits_of_agreement, n_samples,
              optionally high_confidence_metrics.
    """
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)

    bias_val, loa_low, loa_high = bland_altman_limits(pred, gt)

    result: Dict[str, Any] = {
        "n_samples": int(len(pred)),
        "MAE": round(mean_absolute_error(pred, gt), 3),
        "RMSE": round(root_mean_square_error(pred, gt), 3),
        "Pearson_r": round(pearson_correlation(pred, gt), 4),
        "STD_error": round(std_error(pred, gt), 3),
        "bias": round(bias_val, 3),
        "limits_of_agreement": {
            "lower": round(loa_low, 3),
            "upper": round(loa_high, 3),
        },
    }

    if confidences is not None:
        confidences = np.asarray(confidences, dtype=float)
        hc = confidence_coverage(pred, gt, confidences, threshold=0.5)
        if hc is not None:
            result["high_confidence_metrics"] = hc

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Classification metrics (for audio / symptom evaluation)
# ═══════════════════════════════════════════════════════════════════════════

def classification_accuracy(
    pred_labels: List[str], gt_labels: List[str]
) -> float:
    """Simple accuracy = # correct / total."""
    if not pred_labels:
        return 0.0
    correct = sum(1 for p, g in zip(pred_labels, gt_labels) if p == g)
    return round(correct / len(pred_labels), 4)


def classification_report(
    pred_labels: List[str],
    gt_labels: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Per-class precision / recall / f1 + macro averages.

    Returns a JSON-friendly dict.
    """
    if labels is None:
        labels = sorted(set(gt_labels) | set(pred_labels))

    report: Dict[str, Any] = {}
    macro_p, macro_r, macro_f = 0.0, 0.0, 0.0

    for lab in labels:
        tp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(pred_labels, gt_labels) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(pred_labels, gt_labels) if p != lab and g == lab)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        report[lab] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
        macro_p += precision
        macro_r += recall
        macro_f += f1

    n = max(len(labels), 1)
    report["macro_avg"] = {
        "precision": round(macro_p / n, 4),
        "recall": round(macro_r / n, 4),
        "f1": round(macro_f / n, 4),
    }
    report["accuracy"] = classification_accuracy(pred_labels, gt_labels)
    report["n_samples"] = len(pred_labels)
    return report
