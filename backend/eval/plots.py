"""
Eval – Plots
==============
Publication-quality visualisations for rPPG evaluation results.
All functions save figures to ``eval/results/`` by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Guard heavy imports so the rest of eval can load without matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not installed – plot functions will be no-ops.")

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

from backend.eval.config import (
    FIGSIZE_BLAND_ALTMAN,
    FIGSIZE_HISTOGRAM,
    FIGSIZE_TIMESERIES,
    PLOT_DPI,
    PLOT_FORMAT,
    RESULTS_DIR,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. HR Timeseries Comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_hr_timeseries(
    pred: Sequence[float],
    gt: Sequence[float],
    title: str = "Heart-Rate Timeseries – Predicted vs Ground Truth",
    out_dir: Optional[str] = None,
    filename: str = "hr_timeseries",
) -> Optional[str]:
    """Plot predicted and ground-truth BPM on the same time axis.

    Returns the path to the saved figure, or None if matplotlib is
    unavailable.
    """
    if not _HAS_MPL:
        return None

    out = Path(out_dir) if out_dir else RESULTS_DIR
    _ensure_dir(out)

    pred_a = np.asarray(pred)
    gt_a = np.asarray(gt)
    n = min(len(pred_a), len(gt_a))
    if n == 0:
        logger.warning("Empty timeseries – skipping plot.")
        return None

    pred_a, gt_a = pred_a[:n], gt_a[:n]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=FIGSIZE_TIMESERIES)
    ax.plot(x, gt_a, label="Ground Truth", linewidth=1.8, color="#2196F3")
    ax.plot(x, pred_a, label="Predicted", linewidth=1.4, linestyle="--", color="#FF5722")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    filepath = out / f"{filename}.{PLOT_FORMAT}"
    fig.tight_layout()
    fig.savefig(str(filepath), dpi=PLOT_DPI)
    plt.close(fig)
    logger.info("Saved timeseries plot → %s", filepath)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Bland–Altman Plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_bland_altman(
    pred: Sequence[float],
    gt: Sequence[float],
    title: str = "Bland–Altman Plot",
    out_dir: Optional[str] = None,
    filename: str = "bland_altman",
) -> Optional[str]:
    """Classic Bland–Altman agreement plot.

    X-axis = mean of predicted and true BPM.
    Y-axis = difference (predicted − true).
    Horizontal lines at bias ± 1.96 SD.
    """
    if not _HAS_MPL:
        return None

    out = Path(out_dir) if out_dir else RESULTS_DIR
    _ensure_dir(out)

    pred_a = np.asarray(pred, dtype=float)
    gt_a = np.asarray(gt, dtype=float)
    n = min(len(pred_a), len(gt_a))
    if n == 0:
        return None

    pred_a, gt_a = pred_a[:n], gt_a[:n]
    mean_vals = (pred_a + gt_a) / 2.0
    diff_vals = pred_a - gt_a
    bias = float(np.mean(diff_vals))
    sd = float(np.std(diff_vals, ddof=1)) if n > 1 else 0.0

    fig, ax = plt.subplots(figsize=FIGSIZE_BLAND_ALTMAN)
    ax.scatter(mean_vals, diff_vals, s=28, alpha=0.65, edgecolors="k", linewidths=0.3)
    ax.axhline(bias, color="red", linestyle="-", linewidth=1.2, label=f"Bias = {bias:.2f}")
    ax.axhline(bias + 1.96 * sd, color="gray", linestyle="--", linewidth=0.9,
               label=f"+1.96 SD = {bias + 1.96 * sd:.2f}")
    ax.axhline(bias - 1.96 * sd, color="gray", linestyle="--", linewidth=0.9,
               label=f"−1.96 SD = {bias - 1.96 * sd:.2f}")
    ax.set_xlabel("Mean of Predicted & Ground Truth (BPM)")
    ax.set_ylabel("Difference (Predicted − Ground Truth) (BPM)")
    ax.set_title(title)
    ax.legend(fontsize=9)

    filepath = out / f"{filename}.{PLOT_FORMAT}"
    fig.tight_layout()
    fig.savefig(str(filepath), dpi=PLOT_DPI)
    plt.close(fig)
    logger.info("Saved Bland–Altman plot → %s", filepath)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Error Distribution Histogram
# ═══════════════════════════════════════════════════════════════════════════

def plot_error_histogram(
    pred: Sequence[float],
    gt: Sequence[float],
    title: str = "Error Distribution (Predicted − Ground Truth)",
    out_dir: Optional[str] = None,
    filename: str = "error_histogram",
) -> Optional[str]:
    """Histogram of per-sample BPM errors."""
    if not _HAS_MPL:
        return None

    out = Path(out_dir) if out_dir else RESULTS_DIR
    _ensure_dir(out)

    pred_a = np.asarray(pred, dtype=float)
    gt_a = np.asarray(gt, dtype=float)
    n = min(len(pred_a), len(gt_a))
    if n == 0:
        return None

    errors = pred_a[:n] - gt_a[:n]
    bins = max(10, n // 5)

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    ax.hist(errors, bins=bins, edgecolor="black", linewidth=0.5, color="#4CAF50", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Error (BPM)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    filepath = out / f"{filename}.{PLOT_FORMAT}"
    fig.tight_layout()
    fig.savefig(str(filepath), dpi=PLOT_DPI)
    plt.close(fig)
    logger.info("Saved error histogram → %s", filepath)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: generate all three plots in one call
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_plots(
    pred: Sequence[float],
    gt: Sequence[float],
    dataset_name: str = "",
    out_dir: Optional[str] = None,
) -> List[str]:
    """Generate timeseries, Bland–Altman, and histogram plots.

    Returns a list of saved file paths.
    """
    suffix = f"_{dataset_name}" if dataset_name else ""
    paths: List[str] = []
    for fn in (
        lambda: plot_hr_timeseries(pred, gt, out_dir=out_dir,
                                   filename=f"hr_timeseries{suffix}"),
        lambda: plot_bland_altman(pred, gt, out_dir=out_dir,
                                  filename=f"bland_altman{suffix}"),
        lambda: plot_error_histogram(pred, gt, out_dir=out_dir,
                                     filename=f"error_histogram{suffix}"),
    ):
        p = fn()
        if p:
            paths.append(p)
    return paths
