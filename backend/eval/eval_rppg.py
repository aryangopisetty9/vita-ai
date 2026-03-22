#!/usr/bin/env python
"""
Vita AI – rPPG Evaluation Pipeline
====================================
End-to-end evaluation harness that:

1. Loads a benchmark dataset (UBFC / COHFACE / MAHNOB-HCI).
2. Runs ``face_module.analyze_face_video`` on each video.
3. Compares predicted BPM against ground-truth BPM.
4. Computes MAE, RMSE, Pearson r, STD of error, Bland–Altman bias.
5. Saves results as JSON, CSV, and optional plots.

Usage
-----
::

    python eval/eval_rppg.py --dataset ubfc --path datasets/ubfc
    python eval/eval_rppg.py --dataset cohface --path datasets/cohface --save-plots
    python eval/eval_rppg.py --dataset ubfc --path datasets/ubfc --limit 10 --verbose

Outputs are saved to ``eval/results/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
from backend.eval.config import HR_RANGE, RESULTS_DIR, SUPPORTED_DATASETS
from backend.eval.dataset_loader import SampleRecord, detect_and_load
from backend.eval.metrics import compute_all_metrics
from backend.eval.plots import generate_all_plots

from backend.app.ml.face.face_module import analyze_face_video

logger = logging.getLogger("eval_rppg")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


def _extract_prediction(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pull prediction fields from face_module output."""
    hr = result.get("heart_rate") or result.get("value")
    timeseries = result.get("hr_timeseries", [])
    confidence = result.get("heart_rate_confidence") or result.get("confidence", 0.0)
    scan_quality = result.get("scan_quality", 0.0)
    risk = result.get("risk", "error")
    retake = result.get("retake_required", True)

    # Extract BPM values from timeseries dicts if applicable
    ts_bpm: List[float] = []
    if timeseries:
        for entry in timeseries:
            if isinstance(entry, dict):
                val = entry.get("bpm") or entry.get("heart_rate")
                if val is not None and val > 0:
                    ts_bpm.append(float(val))
            elif isinstance(entry, (int, float)) and entry > 0:
                ts_bpm.append(float(entry))

    return {
        "heart_rate": float(hr) if hr else None,
        "hr_timeseries_bpm": ts_bpm,
        "confidence": float(confidence),
        "scan_quality": float(scan_quality),
        "risk": risk,
        "retake_required": retake,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample runner
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_sample(sample: SampleRecord) -> Dict[str, Any]:
    """Run inference on a single sample and compare with ground truth.

    Returns a record dict with prediction, ground truth, and metrics.
    Errors are caught so the pipeline never crashes.
    """
    record: Dict[str, Any] = {
        "subject_id": sample.subject_id,
        "video_path": sample.video_path,
        "gt_mean_bpm": sample.ground_truth_mean_bpm,
        "gt_timeseries_len": len(sample.ground_truth_bpm),
        "status": "pending",
    }

    t0 = time.perf_counter()
    try:
        raw_result = analyze_face_video(sample.video_path)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        record["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        logger.error("  ✗ %s – inference error: %s", sample.subject_id, exc)
        return record
    elapsed = round(time.perf_counter() - t0, 2)
    record["elapsed_sec"] = elapsed

    pred = _extract_prediction(raw_result)
    record.update({
        "pred_bpm": pred["heart_rate"],
        "pred_confidence": pred["confidence"],
        "pred_scan_quality": pred["scan_quality"],
        "pred_risk": pred["risk"],
        "pred_retake": pred["retake_required"],
        "pred_ts_len": len(pred["hr_timeseries_bpm"]),
    })

    # --- Determine comparison mode ---
    if pred["heart_rate"] is None or pred["heart_rate"] <= 0:
        record["status"] = "no_prediction"
        record["error"] = raw_result.get("message", "No heart-rate estimate produced.")
        logger.warning("  ⚠ %s – no prediction (quality %.2f)", sample.subject_id, pred["scan_quality"])
        return record

    # Single-BPM vs mean-GT comparison
    if sample.ground_truth_mean_bpm is not None:
        clamped = _clamp(pred["heart_rate"], *HR_RANGE)
        record["pred_bpm_clamped"] = round(clamped, 1)
        record["error_bpm"] = round(clamped - sample.ground_truth_mean_bpm, 2)
        record["abs_error_bpm"] = round(abs(record["error_bpm"]), 2)

    # Timeseries comparison (if both available)
    if pred["hr_timeseries_bpm"] and sample.ground_truth_bpm:
        n = min(len(pred["hr_timeseries_bpm"]), len(sample.ground_truth_bpm))
        p_ts = np.array(pred["hr_timeseries_bpm"][:n])
        g_ts = np.array(sample.ground_truth_bpm[:n])
        ts_metrics = compute_all_metrics(p_ts, g_ts)
        record["timeseries_metrics"] = ts_metrics

    record["status"] = "ok"
    logger.info(
        "  ✓ %s – pred %.1f BPM (gt %.1f) | err %.1f | conf %.2f | %.1fs",
        sample.subject_id,
        pred["heart_rate"],
        sample.ground_truth_mean_bpm or 0,
        record.get("abs_error_bpm", -1),
        pred["confidence"],
        elapsed,
    )
    return record


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(
    records: List[Dict[str, Any]],
    dataset_name: str,
) -> Dict[str, Any]:
    """Compute aggregate metrics across all successful samples."""
    ok = [r for r in records if r["status"] == "ok" and r.get("pred_bpm") and r.get("gt_mean_bpm")]

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "total_samples": len(records),
        "successful": len(ok),
        "failed": len(records) - len(ok),
    }

    if not ok:
        summary.update({"MAE": None, "RMSE": None, "Pearson_r": None, "STD_error": None, "bias": None})
        return summary

    preds = np.array([r["pred_bpm_clamped"] for r in ok])
    gts = np.array([r["gt_mean_bpm"] for r in ok])
    confs = np.array([r.get("pred_confidence", 0.5) for r in ok])

    metrics = compute_all_metrics(preds, gts, confs)
    summary.update(metrics)

    # Failure analysis
    failures = [r for r in records if r["status"] != "ok"]
    if failures:
        summary["failure_reasons"] = {}
        for f in failures:
            reason = f.get("error", f["status"])
            summary["failure_reasons"][reason] = summary["failure_reasons"].get(reason, 0) + 1

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Saved JSON → %s", path)


def _save_per_sample_csv(records: List[Dict[str, Any]], path: Path) -> None:
    """Flatten per-sample records into a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_keys = [
        "subject_id", "video_path", "status", "gt_mean_bpm",
        "pred_bpm", "pred_bpm_clamped", "error_bpm", "abs_error_bpm",
        "pred_confidence", "pred_scan_quality", "pred_risk",
        "pred_retake", "elapsed_sec", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in flat_keys})
    logger.info("Saved per-sample CSV → %s", path)


def _save_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    logger.info("Saved summary CSV → %s", path)


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"  Evaluation Summary – {summary.get('dataset', '?')}")
    print("=" * 60)
    for key in ("total_samples", "successful", "failed"):
        print(f"  {key:>20s}: {summary.get(key, '—')}")
    print("-" * 60)
    for key in ("MAE", "RMSE", "Pearson_r", "STD_error", "bias"):
        val = summary.get(key)
        print(f"  {key:>20s}: {val if val is not None else '—'}")
    loa = summary.get("limits_of_agreement")
    if loa:
        print(f"  {'LoA (lower)':>20s}: {loa['lower']}")
        print(f"  {'LoA (upper)':>20s}: {loa['upper']}")
    hc = summary.get("high_confidence_metrics")
    if hc:
        print(f"  {'HC count':>20s}: {hc['count']}")
        print(f"  {'HC MAE':>20s}: {hc['MAE']}")
        print(f"  {'HC RMSE':>20s}: {hc['RMSE']}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(
    dataset_type: str,
    dataset_path: str,
    *,
    save_plots: bool = False,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline.

    Returns the summary metrics dict.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)-8s  %(message)s", force=True)

    logger.info("Loading dataset '%s' from %s …", dataset_type, dataset_path)
    samples = detect_and_load(dataset_type, dataset_path, limit=limit)

    if not samples:
        logger.error("No samples loaded – check dataset path and structure.")
        return {"dataset": dataset_type, "error": "no samples loaded"}

    logger.info("Loaded %d samples. Starting evaluation …\n", len(samples))

    records: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples, 1):
        logger.info("[%d/%d] %s", i, len(samples), sample.subject_id)
        rec = evaluate_sample(sample)
        records.append(rec)

    # --- Aggregate ---
    summary = aggregate_results(records, dataset_type)
    _print_summary(summary)

    # --- Save outputs ---
    out_dir = RESULTS_DIR / dataset_type
    _save_json(summary, out_dir / "metrics_summary.json")
    _save_summary_csv(summary, out_dir / "metrics_summary.csv")
    _save_per_sample_csv(records, out_dir / "per_sample_results.csv")
    _save_json(records, out_dir / "per_sample_results.json")

    # --- Plots ---
    if save_plots:
        ok = [r for r in records if r["status"] == "ok" and r.get("pred_bpm_clamped") and r.get("gt_mean_bpm")]
        if ok:
            preds = [r["pred_bpm_clamped"] for r in ok]
            gts = [r["gt_mean_bpm"] for r in ok]
            generate_all_plots(preds, gts, dataset_name=dataset_type, out_dir=str(out_dir))
        else:
            logger.warning("No successful predictions – skipping plots.")

    logger.info("Evaluation complete. Results saved to %s", out_dir)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vita AI – rPPG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python eval/eval_rppg.py --dataset ubfc     --path datasets/ubfc
  python eval/eval_rppg.py --dataset cohface  --path datasets/cohface --save-plots
  python eval/eval_rppg.py --dataset mahnob   --path datasets/mahnob --limit 5 --verbose
""",
    )
    p.add_argument(
        "--dataset", required=True, choices=SUPPORTED_DATASETS,
        help="Dataset type (ubfc | cohface | mahnob).",
    )
    p.add_argument(
        "--path", required=True,
        help="Path to the dataset root directory.",
    )
    p.add_argument(
        "--save-plots", action="store_true", default=False,
        help="Generate and save evaluation plots.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N samples (useful for quick tests).",
    )
    p.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable debug-level logging.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        dataset_type=args.dataset,
        dataset_path=args.path,
        save_plots=args.save_plots,
        limit=args.limit,
        verbose=args.verbose,
    )
