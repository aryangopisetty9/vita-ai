#!/usr/bin/env python
"""
Vita AI – Audio Module Evaluation
===================================
Evaluates breathing-rate estimation against labelled audio samples.

Usage
-----
::

    python eval/eval_audio.py --data eval/datasets/audio_labels.csv

CSV Format
----------
Columns: ``audio_path, gt_breathing_rate, gt_risk``

Results saved to ``eval/results/audio_eval.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.eval.metrics import compute_all_metrics, classification_report
from backend.app.ml.audio.audio_module import analyze_audio

logger = logging.getLogger("eval_audio")


def _load_labels(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "audio_path": row["audio_path"],
                "gt_rate": float(row["gt_breathing_rate"]),
                "gt_risk": row.get("gt_risk", "").strip().lower(),
            })
    return rows


def run_evaluation(csv_path: str, limit: int | None = None) -> Dict[str, Any]:
    labels = _load_labels(csv_path)
    if limit:
        labels = labels[:limit]

    pred_rates, gt_rates = [], []
    pred_risks, gt_risks = [], []
    errors: List[str] = []

    for i, item in enumerate(labels):
        audio_path = item["audio_path"]
        if not os.path.isfile(audio_path):
            errors.append(f"File not found: {audio_path}")
            continue

        logger.info("[%d/%d] %s", i + 1, len(labels), audio_path)
        try:
            result = analyze_audio(audio_path)
        except Exception as exc:
            errors.append(f"{audio_path}: {exc}")
            continue

        pred_rate = result.get("value")
        if pred_rate is not None:
            pred_rates.append(float(pred_rate))
            gt_rates.append(item["gt_rate"])

        pred_risk = result.get("risk", "error")
        if item["gt_risk"]:
            pred_risks.append(pred_risk)
            gt_risks.append(item["gt_risk"])

    report: Dict[str, Any] = {"n_evaluated": len(pred_rates), "errors": errors}

    if pred_rates:
        report["regression_metrics"] = compute_all_metrics(
            np.array(pred_rates), np.array(gt_rates)
        )

    if pred_risks:
        report["classification_metrics"] = classification_report(pred_risks, gt_risks)

    return report


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate audio module")
    parser.add_argument("--data", required=True, help="CSV with labelled audio samples")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    results = run_evaluation(args.data, args.limit)

    out_dir = _EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "audio_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
