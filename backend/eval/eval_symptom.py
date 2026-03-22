#!/usr/bin/env python
"""
Vita AI – Symptom Module Evaluation
======================================
Evaluates symptom risk classification against labelled text samples.

Usage
-----
::

    python eval/eval_symptom.py --data eval/datasets/symptom_labels.csv

CSV Format
----------
Columns: ``text, gt_risk`` (and optionally ``gt_symptoms``)

Results saved to ``eval/results/symptom_eval.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from backend.eval.metrics import classification_report
from backend.app.ml.nlp.symptom_module import analyze_symptoms

logger = logging.getLogger("eval_symptom")


def _load_labels(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "text": row["text"],
                "gt_risk": row["gt_risk"].strip().lower(),
                "gt_symptoms": row.get("gt_symptoms", ""),
            })
    return rows


def run_evaluation(csv_path: str, limit: int | None = None) -> Dict[str, Any]:
    labels = _load_labels(csv_path)
    if limit:
        labels = labels[:limit]

    pred_risks, gt_risks = [], []
    examples: List[Dict[str, Any]] = []

    for i, item in enumerate(labels):
        logger.info("[%d/%d] %s", i + 1, len(labels), item["text"][:60])
        try:
            result = analyze_symptoms(item["text"])
        except Exception as exc:
            logger.warning("Error analysing: %s — %s", item["text"][:40], exc)
            continue

        pred_risk = result.get("risk", "error")
        pred_risks.append(pred_risk)
        gt_risks.append(item["gt_risk"])

        examples.append({
            "text": item["text"][:100],
            "gt_risk": item["gt_risk"],
            "pred_risk": pred_risk,
            "confidence": result.get("confidence"),
            "detected_symptoms": result.get("detected_symptoms", []),
        })

    report: Dict[str, Any] = {
        "n_evaluated": len(pred_risks),
    }
    if pred_risks:
        report["classification_metrics"] = classification_report(
            pred_risks, gt_risks, labels=["low", "moderate", "high"],
        )
    report["sample_results"] = examples[:20]  # first 20 for inspection
    return report


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate symptom module")
    parser.add_argument("--data", required=True, help="CSV with labelled symptom texts")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    results = run_evaluation(args.data, args.limit)

    out_dir = _EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "symptom_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
