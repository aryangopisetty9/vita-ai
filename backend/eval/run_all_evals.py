#!/usr/bin/env python
"""
Vita AI – Run All Evaluations
================================
Unified entry-point to run rPPG, audio, and symptom evaluations.

Usage
-----
::

    python eval/run_all_evals.py \\
        --rppg-dataset ubfc --rppg-path datasets/ubfc \\
        --audio-data eval/datasets/audio_labels.csv \\
        --symptom-data eval/datasets/symptom_labels.csv

All flags are optional — only present modules are evaluated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level="INFO", format="%(levelname)s  %(message)s")
logger = logging.getLogger("run_all_evals")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Vita evaluations")
    parser.add_argument("--rppg-dataset", type=str, default=None)
    parser.add_argument("--rppg-path", type=str, default=None)
    parser.add_argument("--audio-data", type=str, default=None)
    parser.add_argument("--symptom-data", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    combined: Dict[str, Any] = {}

    # rPPG evaluation
    if args.rppg_dataset and args.rppg_path:
        logger.info("=== rPPG Evaluation ===")
        from backend.eval.eval_rppg import run_evaluation as run_rppg
        try:
            combined["rppg"] = run_rppg(
                args.rppg_dataset, args.rppg_path, limit=args.limit,
            )
        except Exception as exc:
            logger.error("rPPG eval failed: %s", exc)
            combined["rppg"] = {"error": str(exc)}

    # Audio evaluation
    if args.audio_data:
        logger.info("=== Audio Evaluation ===")
        from backend.eval.eval_audio import run_evaluation as run_audio
        try:
            combined["audio"] = run_audio(args.audio_data, limit=args.limit)
        except Exception as exc:
            logger.error("Audio eval failed: %s", exc)
            combined["audio"] = {"error": str(exc)}

    # Symptom evaluation
    if args.symptom_data:
        logger.info("=== Symptom Evaluation ===")
        from backend.eval.eval_symptom import run_evaluation as run_symptom
        try:
            combined["symptom"] = run_symptom(args.symptom_data, limit=args.limit)
        except Exception as exc:
            logger.error("Symptom eval failed: %s", exc)
            combined["symptom"] = {"error": str(exc)}

    if not combined:
        logger.warning("No evaluations ran. Provide at least one dataset flag.")
        return

    out_dir = _EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "combined_eval.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results → {out_path}")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
