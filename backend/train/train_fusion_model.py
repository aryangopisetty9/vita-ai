"""
Vita AI – Fusion Model Training Script
========================================
Trains a score-fusion model from labelled scan data.

Usage
-----
::

    python -m train.train_fusion_model \\
        --data train_data.csv \\
        --output models/fusion_trained.json \\
        --backend xgboost

Input CSV Format
----------------
Columns: heart_score, breathing_score, symptom_score,
         conf_heart, conf_breathing, conf_symptom, label

``label`` is the ground-truth Vita Health Score (0-100).
Rows where a module was not run should have its score set to 50 and
its confidence set to 0.

Output
------
Saves the trained model to ``--output`` (default ``models/fusion_trained.json``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level="INFO", format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "heart_score",
    "breathing_score",
    "symptom_score",
    "conf_heart",
    "conf_breathing",
    "conf_symptom",
]
LABEL_COL = "label"


def _load_csv(path: str) -> tuple:
    """Load CSV into X, y numpy arrays."""
    import csv

    rows: list[list[float]] = []
    labels: list[float] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                feats = [float(row[c]) for c in FEATURE_COLS]
                label = float(row[LABEL_COL])
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping row: %s", exc)
                continue
            rows.append(feats)
            labels.append(label)

    if not rows:
        raise ValueError(f"No valid rows in {path}")

    return np.array(rows), np.array(labels)


def train_xgboost(X: np.ndarray, y: np.ndarray, output: str) -> dict:
    """Train an XGBoost regressor and save as JSON."""
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    logger.info("5-fold CV MAE: %.2f ± %.2f", -cv_scores.mean(), cv_scores.std())

    model.fit(X, y)
    model.save_model(output)
    logger.info("Saved XGBoost model → %s", output)

    return {
        "backend": "xgboost",
        "cv_mae": round(float(-cv_scores.mean()), 2),
        "cv_std": round(float(cv_scores.std()), 2),
        "n_samples": len(y),
    }


def train_sklearn(X: np.ndarray, y: np.ndarray, output: str) -> dict:
    """Train a GradientBoosting regressor (sklearn) and save as pkl."""
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    logger.info("5-fold CV MAE: %.2f ± %.2f", -cv_scores.mean(), cv_scores.std())

    model.fit(X, y)
    joblib.dump(model, output)
    logger.info("Saved sklearn model → %s", output)

    return {
        "backend": "sklearn",
        "cv_mae": round(float(-cv_scores.mean()), 2),
        "cv_std": round(float(cv_scores.std()), 2),
        "n_samples": len(y),
    }


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train Vita fusion model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    default_out = os.getenv("VITA_FUSION_MODEL_PATH", "backend/data/models_cache/fusion/fusion_trained.json")
    parser.add_argument(
        "--output",
        default=default_out,
        help="Output model path (.json for XGBoost, .pkl for sklearn)",
    )
    parser.add_argument(
        "--backend",
        choices=["xgboost", "sklearn"],
        default="xgboost",
        help="ML backend to use",
    )
    args = parser.parse_args()

    # Support JSON or CSV training data (auto-detected by extension)
    if args.data.lower().endswith(".json"):
        # Lazy import of JSON loader
        def _load_json(path: str):
            import json as _json
            rows = []
            labels = []
            data = _json.load(open(path, "r", encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("rows", "data", "records"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise ValueError("JSON training data must be a list of records")
            for row in data:
                try:
                    feats = [float(row[c]) for c in FEATURE_COLS]
                    label = float(row[LABEL_COL])
                except (KeyError, ValueError) as exc:
                    logger.warning("Skipping JSON row: %s", exc)
                    continue
                rows.append(feats)
                labels.append(label)
            if not rows:
                raise ValueError(f"No valid rows in {path}")
            return np.array(rows), np.array(labels)

        X, y = _load_json(args.data)
    else:
        X, y = _load_csv(args.data)
    logger.info("Loaded %d samples with %d features", *X.shape)

    if args.backend == "xgboost":
        metrics = train_xgboost(X, y, args.output)
    else:
        if not args.output.endswith(".pkl"):
            args.output = args.output.rsplit(".", 1)[0] + ".pkl"
        metrics = train_sklearn(X, y, args.output)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
