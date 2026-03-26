"""
Train a lightweight good-signal vs bad-signal classifier for face rPPG quality.

Input CSV requirements:
- Must contain feature columns listed in FEATURE_NAMES.
- Must contain label column (default: label) with values:
  - good_signal / bad_signal
  - or 1 / 0

Usage:
python -m backend.train.train_signal_quality_classifier \
  --data path/to/quality_dataset.csv \
  --output models_cache/face_quality/signal_quality_classifier.pkl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from backend.app.ml.face.signal_quality_model import FEATURE_NAMES


def _parse_labels(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.strip().str.lower()
    mapped = []
    for v in values:
        if v in {"1", "true", "good", "good_signal"}:
            mapped.append(1)
        elif v in {"0", "false", "bad", "bad_signal"}:
            mapped.append(0)
        else:
            raise ValueError(f"Unsupported label value: {v}")
    return np.asarray(mapped, dtype=np.int64)


def train_model(data_path: str, output_path: str, label_col: str = "label") -> Dict[str, Any]:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    X = df[FEATURE_NAMES].astype(float).to_numpy()
    y = _parse_labels(df[label_col])

    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must contain both good and bad labels")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "label_column": label_col,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)

    metrics_path = out.with_suffix(out.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "model_path": str(out),
        "metrics_path": str(metrics_path),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train face rPPG signal quality classifier")
    parser.add_argument("--data", required=True, help="Path to CSV feature dataset")
    parser.add_argument(
        "--output",
        default="models_cache/face_quality/signal_quality_classifier.pkl",
        help="Output path for trained model pickle",
    )
    parser.add_argument("--label-col", default="label", help="CSV label column name")
    args = parser.parse_args()

    result = train_model(args.data, args.output, args.label_col)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
