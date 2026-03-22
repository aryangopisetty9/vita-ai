import os
import tempfile
import numpy as np
from pathlib import Path

from backend.app.ml.fusion import fusion_model


def test_train_and_load_sklearn_model(tmp_path):
    # Create a tiny synthetic dataset and train a sklearn model via the training script
    X = np.array([
        [80, 85, 90, 0.9, 0.9, 0.9],
        [60, 70, 50, 0.8, 0.8, 0.6],
        [40, 50, 20, 0.6, 0.5, 0.3],
        [90, 95, 95, 0.95, 0.95, 0.95],
    ])
    y = np.array([85, 65, 35, 95])

    out_path = tmp_path / "fusion_test_model.pkl"
    from sklearn.ensemble import GradientBoostingRegressor
    import joblib
    model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, str(out_path))

    # Point env var to this model and load via public loader
    os.environ["VITA_FUSION_MODEL_PATH"] = str(out_path)
    ok = fusion_model.load_fusion_model()
    assert ok is True

    # Run a prediction using the legacy signature
    pred = fusion_model.predict_score(80, 80, 80, [0.9, 0.9, 0.9])
    assert pred is not None
    assert 0 <= pred <= 100

    # And using dict-style features
    feat = {
        "heart_score": 70,
        "breathing_score": 70,
        "symptom_score": 70,
        "conf_heart": 0.8,
        "conf_breathing": 0.8,
        "conf_symptom": 0.8,
    }
    pred2 = fusion_model.predict_score(feat)
    assert pred2 is not None
    assert 0 <= pred2 <= 100


def test_fallback_when_no_model():
    # Ensure loader returns False when no path set
    os.environ.pop("VITA_FUSION_MODEL_PATH", None)
    ok = fusion_model.load_fusion_model()
    # loader returns False and predict_score returns None
    assert ok is False
    assert fusion_model.predict_score(50, 50, 50, [0.5, 0.5, 0.5]) is None
