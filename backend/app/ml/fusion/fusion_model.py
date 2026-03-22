"""
Vita AI – Trained Fusion Model
================================
Loads and runs a trained ML fusion model (sklearn GradientBoosting or
XGBoost) for combining face, audio, and symptom sub-scores into the
Vita Health Score.  Falls back to weighted-sum when no trained model
is available.

If no pre-trained model file exists the module will **auto-train** a
model from synthetic data on first load so that the fusion path is
always active.

Environment variable
--------------------
``VITA_FUSION_MODEL_PATH`` – path to a saved model file.
  Supported formats:

  * ``.json`` – XGBoost ``save_model`` format
  * ``.pkl``  – scikit-learn / XGBoost pickle via ``joblib``

When the env var is unset the module searches the default cache path
``backend/data/models_cache/fusion/fusion_trained.pkl``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_FUSION_MODEL: Optional[Any] = None
_FUSION_META: Dict[str, Any] = {}
_LOAD_ATTEMPTED = False

# Default path relative to project root
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models_cache" / "fusion"
_DEFAULT_MODEL_PKL = _DEFAULT_MODEL_DIR / "fusion_trained.pkl"
_DEFAULT_MODEL_JSON = _DEFAULT_MODEL_DIR / "fusion_trained.json"

FEATURE_ORDER = [
    "heart_score",
    "breathing_score",
    "symptom_score",
    "conf_heart",
    "conf_breathing",
    "conf_symptom",
]


# ── Synthetic data generation + auto-training ───────────────────────────

def _generate_synthetic_training_data(n: int = 2000) -> tuple:
    """Generate realistic synthetic fusion training samples.

    Ground truth is based on domain-weighted scoring with non-linear
    interactions: very low combined scores are penalised more than
    the linear average would suggest.
    """
    rng = np.random.default_rng(42)

    heart = rng.uniform(15, 100, n)
    breathing = rng.uniform(15, 100, n)
    symptom = rng.uniform(15, 100, n)
    conf_h = rng.uniform(0.1, 1.0, n)
    conf_b = rng.uniform(0.1, 1.0, n)
    conf_s = rng.uniform(0.1, 1.0, n)

    X = np.column_stack([heart, breathing, symptom, conf_h, conf_b, conf_s])

    # Weighted base (matches DEFAULT_WEIGHTS 40/30/30)
    base = 0.40 * heart + 0.30 * breathing + 0.30 * symptom

    # Non-linear: penalise when multiple modules are bad
    min_score = np.minimum(np.minimum(heart, breathing), symptom)
    penalty = np.where(min_score < 40, (40 - min_score) * 0.3, 0.0)

    # Confidence weighting: low confidence → pull toward 50
    avg_conf = (conf_h + conf_b + conf_s) / 3.0
    y = base * (0.6 + 0.4 * avg_conf) - penalty

    # Add realistic noise
    y += rng.normal(0, 2.0, n)
    y = np.clip(y, 0, 100)

    return X, y


def _auto_train_fusion() -> bool:
    """Train a fusion model from synthetic data and save to default path."""
    global _FUSION_MODEL, _FUSION_META

    try:
        import joblib
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        logger.info("scikit-learn not installed — cannot auto-train fusion model.")
        return False

    logger.info("Auto-training fusion model from synthetic data …")
    X, y = _generate_synthetic_training_data(2000)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)

    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = str(_DEFAULT_MODEL_PKL)
    joblib.dump(model, save_path)

    _FUSION_MODEL = model
    _FUSION_META = {
        "backend": "sklearn",
        "path": save_path,
        "format": "pkl",
        "auto_trained": True,
        "n_samples": len(y),
    }
    logger.info("Fusion model auto-trained and saved → %s", save_path)
    return True


# ── Model loading ───────────────────────────────────────────────────────

def _resolve_model_path() -> Optional[str]:
    """Find the fusion model file from env var or default cache."""
    path = os.getenv("VITA_FUSION_MODEL_PATH", "")
    if path and os.path.isfile(path):
        return path
    # Check default cache locations
    if _DEFAULT_MODEL_PKL.is_file():
        return str(_DEFAULT_MODEL_PKL)
    if _DEFAULT_MODEL_JSON.is_file():
        return str(_DEFAULT_MODEL_JSON)
    return None


def _load_fusion_model() -> bool:
    """Try to load a trained fusion model, auto-training if needed."""
    global _FUSION_MODEL, _FUSION_META, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _FUSION_MODEL is not None
    _LOAD_ATTEMPTED = True

    path = _resolve_model_path()
    if not path:
        # No model file found — try auto-training
        if _auto_train_fusion():
            return True
        logger.info("No trained fusion model found. Using weighted-sum fallback.")
        return False

    try:
        if path.endswith(".json"):
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(path)
            _FUSION_MODEL = model
            _FUSION_META = {"backend": "xgboost", "path": path, "format": "json"}
            logger.info("XGBoost fusion model loaded from %s", path)
            return True

        if path.endswith(".pkl"):
            import joblib
            model = joblib.load(path)
            _FUSION_MODEL = model
            backend = type(model).__module__.split(".")[0]
            _FUSION_META = {"backend": backend, "path": path, "format": "pkl"}
            logger.info("Fusion model loaded from %s (backend=%s)", path, backend)
            return True

        logger.warning("Unknown fusion model format: %s (expected .json or .pkl)", path)
        return False

    except Exception as exc:
        logger.warning("Failed to load fusion model from %s: %s", path, exc)
        _FUSION_META = {"error": str(exc)}
        return False


def predict_score(
    heart_score: float,
    breathing_score: float,
    symptom_score: float,
    confidences: List[float],
) -> Optional[float]:
    """Run the fusion model if available.

    Parameters
    ----------
    heart_score, breathing_score, symptom_score : float
        Normalised 0-100 sub-scores.
    confidences : list[float]
        Per-module confidence values (up to 3).

    Returns
    -------
    float or None
        Fused 0-100 score, or None if no model is loaded.
    """
    # Backwards-compatible wrapper: allow either the historical signature
    # (heart_score, breathing_score, symptom_score, confidences)
    # or a single dict of named features. This keeps existing callers
    # working while allowing richer inputs later.
    _load_fusion_model()
    if _FUSION_MODEL is None:
        return None

    # If caller passed a features dict as the first positional arg
    if isinstance(heart_score, dict):
        feat_dict = heart_score
    else:
        conf = (confidences + [0.0, 0.0, 0.0])[:3]
        feat_dict = {
            "heart_score": float(heart_score),
            "breathing_score": float(breathing_score),
            "symptom_score": float(symptom_score),
            "conf_heart": float(conf[0]),
            "conf_breathing": float(conf[1]),
            "conf_symptom": float(conf[2]),
        }

    # Keep a deterministic feature ordering used by the training script
    feature_order = FEATURE_ORDER
    try:
        # All features must be present — the score_engine only calls
        # this function when every module returned a real score.
        missing = [k for k in feature_order if k not in feat_dict]
        if missing:
            logger.warning("Fusion model missing features %s — refusing to predict with fake defaults.", missing)
            return None
        X = np.array([[float(feat_dict[k]) for k in feature_order]])
        prediction = float(_FUSION_MODEL.predict(X)[0])
        return round(float(np.clip(prediction, 0, 100)), 1)
    except Exception as exc:
        logger.warning("Fusion model prediction failed: %s", exc)
        return None


def get_fusion_status() -> Dict[str, Any]:
    """Return fusion model status for health endpoint."""
    _load_fusion_model()
    # Augment with runtime model_status if available
    status = {
        "available": _FUSION_MODEL is not None,
        "method": _FUSION_META.get("backend", "weighted_sum") if _FUSION_MODEL else "weighted_sum",
    }
    status.update({k: v for k, v in _FUSION_META.items() if k != "path"})
    # Integrate with global runtime tracker if present
    try:
        from backend.app.ml.registry.model_status import mark_model_loaded
        mark_model_loaded("fusion", bool(_FUSION_MODEL), bool(_FUSION_MODEL), None, source="trained_model")
    except Exception:
        pass
    return status


def load_fusion_model() -> bool:
    """Public loader used by the registry startup sequence.

    Returns True if a trained model was successfully loaded.
    """
    ok = _load_fusion_model()
    try:
        from backend.app.ml.registry.model_status import mark_model_loaded
        if ok:
            mark_model_loaded("fusion", True, True, None, source=_FUSION_META.get("backend"))
        else:
            err = _FUSION_META.get("error")
            mark_model_loaded("fusion", False, False, err, source="weighted_sum")
    except Exception:
        pass
    return ok
