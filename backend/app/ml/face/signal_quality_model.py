"""
Signal quality classifier for face rPPG candidates.

This module provides:
- Deterministic feature extraction from a pulse signal.
- Optional sklearn model inference (good vs bad signal).
- Rule-based fallback when no trained model is available.

The classifier is only used for quality gating/ranking. It never predicts BPM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.utils.signal_processing import (
    compute_inter_window_consistency,
    compute_peak_prominence,
    compute_periodicity,
    compute_signal_strength,
    compute_snr,
    signal_amplitude,
    signal_stability,
)

FEATURE_NAMES: List[str] = [
    "signal_strength",
    "snr",
    "periodicity",
    "peak_prominence",
    "inter_window_consistency",
    "peak_stability",
    "heart_band_power_ratio",
    "amplitude",
    "stability",
    "motion_contamination",
    "roi_stability",
    "brightness_quality",
    "overexposure_penalty",
    "window_consistency",
    "valid_window_ratio",
]


def _heart_band_power_ratio(
    signal: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
) -> float:
    if len(signal) < 8:
        return 0.0
    sig = signal.astype(np.float64)
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / max(fps, 1e-6))
    mag = np.abs(np.fft.rfft(sig))
    total = float(np.sum(mag))
    if total < 1e-12:
        return 0.0
    in_band = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.clip(np.sum(mag[in_band]) / total, 0.0, 1.0))


def extract_quality_features(
    signal: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    *,
    motion_contamination: float = 0.0,
    roi_stability: float = 1.0,
    brightness_quality: float = 0.5,
    overexposure_ratio: float = 0.0,
    window_consistency: float = 0.0,
    valid_window_ratio: float = 0.0,
) -> Dict[str, float]:
    """Extract quality features used by ML/rule quality classification."""
    ss = compute_signal_strength(signal, fps, low_hz, high_hz)
    snr = compute_snr(signal, fps, low_hz, high_hz)
    periodicity = compute_periodicity(signal, fps, low_hz, high_hz)
    prominence = compute_peak_prominence(signal, fps, low_hz, high_hz)
    consistency = compute_inter_window_consistency(signal, fps, 5.0, 1.0, low_hz, high_hz)
    amplitude = signal_amplitude(signal)
    stability = signal_stability(signal, fps)
    hbpr = _heart_band_power_ratio(signal, fps, low_hz, high_hz)

    features = {
        "signal_strength": float(np.clip(ss.get("signal_strength", 0.0), 0.0, 1.0)),
        "snr": float(np.clip(snr, 0.0, 1.0)),
        "periodicity": float(np.clip(periodicity, 0.0, 1.0)),
        "peak_prominence": float(np.clip(prominence, 0.0, 1.0)),
        "inter_window_consistency": float(np.clip(consistency, 0.0, 1.0)),
        "peak_stability": float(np.clip(ss.get("peak_stability", 0.0), 0.0, 1.0)),
        "heart_band_power_ratio": float(np.clip(hbpr, 0.0, 1.0)),
        "amplitude": float(np.clip(amplitude / 6.0, 0.0, 1.0)),
        "stability": float(np.clip(stability, 0.0, 1.0)),
        "motion_contamination": float(np.clip(motion_contamination, 0.0, 1.0)),
        "roi_stability": float(np.clip(roi_stability, 0.0, 1.0)),
        "brightness_quality": float(np.clip(brightness_quality, 0.0, 1.0)),
        "overexposure_penalty": float(np.clip(1.0 - overexposure_ratio / 0.40, 0.0, 1.0)),
        "window_consistency": float(np.clip(window_consistency, 0.0, 1.0)),
        "valid_window_ratio": float(np.clip(valid_window_ratio, 0.0, 1.0)),
    }
    return features


class _RuleQualityClassifier:
    """Deterministic fallback quality classifier."""

    def predict_good_probability(self, features: Dict[str, float]) -> float:
        p = (
            0.18 * features["signal_strength"]
            + 0.10 * features["snr"]
            + 0.10 * features["periodicity"]
            + 0.08 * features["peak_prominence"]
            + 0.08 * features["inter_window_consistency"]
            + 0.06 * features["peak_stability"]
            + 0.06 * features["heart_band_power_ratio"]
            + 0.06 * features["amplitude"]
            + 0.06 * features["stability"]
            + 0.08 * features["window_consistency"]
            + 0.07 * features["valid_window_ratio"]
            + 0.07 * features["roi_stability"]
            + 0.06 * features["brightness_quality"]
            + 0.04 * features["overexposure_penalty"]
        )
        p *= (1.0 - 0.50 * features["motion_contamination"])
        return float(np.clip(p, 0.0, 1.0))


class SignalQualityClassifier:
    """Optional sklearn classifier with rule-based fallback."""

    def __init__(self) -> None:
        self._model = None
        self._model_source = "rule_fallback"
        self._load_error: Optional[str] = None
        self._rule = _RuleQualityClassifier()
        self._try_load_model()

    def _try_load_model(self) -> None:
        model_path = os.getenv(
            "VITA_SIGNAL_QUALITY_MODEL",
            str(
                Path(__file__).resolve().parents[4]
                / "models_cache"
                / "face_quality"
                / "signal_quality_classifier.pkl"
            ),
        )
        if not os.path.isfile(model_path):
            self._load_error = "model_not_found"
            return
        try:
            import joblib

            payload = joblib.load(model_path)
            if isinstance(payload, dict) and "model" in payload:
                self._model = payload["model"]
            else:
                self._model = payload
            self._model_source = "ml_model"
            self._load_error = None
        except Exception as exc:
            self._model = None
            self._model_source = "rule_fallback"
            self._load_error = str(exc)

    def evaluate(self, features: Dict[str, float]) -> Dict[str, Any]:
        vector = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float64)

        if self._model is not None:
            try:
                if hasattr(self._model, "predict_proba"):
                    p_good = float(self._model.predict_proba(vector)[0][1])
                else:
                    pred = float(self._model.predict(vector)[0])
                    p_good = 1.0 if pred >= 0.5 else 0.0
                p_good = float(np.clip(p_good, 0.0, 1.0))
                return {
                    "good_signal_probability": round(p_good, 4),
                    "label": "good_signal" if p_good >= 0.5 else "bad_signal",
                    "source": "ml_model",
                    "load_error": None,
                }
            except Exception as exc:
                self._load_error = str(exc)

        p_good = self._rule.predict_good_probability(features)
        return {
            "good_signal_probability": round(p_good, 4),
            "label": "good_signal" if p_good >= 0.5 else "bad_signal",
            "source": "rule_fallback",
            "load_error": self._load_error,
        }


_QUALITY_CLASSIFIER: Optional[SignalQualityClassifier] = None


def get_signal_quality_classifier() -> SignalQualityClassifier:
    global _QUALITY_CLASSIFIER
    if _QUALITY_CLASSIFIER is None:
        _QUALITY_CLASSIFIER = SignalQualityClassifier()
    return _QUALITY_CLASSIFIER


def evaluate_signal_quality(features: Dict[str, float]) -> Dict[str, Any]:
    return get_signal_quality_classifier().evaluate(features)
