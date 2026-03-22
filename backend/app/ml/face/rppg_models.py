"""
Vita AI – rPPG Models (Open-rPPG only)
========================================
rPPG model inference using Open-rPPG as the primary backend.
When Open-rPPG is unavailable the caller falls back to the classical
signal-processing pipeline.

Environment Variables
---------------------
VITA_OPEN_RPPG_MODEL – Open-rPPG model name (default: "FacePhys.rlap")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.ml.registry.model_registry import get_model_path, mark_loaded
from backend.app.ml.face.open_rppg_backend import (
    infer_open_rppg,
    is_open_rppg_available,
    load_open_rppg,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Status notification helper
# ═══════════════════════════════════════════════════════════════════════════

def _notify_rppg_status(name: str, loaded: bool, active: bool = False,
                        error: str | None = None, source: str | None = None) -> None:
    """Update both the old registry and the new status tracker."""
    mark_loaded(name, loaded, error)
    try:
        from backend.app.ml.registry.model_status import mark_model_loaded
        mark_model_loaded(name, loaded, active, error, source)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Open-rPPG loader / inferrer
# ═══════════════════════════════════════════════════════════════════════════

def _load_open_rppg() -> bool:
    """Attempt to load the Open-rPPG model (top priority)."""
    if not is_open_rppg_available():
        return False
    model_name = os.getenv("VITA_OPEN_RPPG_MODEL", "FacePhys.rlap")
    return load_open_rppg(model_name)


def _infer_open_rppg_wrapper(frames: List[np.ndarray], fps: float) -> Optional[Dict[str, Any]]:
    """Thin wrapper so _MODEL_CHAIN can call Open-rPPG uniformly."""
    result = infer_open_rppg(frames, fps)
    if result.get("bpm") is None:
        return None
    return result


_MODEL_CHAIN = [
    ("open_rppg", _load_open_rppg, _infer_open_rppg_wrapper),
]


def infer_rppg_models(frames: List[np.ndarray], fps: float) -> Dict[str, Any]:
    """Try each pretrained rPPG model in priority order.

    Returns the result from the model with the highest confidence,
    or a fallback dict if no model is available.
    """
    available: List[str] = []
    results: List[Dict[str, Any]] = []

    for name, loader, inferrer in _MODEL_CHAIN:
        if loader():
            available.append(name)
            res = inferrer(frames, fps)
            if res is not None:
                results.append(res)

    if results:
        best = max(results, key=lambda r: r.get("confidence", 0))
        best["available_models"] = available
        best["model_available"] = True
        best["model_loaded"] = True
        best["inference_source"] = best.get("model", "unknown")
        best["model_priority"] = [n for n, _, _ in _MODEL_CHAIN]
        best["legacy_fallback_used"] = False
        best["open_rppg_active"] = best.get("inference_source", "").startswith("open_rppg")
        best["classical_fallback_used"] = False
        return best

    return {
        "bpm": None, "confidence": 0.0, "model": "none",
        "available_models": available, "waveform": None,
        "model_available": False, "model_loaded": False,
        "inference_source": "classical_pipeline",
        "model_priority": [n for n, _, _ in _MODEL_CHAIN],
        "legacy_fallback_used": False,
        "open_rppg_active": False,
        "classical_fallback_used": True,
    }


def compare_with_signal_pipeline(
    model_result: Dict[str, Any],
    signal_bpm: float,
    signal_quality: float,
) -> Dict[str, Any]:
    """Compare deep-model rPPG output with the classical signal pipeline.

    Uses SNR-aware blending when both sources agree.
    """
    model_bpm = model_result.get("bpm")
    model_conf = model_result.get("confidence", 0.0)
    model_name = model_result.get("model", "none")
    model_snr = model_result.get("snr", 0.0)

    if model_name == "none" or model_bpm is None:
        return {
            "bpm": signal_bpm, "confidence": signal_quality,
            "source": "signal_pipeline", "model_name": model_name,
            "model_available": False, "model_loaded": False,
            "inference_source": "classical_pipeline",
            "selection_reason": "No pretrained rPPG model available.",
        }

    bpm_diff = abs(model_bpm - signal_bpm)

    # Agreement: blend weighted by confidence
    if bpm_diff < 10 and model_conf > 0.3 and signal_quality > 0.1:
        total_w = model_conf + signal_quality
        blended_bpm = (model_bpm * model_conf + signal_bpm * signal_quality) / total_w
        blended_conf = min((model_conf + signal_quality) / 1.8, 1.0)
        return {
            "bpm": round(blended_bpm, 1),
            "confidence": round(blended_conf, 3),
            "source": "blended", "model_name": model_name,
            "model_available": True, "model_loaded": True,
            "inference_source": "blended",
            "model_snr": model_snr,
            "selection_reason": f"Model and signal agree (Δ {bpm_diff:.1f} BPM) — blended.",
        }

    # Disagreement: pick higher-confidence source
    if model_conf > signal_quality:
        return {
            "bpm": model_bpm, "confidence": model_conf,
            "source": model_name, "model_name": model_name,
            "model_available": True, "model_loaded": True,
            "inference_source": model_name,
            "model_snr": model_snr,
            "selection_reason": f"Model confidence ({model_conf:.2f}) > signal ({signal_quality:.2f}).",
        }

    return {
        "bpm": signal_bpm, "confidence": signal_quality,
        "source": "signal_pipeline", "model_name": model_name,
        "model_available": True, "model_loaded": True,
        "inference_source": "classical_pipeline",
        "model_snr": model_snr,
        "selection_reason": f"Signal confidence ({signal_quality:.2f}) >= model ({model_conf:.2f}).",
    }


def get_available_models() -> List[str]:
    """Return names of models whose loaders succeed."""
    available: List[str] = []
    for name, loader, _ in _MODEL_CHAIN:
        if loader():
            available.append(name)
    return available
