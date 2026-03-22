"""
Vita AI – Open-rPPG Backend
=============================
Wraps the ``open-rppg`` package (PyPI: ``open-rppg``, import: ``rppg``)
as the **primary** deep face/heart-rate inference engine.

Open-rPPG ships with pretrained weights for multiple architectures
(FacePhys, EfficientPhys, PhysFormer, PhysMamba, RhythmMamba, TSCAN,
PhysNet, etc.) — no manual weight download required.

Public API
----------
- ``is_open_rppg_available()``   – True if the package is importable
- ``load_open_rppg(model_name)`` – initialise the rppg.Model singleton
- ``infer_open_rppg(frames, fps)`` – run heart-rate inference on frames
- ``get_open_rppg_status()``     – structured status dict for health endpoint
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Ensure KERAS_BACKEND is set (defence-in-depth) ──────────────────────
# backend/__init__.py already sets this, but if someone imports this module
# directly we still need it before the rppg package touches Keras.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ── Optional import ──────────────────────────────────────────────────────
_HAS_OPEN_RPPG = False
_SUPPORTED_MODELS: List[str] = []

try:
    from rppg import Model as _RppgModel, supported_models as _supported  # type: ignore
    _HAS_OPEN_RPPG = True
    _SUPPORTED_MODELS = list(_supported)
except ImportError:
    _RppgModel = None  # type: ignore
    logger.info("open-rppg not installed – Open-rPPG backend unavailable.")
except Exception as exc:
    _RppgModel = None  # type: ignore
    logger.warning("open-rppg import failed: %s", exc)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ── Singleton state ──────────────────────────────────────────────────────
_MODEL_INSTANCE: Optional[Any] = None
_MODEL_NAME: Optional[str] = None
_MODEL_LOADED = False
_MODEL_ERROR: Optional[str] = None
_LOCK = threading.Lock()

# Default model — FacePhys with RLAP training (best balanced option)
DEFAULT_MODEL = "FacePhys.rlap"

# Fallback model variants to try if the default fails
_FALLBACK_MODELS = [
    "FacePhys.rlap",
    "EfficientPhys.rlap",
    "PhysNet.rlap",
    "TSCAN.rlap",
    "EfficientPhys.pure",
    "PhysNet.pure",
]


# ═══════════════════════════════════════════════════════════════════════════
# Public helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_open_rppg_available() -> bool:
    """Return True if the ``open-rppg`` package can be imported."""
    return _HAS_OPEN_RPPG


def get_supported_models() -> List[str]:
    """Return list of model names supported by open-rppg."""
    return list(_SUPPORTED_MODELS)


# ═══════════════════════════════════════════════════════════════════════════
# Model lifecycle
# ═══════════════════════════════════════════════════════════════════════════

def load_open_rppg(model_name: str | None = None) -> bool:
    """Initialise the Open-rPPG model singleton.

    Tries the requested model first, then falls back through
    ``_FALLBACK_MODELS`` if the initial load fails.

    Parameters
    ----------
    model_name : str, optional
        One of ``rppg.supported_models`` (e.g. ``'FacePhys.rlap'``).
        Defaults to ``DEFAULT_MODEL`` or ``VITA_OPEN_RPPG_MODEL`` env var.

    Returns
    -------
    bool
        True if the model was loaded (or was already loaded).
    """
    global _MODEL_INSTANCE, _MODEL_NAME, _MODEL_LOADED, _MODEL_ERROR

    if not _HAS_OPEN_RPPG:
        _MODEL_ERROR = "open-rppg package not installed"
        return False

    name = model_name or os.getenv("VITA_OPEN_RPPG_MODEL", DEFAULT_MODEL)

    with _LOCK:
        # Already loaded with same model?
        if _MODEL_LOADED and _MODEL_INSTANCE is not None and _MODEL_NAME == name:
            return True

        # Build ordered candidate list: requested model first, then fallbacks
        candidates = [name] + [m for m in _FALLBACK_MODELS if m != name]

        for candidate in candidates:
            if candidate not in _SUPPORTED_MODELS:
                continue
            ok = _try_load_model(candidate)
            if ok:
                return True
            logger.info("Open-rPPG variant '%s' failed, trying next …", candidate)

        # All candidates exhausted
        _MODEL_ERROR = _MODEL_ERROR or "All Open-rPPG model variants failed to load"
        _notify_status(False, error=_MODEL_ERROR)
        logger.warning("Open-rPPG: all model variants failed — classical fallback will be used.")
        return False


def _try_load_model(name: str) -> bool:
    """Attempt to load a single Open-rPPG model variant.

    Uses a background thread with a timeout so that extremely slow JAX
    JIT compilation doesn't block the startup event loop forever.
    """
    global _MODEL_INSTANCE, _MODEL_NAME, _MODEL_LOADED, _MODEL_ERROR

    load_timeout = float(os.getenv("VITA_OPEN_RPPG_LOAD_TIMEOUT", "300"))
    result_holder: Dict[str, Any] = {"instance": None, "error": None}

    def _load_worker():
        try:
            result_holder["instance"] = _RppgModel(model=name)
        except Exception as exc:
            result_holder["error"] = str(exc)

    logger.info("Loading Open-rPPG model: %s (timeout=%.0fs) …", name, load_timeout)
    t0 = time.monotonic()
    worker = threading.Thread(target=_load_worker, daemon=True)
    worker.start()
    worker.join(timeout=load_timeout)

    if worker.is_alive():
        err = f"Open-rPPG model '{name}' load timed out after {load_timeout:.0f}s"
        logger.warning(err)
        _MODEL_ERROR = err
        _notify_status(False, error=err)
        return False

    elapsed = time.monotonic() - t0
    instance = result_holder["instance"]
    error = result_holder["error"]

    if error or instance is None:
        err_msg = error or "Model() returned None"
        logger.warning("Failed to load Open-rPPG model '%s': %s (%.1fs)", name, err_msg, elapsed)
        _MODEL_ERROR = err_msg
        _notify_status(False, error=err_msg)
        return False

    _MODEL_INSTANCE = instance
    _MODEL_NAME = name
    _MODEL_LOADED = True
    _MODEL_ERROR = None
    _notify_status(True, active=True, source=f"open-rppg:{name}")
    logger.info(
        "Open-rPPG initialized successfully — model='%s', fps=%s, input=%s (%.1fs)",
        name, instance.fps, instance.input, elapsed,
    )
    return True


def _notify_status(loaded: bool, active: bool = False,
                   error: str | None = None, source: str | None = None) -> None:
    """Update model_status tracker for 'open_rppg'."""
    try:
        from backend.app.ml.registry.model_status import mark_model_loaded
        mark_model_loaded("open_rppg", loaded, active, error, source)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Tensor-safety helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_numpy(x: Any) -> np.ndarray:
    """Convert any tensor-like (TF, JAX, torch) to a numpy array.

    Prevents ''EagerTensor is not a valid JAX type'' by materialising
    non-numpy objects before they cross framework boundaries.
    """
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):          # TF EagerTensor / torch Tensor
        return np.asarray(x.numpy())
    return np.asarray(x)


def _ensure_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    """Normalise a frame to uint8 BGR for Open-rPPG."""
    frame = _to_numpy(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255)
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

def infer_open_rppg(
    frames: List[np.ndarray],
    fps: float,
    *,
    timeout_sec: float = 30.0,
) -> Dict[str, Any]:
    """Run Open-rPPG heart-rate inference on a list of BGR frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        BGR video frames (H, W, 3) from OpenCV.
    fps : float
        Frame rate of the source video.
    timeout_sec : float
        Maximum time allowed for inference (safety limit).

    Returns
    -------
    dict
        Keys: hr, sqi, hrv, bpm, confidence, model, waveform,
              open_rppg_active, open_rppg_backend_name, latency.
    """
    if not _MODEL_LOADED or _MODEL_INSTANCE is None:
        return _fallback_result("Open-rPPG model not loaded")

    if not frames or len(frames) < 30:
        return _fallback_result("Too few frames for Open-rPPG inference")

    if not _HAS_CV2:
        return _fallback_result("OpenCV not available")

    model = _MODEL_INSTANCE

    try:
        t0 = time.monotonic()

        # --- Feed frames through Open-rPPG's context-manager pipeline ---
        with model:
            for i, frame in enumerate(frames):
                ts = i / fps  # synthetic timestamp based on fps
                # Ensure frame is a plain numpy uint8 array — prevents
                # TF EagerTensors or JAX DeviceArrays from leaking across
                # framework boundaries.
                safe_frame = _ensure_uint8_bgr(frame)
                model.update_frame(safe_frame, ts=ts)

                # Safety: abort if taking too long
                if time.monotonic() - t0 > timeout_sec:
                    logger.warning("Open-rPPG inference timed out after %.1fs", timeout_sec)
                    break

        elapsed = time.monotonic() - t0

        # --- Extract results ---
        hr_result = model.hr(return_hrv=True)

        if hr_result is None:
            return _fallback_result("Open-rPPG returned no HR result")

        hr = hr_result.get("hr")
        sqi = hr_result.get("SQI")
        hrv = hr_result.get("hrv", {})
        latency = hr_result.get("latency", 0.0)

        # Convert any JAX/TF scalars to plain Python floats
        hr = float(_to_numpy(np.asarray(hr))) if hr is not None else None
        sqi = float(_to_numpy(np.asarray(sqi))) if sqi is not None else None

        # Extract BVP waveform (materialise to plain list)
        bvp, bvp_ts = model.bvp()
        bvp = _to_numpy(np.asarray(bvp)) if hasattr(bvp, '__len__') and len(bvp) > 0 else np.array([])
        waveform = bvp.tolist()[:300] if len(bvp) > 0 else []

        if hr is None or not (30 <= hr <= 220):
            return _fallback_result(
                f"Open-rPPG HR out of range: {hr}",
                partial_data={"sqi": sqi, "waveform_length": len(waveform)},
            )

        # Compute confidence from SQI
        confidence = 0.0
        if sqi is not None:
            confidence = float(np.clip(sqi, 0.0, 1.0))

        return {
            "bpm": round(float(hr), 1),
            "confidence": round(confidence, 3),
            "model": f"open_rppg:{_MODEL_NAME}",
            "snr": round(float(sqi), 3) if sqi is not None else 0.0,
            "waveform": waveform[:300],
            "waveform_length": len(waveform),
            "hrv": _safe_hrv(hrv),
            "latency": round(float(latency), 3),
            "inference_time_sec": round(elapsed, 2),
            # Status fields
            "open_rppg_active": True,
            "open_rppg_backend_name": _MODEL_NAME,
            "model_available": True,
            "model_loaded": True,
            "inference_source": "open_rppg",
            "classical_fallback_used": False,
        }

    except Exception as exc:
        logger.warning("Open-rPPG inference error: %s", exc)
        return _fallback_result(f"Open-rPPG inference exception: {exc}")


def _safe_hrv(hrv: Any) -> Dict[str, Any]:
    """Sanitise HRV dict for JSON serialisation (handles JAX/TF tensors)."""
    if not isinstance(hrv, dict):
        return {}
    clean: Dict[str, Any] = {}
    for k, v in hrv.items():
        try:
            v = _to_numpy(np.asarray(v)) if not isinstance(v, (int, float, type(None))) else v
        except Exception:
            clean[k] = None
            continue
        if isinstance(v, (int, float)):
            if np.isnan(v) or np.isinf(v):
                clean[k] = None
            else:
                clean[k] = round(float(v), 4)
        elif isinstance(v, np.ndarray):
            clean[k] = [round(float(x), 4) if not np.isnan(x) else None for x in v.flat[:20]]
        else:
            clean[k] = v
    return clean


def _fallback_result(reason: str, partial_data: dict | None = None) -> Dict[str, Any]:
    """Return a structured result indicating Open-rPPG was unavailable."""
    result = {
        "bpm": None,
        "confidence": 0.0,
        "model": "none",
        "snr": 0.0,
        "waveform": [],
        "waveform_length": 0,
        "open_rppg_active": False,
        "open_rppg_backend_name": _MODEL_NAME,
        "model_available": _HAS_OPEN_RPPG,
        "model_loaded": _MODEL_LOADED,
        "inference_source": "classical_pipeline",
        "classical_fallback_used": True,
        "fallback_reason": reason,
    }
    if partial_data:
        result.update(partial_data)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════════

def get_open_rppg_status() -> Dict[str, Any]:
    """Return structured status for health/status endpoints."""
    active = _MODEL_LOADED and _MODEL_INSTANCE is not None
    status: Dict[str, Any] = {
        "installed": _HAS_OPEN_RPPG,
        "loaded": _MODEL_LOADED,
        "active": active,
        "model_name": _MODEL_NAME,
        "supported_models": _SUPPORTED_MODELS,
        "error": _MODEL_ERROR,
    }
    if not active:
        status["fallback"] = "classical"
    return status
