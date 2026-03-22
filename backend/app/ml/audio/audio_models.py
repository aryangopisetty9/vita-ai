"""
Vita AI – Pretrained Audio Models (YAMNet only)
========================================================================
Audio classifier for respiratory and health sound analysis.

Supported model:
1. **YAMNet** – TF/TFLite general audio event classifier (521 classes).
   Auto-downloaded from TF-Hub and cached in ``models_cache/yamnet_saved_model/``.

When unavailable the caller falls back to the Librosa signal-processing
pipeline.

Environment Variables
---------------------
VITA_ENABLE_YAMNET       – "true" (default) or "false"
VITA_YAMNET_MODEL_PATH   – SavedModel dir, "tfhub", or .tflite file (override)
VITA_AUTO_DOWNLOAD_MODELS – "true" (default) or "false"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.ml.registry.model_registry import get_model_path, mark_loaded

logger = logging.getLogger(__name__)

# ── Optional heavy imports ───────────────────────────────────────────────
try:
    import librosa as _librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

# ── Cached model objects ─────────────────────────────────────────────────
_YAMNET_MODEL: Optional[Any] = None
_YAMNET_CLASS_NAMES: Optional[List[str]] = None

_YAMNET_LOADED = False


# ═══════════════════════════════════════════════════════════════════════════
# Audio preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 16 kHz if needed."""
    if sr == 16000:
        return audio.astype(np.float32)
    if _HAS_LIBROSA:
        return _librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    # Manual linear interpolation fallback
    ratio = 16000 / sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Respiratory label matching
# ═══════════════════════════════════════════════════════════════════════════

_RESPIRATORY_KEYWORDS = {
    "cough", "wheeze", "sneeze", "snore", "gasp",
    "breathing", "respiratory", "stridor", "snoring",
    "hiccup", "sigh", "pant",
}

_HEALTH_CONCERN_KEYWORDS = {
    "cough", "wheeze", "gasp", "stridor",
    "respiratory", "choking",
}


def _is_respiratory(label: str) -> bool:
    label_lower = label.lower()
    return any(kw in label_lower for kw in _RESPIRATORY_KEYWORDS)


def _is_health_concern(label: str) -> bool:
    label_lower = label.lower()
    return any(kw in label_lower for kw in _HEALTH_CONCERN_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════════
# YAMNet loader & inference
# ═══════════════════════════════════════════════════════════════════════════

def _load_yamnet_class_names() -> List[str]:
    """Load YAMNet 521-class names."""
    global _YAMNET_CLASS_NAMES
    if _YAMNET_CLASS_NAMES is not None:
        return _YAMNET_CLASS_NAMES
    # Try loading from cache, then common locations
    try:
        import csv
        search_dirs = [
            os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "yamnet_class_map.csv"),
        ]
        # Also check the project cache
        try:
            from backend.app.ml.registry.model_paths import YAMNET_CACHE_DIR
            cache_csv = os.path.join(str(YAMNET_CACHE_DIR), "yamnet_class_map.csv")
            search_dirs.insert(0, cache_csv)
        except Exception:
            pass
        for base in search_dirs:
            if os.path.isfile(base):
                with open(base) as f:
                    reader = csv.DictReader(f)
                    _YAMNET_CLASS_NAMES = [
                        row.get("display_name", row.get("name", f"class_{i}"))
                        for i, row in enumerate(reader)
                    ]
                    return _YAMNET_CLASS_NAMES
    except Exception:
        pass
    _YAMNET_CLASS_NAMES = [f"class_{i}" for i in range(521)]
    return _YAMNET_CLASS_NAMES


def _notify_audio_status(name: str, loaded: bool, active: bool, error: str | None = None) -> None:
    """Update both the old registry and the new status tracker."""
    mark_loaded(name, loaded, error)
    try:
        from backend.app.ml.registry.model_status import mark_model_loaded
        mark_model_loaded(name, loaded, active, error)
    except Exception:
        pass


def _resolve_yamnet_path() -> Optional[str]:
    """Resolve YAMNet model path: env var -> cache -> auto-download -> None."""
    # 1. Check env var / registry
    reg_path = get_model_path("yamnet")
    if reg_path:
        return reg_path
    # 2. Check project-local cache
    try:
        from backend.app.ml.registry.model_paths import is_model_cached, get_cache_dir
        if is_model_cached("yamnet"):
            return str(get_cache_dir("yamnet"))
    except Exception:
        pass
    # 3. Auto-download
    try:
        from backend.app.ml.registry.model_download import is_auto_download_enabled, ensure_downloaded
        if is_auto_download_enabled():
            if ensure_downloaded("yamnet"):
                from backend.app.ml.registry.model_paths import get_cache_dir
                return str(get_cache_dir("yamnet"))
    except Exception as exc:
        logger.debug("YAMNet auto-download check: %s", exc)
    return None


def _load_yamnet() -> bool:
    global _YAMNET_MODEL, _YAMNET_LOADED
    if _YAMNET_LOADED:
        return _YAMNET_MODEL is not None
    _YAMNET_LOADED = True

    # Check enable flag
    try:
        from backend.app.ml.registry.model_download import is_yamnet_enabled
        if not is_yamnet_enabled():
            logger.info("YAMNet disabled via VITA_ENABLE_YAMNET.")
            _notify_audio_status("yamnet", False, False, "disabled")
            return False
    except Exception:
        pass

    model_path = _resolve_yamnet_path()
    if not model_path:
        logger.info("YAMNet: not configured and auto-download unavailable.")
        _notify_audio_status("yamnet", False, False, "not configured")
        return False

    try:
        if model_path == "tfhub":
            import tensorflow_hub as hub
            _YAMNET_MODEL = {"type": "tfhub", "model": hub.load("https://tfhub.dev/google/yamnet/1")}
        elif model_path.endswith(".tflite") and os.path.isfile(model_path):
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            _YAMNET_MODEL = {"type": "tflite", "interpreter": interpreter}
        elif os.path.isdir(model_path):
            import tensorflow as tf
            _YAMNET_MODEL = {"type": "saved_model", "model": tf.saved_model.load(model_path)}
        else:
            logger.info("YAMNet: path %s not found.", model_path)
            _notify_audio_status("yamnet", False, False, "path not found")
            return False
        _notify_audio_status("yamnet", True, True)
        _load_yamnet_class_names()
        logger.info("YAMNet loaded from %s", model_path)
        return True
    except ImportError as exc:
        logger.info("YAMNet: TensorFlow not installed – %s", exc)
        _notify_audio_status("yamnet", False, False, "tensorflow not installed")
        return False
    except Exception as exc:
        logger.warning("Failed to load YAMNet: %s", exc)
        _notify_audio_status("yamnet", False, False, str(exc))
        return False


def _infer_yamnet(audio: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
    if _YAMNET_MODEL is None:
        return None
    try:
        audio_16k = _resample_to_16k(audio, sr)
        class_names = _load_yamnet_class_names()

        model_type = _YAMNET_MODEL["type"]
        if model_type in ("tfhub", "saved_model"):
            import tensorflow as tf
            model = _YAMNET_MODEL["model"]
            waveform_tf = tf.constant(audio_16k)
            scores, embeddings, spectrogram = model(waveform_tf)
            scores_np = scores.numpy()
        elif model_type == "tflite":
            interpreter = _YAMNET_MODEL["interpreter"]
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.resize_tensor_input(input_details[0]['index'], audio_16k.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], audio_16k)
            interpreter.invoke()
            scores_np = interpreter.get_tensor(output_details[0]['index'])
        else:
            return None

        mean_scores = np.mean(scores_np, axis=0)
        top_5_indices = np.argsort(mean_scores)[-5:][::-1]
        top_label = class_names[top_5_indices[0]] if top_5_indices[0] < len(class_names) else f"class_{top_5_indices[0]}"
        top_confidence = float(mean_scores[top_5_indices[0]])
        detected_labels = [
            class_names[i] if i < len(class_names) else f"class_{i}"
            for i in top_5_indices
        ]
        respiratory = any(_is_respiratory(l) for l in detected_labels)
        health_concern = any(_is_health_concern(l) for l in detected_labels)

        return {
            "labels": detected_labels,
            "top_label": top_label,
            "confidence": round(top_confidence, 3),
            "respiratory_detected": respiratory,
            "health_concern": health_concern,
            "model": "yamnet",
            "model_available": True,
            "model_loaded": True,
            "model_cached": True,
            "inference_source": "yamnet",
        }
    except Exception as exc:
        logger.warning("YAMNet inference failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Unified inference
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_CHAIN = [
    ("yamnet", _load_yamnet, _infer_yamnet),
]


def infer_audio_models(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Try each pretrained audio model in priority order."""
    available: List[str] = []
    results: List[Dict[str, Any]] = []

    for name, loader, inferrer in _MODEL_CHAIN:
        if loader():
            available.append(name)
            res = inferrer(audio, sr)
            if res is not None:
                results.append(res)

    if results:
        best = max(results, key=lambda r: r.get("confidence", 0))
        best["available_models"] = available
        return best

    return {
        "labels": [], "top_label": None, "confidence": 0.0,
        "respiratory_detected": False, "model": "none",
        "available_models": available,
        "model_available": False, "model_loaded": False,
        "model_cached": False,
        "inference_source": "librosa_pipeline",
    }


def compare_with_librosa_pipeline(
    model_result: Dict[str, Any],
    librosa_risk: str,
    librosa_confidence: float,
    librosa_breathing_rate: Optional[float],
) -> Dict[str, Any]:
    """Compare deep-model output with the Librosa pipeline."""
    model_conf = model_result.get("confidence", 0.0)
    model_name = model_result.get("model", "none")
    respiratory = model_result.get("respiratory_detected", False)
    health_concern = model_result.get("health_concern", False)

    if model_name == "none" or model_conf <= 0:
        return {
            "risk": librosa_risk, "confidence": librosa_confidence,
            "breathing_rate": librosa_breathing_rate,
            "source": "librosa_pipeline", "model_name": model_name,
            "model_available": False, "model_loaded": False,
            "inference_source": "librosa_pipeline",
            "selection_reason": "No pretrained audio model available.",
        }

    # If model detects respiratory concern with high confidence
    if (respiratory or health_concern) and model_conf > 0.4:
        if health_concern:
            escalated_risk = "high" if librosa_risk != "low" else "moderate"
        else:
            escalated_risk = "moderate" if librosa_risk == "low" else librosa_risk
        blended_conf = round(min((model_conf + librosa_confidence) / 2.0, 1.0), 3)
        return {
            "risk": escalated_risk, "confidence": blended_conf,
            "breathing_rate": librosa_breathing_rate,
            "source": "blended", "model_name": model_name,
            "model_available": True, "model_loaded": True,
            "inference_source": "blended",
            "model_labels": model_result.get("labels", []),
            "selection_reason": f"{model_name} detected respiratory sounds — escalating risk.",
        }

    # Model available but no respiratory concern — use librosa as-is with model confirmation
    if model_conf > librosa_confidence:
        return {
            "risk": librosa_risk,
            "confidence": round(min(librosa_confidence + 0.1, 1.0), 3),
            "breathing_rate": librosa_breathing_rate,
            "source": "librosa_confirmed", "model_name": model_name,
            "model_available": True, "model_loaded": True,
            "inference_source": "librosa_confirmed",
            "selection_reason": f"{model_name} confirms normal audio — boosting confidence.",
        }

    return {
        "risk": librosa_risk, "confidence": librosa_confidence,
        "breathing_rate": librosa_breathing_rate,
        "source": "librosa_pipeline", "model_name": model_name,
        "model_available": True, "model_loaded": True,
        "inference_source": "librosa_pipeline",
        "selection_reason": "Librosa pipeline confidence higher than model.",
    }


def get_available_models() -> List[str]:
    """Return names of models whose loaders succeed."""
    available: List[str] = []
    for name, loader, _ in _MODEL_CHAIN:
        if loader():
            available.append(name)
    return available
