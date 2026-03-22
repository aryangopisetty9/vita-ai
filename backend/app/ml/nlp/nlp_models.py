"""
Vita AI - Pretrained NLP Models (v4 - auto-download + local cache)
===================================================================
Transformer-based NLP models for symptom text analysis.

Supported models (in priority order):
1. **BioBERT** - biomedical domain BERT.
2. **DistilBERT** - lightweight sentiment / tone classifier (fallback).

Both models are auto-downloaded on first use and cached in
``models_cache/biobert/`` and ``models_cache/distilbert/``.

When neither is available the caller falls back to keyword-based
symptom analysis.

Environment Variables
---------------------
VITA_ENABLE_BIOBERT      - "true" (default) or "false"
VITA_ENABLE_DISTILBERT   - "true" (default) or "false"
VITA_BIOBERT_MODEL       - HuggingFace model ID or local path override
VITA_AUTO_DOWNLOAD_MODELS - "true" (default) or "false"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from backend.app.ml.registry.model_registry import get_model_path, mark_loaded

logger = logging.getLogger(__name__)

# -- Cached model pipelines ---------------------------------------------------
_BIOBERT_PIPELINE: Optional[Any] = None
_BIOBERT_TOKENIZER: Optional[Any] = None
_BIOBERT_MODEL_OBJ: Optional[Any] = None
_BIOBERT_LOADED = False

_DISTILBERT_PIPELINE: Optional[Any] = None
_DISTILBERT_TOKENIZER: Optional[Any] = None
_DISTILBERT_MODEL_OBJ: Optional[Any] = None
_DISTILBERT_LOADED = False

# -- Optional heavy imports ----------------------------------------------------
_HAS_TRANSFORMERS = False
_HAS_TORCH = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

try:
    import transformers as _tf_lib
    _HAS_TRANSFORMERS = True
except ImportError:
    pass


# ==============================================================================
# Helpers
# ==============================================================================

def _resolve_model_path(name: str) -> Optional[str]:
    """Resolve model path: env var override -> project cache -> None."""
    reg_path = get_model_path(name)
    if reg_path:
        return reg_path
    try:
        from backend.app.ml.registry.model_paths import is_model_cached, get_cache_dir
        if is_model_cached(name):
            return str(get_cache_dir(name))
    except Exception:
        pass
    return None


def _maybe_auto_download(name: str) -> Optional[str]:
    """If auto-download is enabled and model not cached, download it now."""
    try:
        from backend.app.ml.registry.model_download import is_auto_download_enabled, ensure_downloaded
        if is_auto_download_enabled():
            if ensure_downloaded(name):
                from backend.app.ml.registry.model_paths import get_cache_dir
                return str(get_cache_dir(name))
    except Exception as exc:
        logger.debug("Auto-download check for %s: %s", name, exc)
    return None


def _notify_status(name: str, loaded: bool, active: bool, error: str | None = None) -> None:
    """Update both the old registry and the new status tracker."""
    mark_loaded(name, loaded, error)
    try:
        from backend.app.ml.registry.model_status import mark_model_loaded
        mark_model_loaded(name, loaded, active, error)
    except Exception:
        pass


def _label_to_severity(
    label: str,
    confidence: float,
    probs: Optional[Any] = None,
) -> float:
    """Map a classification label to a [0,1] severity score."""
    label_lower = label.lower()
    if any(kw in label_lower for kw in ("negative", "severe", "high", "critical", "abnormal")):
        return confidence
    if any(kw in label_lower for kw in ("positive", "normal", "low", "healthy", "benign")):
        return 1.0 - confidence
    if label_lower == "label_0":
        return confidence
    if label_lower == "label_1":
        return 1.0 - confidence
    return confidence * 0.5


# ==============================================================================
# BioBERT loader & inference
# ==============================================================================

def _load_biobert() -> bool:
    global _BIOBERT_PIPELINE, _BIOBERT_TOKENIZER, _BIOBERT_MODEL_OBJ, _BIOBERT_LOADED
    if _BIOBERT_LOADED:
        return _BIOBERT_PIPELINE is not None or _BIOBERT_MODEL_OBJ is not None
    _BIOBERT_LOADED = True

    try:
        from backend.app.ml.registry.model_download import is_biobert_enabled
        if not is_biobert_enabled():
            logger.info("BioBERT disabled via VITA_ENABLE_BIOBERT.")
            _notify_status("biobert", False, False, "disabled")
            return False
    except Exception:
        pass

    if not _HAS_TRANSFORMERS:
        _notify_status("biobert", False, False, "transformers not installed")
        return False

    model_id = _resolve_model_path("biobert")
    if not model_id:
        model_id = _maybe_auto_download("biobert")
    if not model_id:
        _notify_status("biobert", False, False, "not configured and auto-download unavailable")
        return False

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        logger.info("Loading BioBERT from %s ...", model_id)
        _BIOBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _BIOBERT_MODEL_OBJ = AutoModel.from_pretrained(model_id)
        _BIOBERT_MODEL_OBJ.eval()
        _BIOBERT_PIPELINE = True  # sentinel - inference uses forward pass
        _notify_status("biobert", True, True)
        logger.info("BioBERT loaded successfully.")
        return True
    except Exception as exc:
        logger.info("BioBERT not available (%s) - will use DistilBERT fallback.", exc)
        _notify_status("biobert", False, False, str(exc))
        return False


def _infer_biobert(text: str) -> Optional[Dict[str, Any]]:
    if _BIOBERT_MODEL_OBJ is None or _BIOBERT_TOKENIZER is None:
        return None
    if not _HAS_TORCH:
        return None
    try:
        inputs = _BIOBERT_TOKENIZER(
            text[:512], return_tensors="pt", truncation=True, padding=True,
        )
        with torch.no_grad():
            outputs = _BIOBERT_MODEL_OBJ(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        norm = float(torch.norm(cls_embedding).item())
        severity = float(min(max((norm - 15.0) / 15.0, 0.0), 1.0))
        confidence = round(min(0.5 + severity * 0.4, 0.95), 3)

        return {
            "severity_score": round(severity, 3),
            "confidence": confidence,
            "model": "biobert",
            "raw_label": f"cls_norm={norm:.1f}",
            "model_available": True,
            "model_loaded": True,
            "model_cached": True,
            "inference_source": "biobert_forward",
        }
    except Exception as exc:
        logger.warning("BioBERT inference failed: %s", exc)
        return None


# ==============================================================================
# DistilBERT loader & inference
# ==============================================================================

def _load_distilbert() -> bool:
    global _DISTILBERT_PIPELINE, _DISTILBERT_TOKENIZER, _DISTILBERT_MODEL_OBJ, _DISTILBERT_LOADED
    if _DISTILBERT_LOADED:
        return _DISTILBERT_PIPELINE is not None
    _DISTILBERT_LOADED = True

    try:
        from backend.app.ml.registry.model_download import is_distilbert_enabled
        if not is_distilbert_enabled():
            logger.info("DistilBERT disabled via VITA_ENABLE_DISTILBERT.")
            _notify_status("distilbert", False, False, "disabled")
            return False
    except Exception:
        pass

    if not _HAS_TRANSFORMERS:
        _notify_status("distilbert", False, False, "transformers not installed")
        return False

    model_id = _resolve_model_path("distilbert")
    if not model_id:
        model_id = _maybe_auto_download("distilbert")
    if not model_id:
        from backend.app.ml.registry.model_paths import DISTILBERT_HF_MODEL_ID
        model_id = DISTILBERT_HF_MODEL_ID

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline as hf_pipeline,
        )

        logger.info("Loading DistilBERT from %s ...", model_id)
        _DISTILBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _DISTILBERT_MODEL_OBJ = AutoModelForSequenceClassification.from_pretrained(model_id)
        _DISTILBERT_MODEL_OBJ.eval()
        _DISTILBERT_PIPELINE = hf_pipeline(
            "sentiment-analysis",
            model=_DISTILBERT_MODEL_OBJ,
            tokenizer=_DISTILBERT_TOKENIZER,
            device=-1,
        )
        _notify_status("distilbert", True, True)
        logger.info("DistilBERT loaded successfully.")
        return True
    except Exception as exc:
        logger.info("DistilBERT not available: %s", exc)
        _notify_status("distilbert", False, False, str(exc))
        return False


def _infer_distilbert(text: str) -> Optional[Dict[str, Any]]:
    if _DISTILBERT_PIPELINE is None:
        return None
    try:
        result = _DISTILBERT_PIPELINE(text[:512])
        if isinstance(result, list) and len(result) > 0:
            top = result[0]
            label = str(top.get("label", ""))
            score = float(top.get("score", 0.0))
            if label == "NEGATIVE":
                severity = score
            else:
                severity = 1.0 - score
            return {
                "severity_score": round(severity, 3),
                "confidence": round(score, 3),
                "model": "distilbert",
                "raw_label": label,
                "model_available": True,
                "model_loaded": True,
                "model_cached": True,
                "inference_source": "distilbert",
            }
    except Exception as exc:
        logger.warning("DistilBERT inference failed: %s", exc)
    return None


# ==============================================================================
# Unified inference (priority chain)
# ==============================================================================

def infer_nlp_models(text: str) -> Dict[str, Any]:
    """Try BioBERT first, then DistilBERT; return best result or fallback."""
    available = get_available_models()

    _load_biobert()
    bio_result = _infer_biobert(text)
    if bio_result is not None:
        bio_result["available_models"] = available
        return bio_result

    _load_distilbert()
    distil_result = _infer_distilbert(text)
    if distil_result is not None:
        distil_result["available_models"] = available
        return distil_result

    return {
        "severity_score": None, "confidence": 0.0,
        "model": "none", "available_models": available,
        "model_available": False, "model_loaded": False,
        "model_cached": False,
        "inference_source": "keyword_rules",
    }


def compare_with_distilbert(
    model_result: Dict[str, Any],
    distilbert_score: Optional[float],
    keyword_risk: str,
    keyword_confidence: float,
) -> Dict[str, Any]:
    """Compare pretrained model output with DistilBERT + keyword pipeline."""
    model_conf = model_result.get("confidence", 0.0)
    model_severity = model_result.get("severity_score")
    model_name = model_result.get("model", "none")

    distilbert_conf = distilbert_score if distilbert_score is not None else 0.0

    if model_name == "none" or model_severity is None:
        return {
            "severity_score": distilbert_score,
            "confidence": keyword_confidence,
            "risk": keyword_risk,
            "source": "distilbert_keywords",
            "model_name": model_name,
            "model_available": False,
            "model_loaded": False,
            "model_cached": False,
            "inference_source": "distilbert_keywords",
            "selection_reason": "No pretrained NLP model available; using DistilBERT + keywords.",
            "available_models": model_result.get("available_models", []),
        }

    if model_severity > 0.7:
        model_risk = "high"
    elif model_severity > 0.4:
        model_risk = "moderate"
    else:
        model_risk = "low"

    if keyword_risk == "high" and model_risk == "low":
        model_risk = "moderate"

    if model_conf > distilbert_conf:
        blended_conf = round(min((model_conf + keyword_confidence) / 2.0, 1.0), 3)
        return {
            "severity_score": model_severity,
            "confidence": blended_conf,
            "risk": model_risk,
            "source": model_name,
            "model_name": model_name,
            "model_available": True,
            "model_loaded": True,
            "model_cached": True,
            "inference_source": model_name,
            "selection_reason": f"{model_name} confidence ({model_conf:.2f}) > DistilBERT ({distilbert_conf:.2f}).",
            "available_models": model_result.get("available_models", []),
        }

    return {
        "severity_score": distilbert_score,
        "confidence": keyword_confidence,
        "risk": keyword_risk,
        "source": "distilbert_keywords",
        "model_name": model_name,
        "model_available": True,
        "model_loaded": True,
        "model_cached": True,
        "inference_source": "distilbert_keywords",
        "selection_reason": f"DistilBERT confidence ({distilbert_conf:.2f}) >= {model_name} ({model_conf:.2f}).",
        "available_models": model_result.get("available_models", []),
    }


def get_available_models() -> List[str]:
    """Return names of available NLP models."""
    available: List[str] = []
    if _load_biobert():
        available.append("biobert")
    if _load_distilbert():
        available.append("distilbert")
    elif _HAS_TRANSFORMERS:
        available.append("distilbert")
    return available
