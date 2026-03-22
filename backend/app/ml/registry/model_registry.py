"""
Vita AI – Model Registry
==========================
Unified registry for all pretrained models.  Provides:

- Central configuration via environment variables
- Integration with project-local cache (``models_cache/``)
- Lazy loading with caching
- Status reporting for health endpoints
- Version metadata tracking

Auto-downloadable models (DistilBERT, BioBERT, YAMNet) resolve
their paths automatically from the project cache when no explicit env
var is set.

Environment Variables
---------------------
VITA_YAMNET_MODEL_PATH        – path to YAMNet saved-model dir or "tfhub"
VITA_BIOBERT_MODEL            – HuggingFace model ID or local path
VITA_FUSION_MODEL_PATH        – path to trained fusion model
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a single model entry."""
    name: str
    category: str                   # rppg | audio | nlp | fusion
    env_var: str                    # environment variable for path
    path: Optional[str] = None     # resolved path (from env or cache)
    available: bool = False        # True if path exists on disk
    loaded: bool = False           # True if weights are in memory
    version: str = "unknown"
    backend: str = "unknown"       # torch | tensorflow | sklearn | xgboost
    error: Optional[str] = None
    auto_downloadable: bool = False


# Singleton registry
_REGISTRY: Dict[str, ModelInfo] = {}
_INITIALIZED = False


def _resolve_cache_path(name: str) -> Optional[str]:
    """Check if a model has been cached locally in models_cache/."""
    try:
        from backend.app.ml.registry.model_paths import is_model_cached, get_cache_dir
        if is_model_cached(name):
            return str(get_cache_dir(name))
    except Exception:
        pass
    return None


def _init_registry() -> None:
    """Scan environment variables and project cache, then populate the registry."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    _entries = [
        # (name, category, env_var, backend, version, auto_downloadable)
        # Open-rPPG (primary, pip-installed with bundled weights)
        ("open_rppg",  "rppg",  "VITA_OPEN_RPPG_MODEL",      "open-rppg",    "0.1.1", True),
        # Audio
        ("yamnet",     "audio", "VITA_YAMNET_MODEL_PATH",     "tensorflow",   "1.0", True),
        # NLP
        ("biobert",    "nlp",   "VITA_BIOBERT_MODEL",         "transformers", "1.0", True),
        ("distilbert", "nlp",   "VITA_ENABLE_DISTILBERT",     "transformers", "1.0", True),
        # Fusion
        ("fusion",     "fusion","VITA_FUSION_MODEL_PATH",     "xgboost",      "1.0", False),
        # MediaPipe availability (tool/runtime dependency)
        ("mediapipe",  "tool",   "VITA_MEDIAPIPE",            "python",       "0.0", False),
    ]

    for name, category, env_var, backend, version, auto_dl in _entries:
        raw_path = os.getenv(env_var)
        available = False

        # Special case: open_rppg is available if the package is installed
        if name == "open_rppg":
            try:
                from backend.app.ml.face.open_rppg_backend import is_open_rppg_available
                available = is_open_rppg_available()
                raw_path = raw_path or "FacePhys.rlap"
            except Exception:
                pass
            _REGISTRY[name] = ModelInfo(
                name=name, category=category, env_var=env_var,
                path=raw_path, available=available, loaded=False,
                version=version, backend=backend,
                auto_downloadable=auto_dl,
            )
            continue

        if raw_path:
            # For HF model IDs or special tokens like "tfhub", mark as available
            if raw_path in ("tfhub",) or "/" in raw_path:
                available = True
            elif os.path.exists(raw_path):
                available = True
        # Special-case: mediapipe is "available" if the package can be imported
        if name == "mediapipe":
            try:
                import importlib
                available = importlib.util.find_spec("mediapipe") is not None
                raw_path = raw_path or "python-package"
            except Exception:
                available = False

        # Auto-downloadable models: also check project-local cache
        if not available:
            cached_path = _resolve_cache_path(name)
            if cached_path:
                raw_path = cached_path
                available = True

        _REGISTRY[name] = ModelInfo(
            name=name,
            category=category,
            env_var=env_var,
            path=raw_path,
            available=available,
            loaded=False,
            version=version,
            backend=backend,
            auto_downloadable=auto_dl,
        )

    logger.info(
        "Model registry initialized: %s",
        {k: v.available for k, v in _REGISTRY.items()},
    )


def refresh_registry() -> None:
    """Re-scan env vars and cache (e.g. after downloading models)."""
    global _INITIALIZED
    _INITIALIZED = False
    _REGISTRY.clear()
    _init_registry()


def get_model_info(name: str) -> Optional[ModelInfo]:
    """Return info for a single model."""
    _init_registry()
    return _REGISTRY.get(name)


def get_model_path(name: str) -> Optional[str]:
    """Return the configured path for a model, or None."""
    _init_registry()
    info = _REGISTRY.get(name)
    return info.path if info else None


def mark_loaded(name: str, loaded: bool = True, error: Optional[str] = None) -> None:
    """Mark a model as loaded (or failed) after an attempt."""
    _init_registry()
    if name in _REGISTRY:
        _REGISTRY[name].loaded = loaded
        _REGISTRY[name].error = error


def get_all_status() -> Dict[str, Dict[str, Any]]:
    """Return status dict for all models (used by /health endpoint)."""
    _init_registry()
    out = {}
    for name, info in _REGISTRY.items():
        out[name] = {
            "available": info.available,
            "loaded": info.loaded,
            "backend": info.backend,
            "version": info.version,
            "env_var": info.env_var,
            "path_configured": info.path is not None,
        }
        if info.error:
            out[name]["error"] = info.error
    return out


def get_models_by_category(category: str) -> List[ModelInfo]:
    """Return all models in a category (rppg, audio, nlp, fusion)."""
    _init_registry()
    return [m for m in _REGISTRY.values() if m.category == category]


def is_available(name: str) -> bool:
    """Check if a model has its path configured and file exists."""
    _init_registry()
    info = _REGISTRY.get(name)
    return info.available if info else False


def is_loaded(name: str) -> bool:
    """Check if a model is currently loaded in memory."""
    _init_registry()
    info = _REGISTRY.get(name)
    return info.loaded if info else False


def ensure_supported_models_downloaded() -> Dict[str, bool]:
    """Download all auto-downloadable models that are not yet cached.

    Returns a dict of {model_name: success_bool}.
    """
    try:
        from backend.app.ml.registry.model_download import download_all_supported
        results = download_all_supported()
        refresh_registry()
        return results
    except Exception as exc:
        logger.warning("ensure_supported_models_downloaded failed: %s", exc)
        return {}


def load_supported_models() -> Dict[str, bool]:
    """Attempt to load all auto-downloadable models into memory.

    Returns a dict of {model_name: loaded_bool}.
    """
    loaded: Dict[str, bool] = {}
    try:
        from backend.app.ml.nlp.nlp_models import get_available_models as get_nlp
        nlp_available = get_nlp()
        loaded["biobert"] = "biobert" in nlp_available
        loaded["distilbert"] = "distilbert" in nlp_available
    except Exception as exc:
        logger.warning("NLP model preload failed: %s", exc)
        loaded["biobert"] = False
        loaded["distilbert"] = False
    try:
        from backend.app.ml.audio.audio_models import get_available_models as get_audio
        audio_available = get_audio()
        loaded["yamnet"] = "yamnet" in audio_available
    except Exception as exc:
        logger.warning("Audio model preload failed: %s", exc)
        loaded["yamnet"] = False
    return loaded


def list_active_models() -> List[str]:
    """Return names of models currently loaded and active."""
    _init_registry()
    active = [name for name, info in _REGISTRY.items() if info.loaded]
    # Also check runtime status
    try:
        from backend.app.ml.registry.model_status import list_active_models as rt_active
        for name in rt_active():
            if name not in active:
                active.append(name)
    except Exception:
        pass
    return sorted(active)


def preload_all_models() -> Dict[str, bool]:
    """Eagerly load ALL supported models at startup.

    Each model is loaded inside its own try/except so a single failure
    does not block the others.  Returns ``{model_name: success}``.
    """
    results: Dict[str, bool] = {}

    from backend.app.ml.registry.model_status import mark_model_loaded

    # 1. Open-rPPG
    try:
        from backend.app.ml.face.open_rppg_backend import load_open_rppg
        import os as _os
        model_name = _os.getenv("VITA_OPEN_RPPG_MODEL", "FacePhys.rlap")
        logger.info("Preloading Open-rPPG (%s) …", model_name)
        results["open_rppg"] = load_open_rppg(model_name)
    except Exception as exc:
        logger.warning("Open-rPPG preload failed: %s", exc)
        results["open_rppg"] = False
        mark_model_loaded("open_rppg", False, False, str(exc))

    # 2. BioBERT
    try:
        from backend.app.ml.nlp.nlp_models import _load_biobert
        logger.info("Preloading BioBERT …")
        results["biobert"] = _load_biobert()
    except Exception as exc:
        logger.warning("BioBERT preload failed: %s", exc)
        results["biobert"] = False
        mark_model_loaded("biobert", False, False, str(exc))

    # 3. DistilBERT
    try:
        from backend.app.ml.nlp.nlp_models import _load_distilbert
        logger.info("Preloading DistilBERT …")
        results["distilbert"] = _load_distilbert()
    except Exception as exc:
        logger.warning("DistilBERT preload failed: %s", exc)
        results["distilbert"] = False
        mark_model_loaded("distilbert", False, False, str(exc))

    # 4. YAMNet
    try:
        from backend.app.ml.audio.audio_models import _load_yamnet
        logger.info("Preloading YAMNet …")
        results["yamnet"] = _load_yamnet()
    except Exception as exc:
        logger.warning("YAMNet preload failed: %s", exc)
        results["yamnet"] = False
        mark_model_loaded("yamnet", False, False, str(exc))

    # 5. Fusion (trained model if present)
    try:
        from backend.app.ml.fusion.fusion_model import load_fusion_model
        logger.info("Preloading fusion model (if trained) …")
        results["fusion"] = load_fusion_model()
    except Exception as exc:
        logger.warning("Fusion preload failed: %s", exc)
        results["fusion"] = False
        mark_model_loaded("fusion", False, False, str(exc))

    # 6. MediaPipe availability (runtime check)
    try:
        import importlib
        mp_available = importlib.util.find_spec("mediapipe") is not None
        # mark in registry status via model_status
        try:
            from backend.app.ml.registry.model_status import mark_model_loaded as _m
            _m("mediapipe", mp_available, mp_available, None, source="python-package")
        except Exception:
            pass
        results["mediapipe"] = mp_available
    except Exception:
        results["mediapipe"] = False

    # Update registry loaded flags
    for name, ok in results.items():
        mark_loaded(name, ok)

    # Log summary banner
    total = len(results)
    loaded = sum(1 for v in results.values() if v)
    failed = total - loaded

    logger.info("=" * 60)
    logger.info("  VITA AI – MODEL INITIALIZATION STATUS")
    logger.info("=" * 60)
    for name, ok in results.items():
        status = "LOADED ✓" if ok else "FAILED ✗"
        logger.info("  %-14s %s", name, status)
    logger.info("-" * 60)
    logger.info("  Total: %d | Loaded: %d | Failed: %d", total, loaded, failed)
    if failed:
        logger.warning("  Some models failed — fallback pipelines will be used.")
    else:
        logger.info("  All models loaded and active.")
    logger.info("=" * 60)

    return results
