"""
Vita AI – Model Status
=======================
Aggregates download / cache / load / active status for every model
in the system.  Used by the ``/health`` and ``/status`` endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from backend.app.ml.registry.model_download import (
    is_biobert_enabled, is_distilbert_enabled, is_yamnet_enabled,
)
from backend.app.ml.registry.model_paths import is_model_cached

logger = logging.getLogger(__name__)

# ── Runtime load state (set by model loaders at load time) ───────────────
_LOADED: Dict[str, bool] = {}
_ACTIVE: Dict[str, bool] = {}
_ERRORS: Dict[str, str] = {}
_SOURCES: Dict[str, str] = {}  # "downloaded", "manual", etc.

# Models that can be fully auto-downloaded from Python
AUTO_DOWNLOADABLE = {"distilbert", "biobert", "yamnet", "open_rppg"}
# Other manual models
MANUAL_MODELS = {"fusion"}
ALL_MODELS = AUTO_DOWNLOADABLE | MANUAL_MODELS


def mark_model_loaded(name: str, loaded: bool, active: bool = False,
                      error: str | None = None, source: str | None = None) -> None:
    """Called by model loaders after attempting to load a model."""
    _LOADED[name] = loaded
    _ACTIVE[name] = active
    if source:
        _SOURCES[name] = source
    if error:
        _ERRORS[name] = error
    elif name in _ERRORS:
        del _ERRORS[name]


def is_loaded(name: str) -> bool:
    return _LOADED.get(name, False)


def is_active(name: str) -> bool:
    return _ACTIVE.get(name, False)


def get_model_status(name: str) -> Dict[str, Any]:
    """Full status for one model."""
    # open_rppg uses its own status tracker
    if name == "open_rppg":
        try:
            from backend.app.ml.face.open_rppg_backend import get_open_rppg_status
            st = get_open_rppg_status()
            return {
                "cached": st["installed"],  # pip-installed ≡ cached
                "installed": st["installed"],
                "loaded": st["loaded"],
                "active": st["active"],
                "model_name": st.get("model_name"),
                "source": "open-rppg",
            }
        except Exception:
            return {"cached": False, "installed": False, "loaded": False, "active": False}

    cached = is_model_cached(name)
    loaded = _LOADED.get(name, False)
    active = _ACTIVE.get(name, False)
    error = _ERRORS.get(name)
    source = _SOURCES.get(name)

    info: Dict[str, Any] = {
        "cached": cached,
        "loaded": loaded,
        "active": active,
    }

    if source:
        info["source"] = source

    if error:
        info["error"] = error

    if name in MANUAL_MODELS:
        info["note"] = "installed" if cached else "manual – train or supply weights"

    return info


def get_all_model_status() -> Dict[str, Dict[str, Any]]:
    """Full status dict for all known models."""
    return {name: get_model_status(name) for name in sorted(ALL_MODELS)}


def list_active_models() -> list[str]:
    """Return names of models currently active for inference."""
    return [name for name in sorted(ALL_MODELS) if _ACTIVE.get(name, False)]
