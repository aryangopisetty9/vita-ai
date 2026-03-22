"""
Vita AI – Model Paths
======================
Resolves and manages project-local cache directories for pretrained
models.  Every downloadable model gets a deterministic subdirectory
under ``<project_root>/models_cache/``.

Supported models: DistilBERT, BioBERT, YAMNet, Open-rPPG, Fusion.
"""

from __future__ import annotations

import os
from pathlib import Path

# model_paths.py lives at backend/app/ml/registry/model_paths.py
# parents[4] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODELS_CACHE_DIR = PROJECT_ROOT / "backend" / "data" / "models_cache"

# ── Per-model cache directories ─────────────────────────────────────────
DISTILBERT_CACHE_DIR = MODELS_CACHE_DIR / "distilbert"
BIOBERT_CACHE_DIR = MODELS_CACHE_DIR / "biobert"
YAMNET_CACHE_DIR = MODELS_CACHE_DIR / "yamnet_saved_model"

# ── Hugging Face model IDs ──────────────────────────────────────────────
DISTILBERT_HF_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
BIOBERT_HF_MODEL_ID = "dmis-lab/biobert-base-cased-v1.1"

# ── TF Hub URL ──────────────────────────────────────────────────────────
YAMNET_TFHUB_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)


def ensure_cache_dirs() -> None:
    """Create all cache directories if they don't exist."""
    for d in (
        MODELS_CACHE_DIR,
        DISTILBERT_CACHE_DIR,
        BIOBERT_CACHE_DIR,
        YAMNET_CACHE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def is_model_cached(name: str) -> bool:
    """Check if a model's cache directory contains files."""
    if name == "open_rppg":
        try:
            from backend.app.ml.face.open_rppg_backend import is_open_rppg_available
            return is_open_rppg_available()
        except Exception:
            return False
    cache_map = {
        "distilbert": DISTILBERT_CACHE_DIR,
        "biobert": BIOBERT_CACHE_DIR,
        "yamnet": YAMNET_CACHE_DIR,
    }
    cache_dir = cache_map.get(name)
    if cache_dir is None or not cache_dir.exists():
        return False
    return any(cache_dir.iterdir())


def get_cache_dir(name: str) -> Path:
    """Return the cache directory for a given model name."""
    cache_map = {
        "distilbert": DISTILBERT_CACHE_DIR,
        "biobert": BIOBERT_CACHE_DIR,
        "yamnet": YAMNET_CACHE_DIR,
    }
    return cache_map.get(name, MODELS_CACHE_DIR / name)
