"""
Vita AI – Model Download / Cache Manager
==========================================
Downloads and caches pretrained models locally inside the project.

Auto-downloadable models:
- **DistilBERT** (Hugging Face ``transformers``)
- **BioBERT**    (Hugging Face ``transformers``)
- **YAMNet**     (TensorFlow Hub → SavedModel)

Environment Variable Overrides
------------------------------
VITA_ENABLE_DISTILBERT           – "true" (default) or "false"
VITA_ENABLE_BIOBERT              – "true" (default) or "false"
VITA_ENABLE_YAMNET               – "true" (default) or "false"
VITA_AUTO_DOWNLOAD_MODELS        – "true" (default) or "false"
"""

from __future__ import annotations

import logging
import os
from typing import Dict

from backend.app.ml.registry.model_paths import (
    BIOBERT_CACHE_DIR,
    BIOBERT_HF_MODEL_ID,
    DISTILBERT_CACHE_DIR,
    DISTILBERT_HF_MODEL_ID,
    MODELS_CACHE_DIR,
    YAMNET_CACHE_DIR,
    YAMNET_CLASS_MAP_URL,
    YAMNET_TFHUB_URL,
    ensure_cache_dirs,
    is_model_cached,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = True) -> bool:
    """Read a boolean flag from an environment variable."""
    val = os.getenv(name, "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    return default


def is_auto_download_enabled() -> bool:
    return _env_flag("VITA_AUTO_DOWNLOAD_MODELS", default=True)


def is_distilbert_enabled() -> bool:
    return _env_flag("VITA_ENABLE_DISTILBERT", default=True)


def is_biobert_enabled() -> bool:
    return _env_flag("VITA_ENABLE_BIOBERT", default=True)


def is_yamnet_enabled() -> bool:
    return _env_flag("VITA_ENABLE_YAMNET", default=True)


# ═══════════════════════════════════════════════════════════════════════════
# DistilBERT
# ═══════════════════════════════════════════════════════════════════════════

def download_distilbert(force: bool = False) -> bool:
    """Download DistilBERT model + tokenizer to local cache.

    Returns True on success.
    """
    if not is_distilbert_enabled():
        logger.info("DistilBERT download disabled via VITA_ENABLE_DISTILBERT.")
        return False
    if is_model_cached("distilbert") and not force:
        logger.info("DistilBERT already cached at %s", DISTILBERT_CACHE_DIR)
        return True
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        ensure_cache_dirs()
        logger.info("Downloading DistilBERT (%s) …", DISTILBERT_HF_MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_HF_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_HF_MODEL_ID)
        tokenizer.save_pretrained(str(DISTILBERT_CACHE_DIR))
        model.save_pretrained(str(DISTILBERT_CACHE_DIR))
        logger.info("DistilBERT saved to %s", DISTILBERT_CACHE_DIR)
        return True
    except ImportError:
        logger.error("Cannot download DistilBERT – 'transformers' package not installed.")
        return False
    except Exception as exc:
        logger.error("DistilBERT download failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# BioBERT
# ═══════════════════════════════════════════════════════════════════════════

def download_biobert(force: bool = False) -> bool:
    """Download BioBERT model + tokenizer to local cache.

    Returns True on success.
    """
    if not is_biobert_enabled():
        logger.info("BioBERT download disabled via VITA_ENABLE_BIOBERT.")
        return False
    if is_model_cached("biobert") and not force:
        logger.info("BioBERT already cached at %s", BIOBERT_CACHE_DIR)
        return True
    try:
        from transformers import BertModel, BertTokenizer

        ensure_cache_dirs()
        logger.info("Downloading BioBERT (%s) …", BIOBERT_HF_MODEL_ID)
        # BioBERT v1.1 ships only a vocab.txt and a config without
        # model_type, so Auto* classes fail; use Bert* directly.
        tokenizer = BertTokenizer.from_pretrained(BIOBERT_HF_MODEL_ID)
        model = BertModel.from_pretrained(BIOBERT_HF_MODEL_ID)
        tokenizer.save_pretrained(str(BIOBERT_CACHE_DIR))
        model.save_pretrained(str(BIOBERT_CACHE_DIR))
        logger.info("BioBERT saved to %s", BIOBERT_CACHE_DIR)
        return True
    except ImportError:
        logger.error("Cannot download BioBERT – 'transformers' package not installed.")
        return False
    except Exception as exc:
        logger.error("BioBERT download failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# YAMNet
# ═══════════════════════════════════════════════════════════════════════════

def download_yamnet(force: bool = False) -> bool:
    """Download YAMNet from TF Hub and export as SavedModel.

    Returns True on success.
    """
    if not is_yamnet_enabled():
        logger.info("YAMNet download disabled via VITA_ENABLE_YAMNET.")
        return False
    if is_model_cached("yamnet") and not force:
        logger.info("YAMNet already cached at %s", YAMNET_CACHE_DIR)
        return True
    try:
        import tensorflow as tf
        import tensorflow_hub as hub

        ensure_cache_dirs()
        logger.info("Downloading YAMNet from TF Hub …")
        model = hub.load(YAMNET_TFHUB_URL)

        # Save as a concrete SavedModel so future loads don't need Hub
        logger.info("Exporting YAMNet SavedModel to %s …", YAMNET_CACHE_DIR)
        tf.saved_model.save(model, str(YAMNET_CACHE_DIR))
        logger.info("YAMNet saved to %s", YAMNET_CACHE_DIR)

        # Also try to download the class map CSV
        _download_yamnet_class_map()
        return True
    except ImportError as exc:
        logger.error("Cannot download YAMNet – TensorFlow / TF-Hub not installed: %s", exc)
        return False
    except Exception as exc:
        logger.error("YAMNet download failed: %s", exc)
        return False


def _download_yamnet_class_map() -> None:
    """Download yamnet_class_map.csv into the cache directory."""
    csv_path = YAMNET_CACHE_DIR / "yamnet_class_map.csv"
    if csv_path.exists():
        return
    try:
        import urllib.request
        logger.info("Downloading YAMNet class map CSV …")
        urllib.request.urlretrieve(YAMNET_CLASS_MAP_URL, str(csv_path))
        logger.info("Class map saved to %s", csv_path)
    except Exception as exc:
        logger.warning("Could not download YAMNet class map: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Unified helpers
# ═══════════════════════════════════════════════════════════════════════════

def download_all_supported(force: bool = False) -> Dict[str, bool]:
    """Download all auto-downloadable models.

    Returns a dict of {model_name: success_bool}.
    """
    ensure_cache_dirs()
    results: Dict[str, bool] = {}
    results["distilbert"] = download_distilbert(force=force)
    results["biobert"] = download_biobert(force=force)
    results["yamnet"] = download_yamnet(force=force)
    return results


def ensure_downloaded(name: str, force: bool = False) -> bool:
    """Ensure a specific model is downloaded. Returns True if cached."""
    downloaders = {
        "distilbert": download_distilbert,
        "biobert": download_biobert,
        "yamnet": download_yamnet,
    }
    fn = downloaders.get(name)
    if fn is None:
        return is_model_cached(name)
    return fn(force=force)


def download_all_models(force: bool = False) -> Dict[str, bool]:
    """Download ALL downloadable models.

    Returns a dict of {model_name: success_bool}.
    """
    return download_all_supported(force=force)
