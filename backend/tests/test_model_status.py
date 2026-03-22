"""
Tests for model_status module.
Covers: mark_model_loaded, get_model_status, get_all_model_status,
        list_active_models, source tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest



# ═══════════════════════════════════════════════════════════════════════════
# mark_model_loaded
# ═══════════════════════════════════════════════════════════════════════════

class TestMarkModelLoaded:
    def setup_method(self):
        """Reset internal state before each test."""
        from backend.app.ml.registry import model_status
        model_status._LOADED.clear()
        model_status._ACTIVE.clear()
        model_status._ERRORS.clear()
        model_status._SOURCES.clear()

    def test_mark_loaded_basic(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, is_loaded, is_active
        mark_model_loaded("yamnet", loaded=True)
        assert is_loaded("yamnet") is True
        assert is_active("yamnet") is False

    def test_mark_loaded_with_active(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, is_active
        mark_model_loaded("biobert", loaded=True, active=True)
        assert is_active("biobert") is True

    def test_mark_loaded_with_source(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, _SOURCES
        mark_model_loaded("yamnet", loaded=True, source="saved_model")
        assert _SOURCES["yamnet"] == "saved_model"

    def test_mark_loaded_with_error(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, _ERRORS
        mark_model_loaded("distilbert", loaded=False, error="file not found")
        assert _ERRORS["distilbert"] == "file not found"

    def test_mark_loaded_clears_error(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, _ERRORS
        mark_model_loaded("biobert", loaded=False, error="broken")
        assert "biobert" in _ERRORS
        mark_model_loaded("biobert", loaded=True)
        assert "biobert" not in _ERRORS

    def test_unloaded_model_defaults(self):
        from backend.app.ml.registry.model_status import is_loaded, is_active
        assert is_loaded("nonexistent") is False
        assert is_active("nonexistent") is False


# ═══════════════════════════════════════════════════════════════════════════
# get_model_status
# ═══════════════════════════════════════════════════════════════════════════

class TestGetModelStatus:
    def setup_method(self):
        from backend.app.ml.registry import model_status
        model_status._LOADED.clear()
        model_status._ACTIVE.clear()
        model_status._ERRORS.clear()
        model_status._SOURCES.clear()

    def test_auto_model_status(self):
        from backend.app.ml.registry.model_status import get_model_status
        with patch("backend.app.ml.registry.model_status.is_model_cached", return_value=True):
            status = get_model_status("distilbert")
        assert status["cached"] is True
        assert "loaded" in status
        assert "active" in status

    def test_manual_model_cached_has_installed_note(self):
        from backend.app.ml.registry.model_status import get_model_status
        with patch("backend.app.ml.registry.model_status.is_model_cached", return_value=True):
            status = get_model_status("fusion")
        assert "note" in status
        assert "installed" in status["note"]

    def test_source_included_when_set(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, get_model_status
        mark_model_loaded("yamnet", loaded=True, source="saved_model")
        with patch("backend.app.ml.registry.model_status.is_model_cached", return_value=True):
            status = get_model_status("yamnet")
        assert status.get("source") == "saved_model"

    def test_error_included(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, get_model_status
        mark_model_loaded("distilbert", loaded=False, error="corrupt file")
        with patch("backend.app.ml.registry.model_status.is_model_cached", return_value=False):
            status = get_model_status("distilbert")
        assert status.get("error") == "corrupt file"


# ═══════════════════════════════════════════════════════════════════════════
# get_all_model_status & list_active_models
# ═══════════════════════════════════════════════════════════════════════════

class TestAllModelStatus:
    def setup_method(self):
        from backend.app.ml.registry import model_status
        model_status._LOADED.clear()
        model_status._ACTIVE.clear()
        model_status._ERRORS.clear()
        model_status._SOURCES.clear()

    def test_returns_all_known_models(self):
        from backend.app.ml.registry.model_status import get_all_model_status, ALL_MODELS
        with patch("backend.app.ml.registry.model_status.is_model_cached", return_value=False):
            all_status = get_all_model_status()
        assert set(all_status.keys()) == ALL_MODELS

    def test_list_active_models_empty(self):
        from backend.app.ml.registry.model_status import list_active_models
        assert list_active_models() == []

    def test_list_active_models_after_activation(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, list_active_models
        mark_model_loaded("yamnet", loaded=True, active=True)
        mark_model_loaded("biobert", loaded=True, active=True)
        mark_model_loaded("distilbert", loaded=True, active=False)
        active = list_active_models()
        assert "yamnet" in active
        assert "biobert" in active
        assert "distilbert" not in active


# ═══════════════════════════════════════════════════════════════════════════
# Model sets
# ═══════════════════════════════════════════════════════════════════════════

class TestModelSets:
    def test_auto_downloadable_models(self):
        from backend.app.ml.registry.model_status import AUTO_DOWNLOADABLE
        assert "distilbert" in AUTO_DOWNLOADABLE
        assert "biobert" in AUTO_DOWNLOADABLE
        assert "yamnet" in AUTO_DOWNLOADABLE
        assert "open_rppg" in AUTO_DOWNLOADABLE

    def test_manual_models_contains_fusion(self):
        from backend.app.ml.registry.model_status import MANUAL_MODELS
        assert "fusion" in MANUAL_MODELS

    def test_all_models_is_union(self):
        from backend.app.ml.registry.model_status import AUTO_DOWNLOADABLE, MANUAL_MODELS, ALL_MODELS
        assert ALL_MODELS == AUTO_DOWNLOADABLE | MANUAL_MODELS
