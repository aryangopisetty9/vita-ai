"""
Tests for model download / cache system.

All downloads are mocked to avoid network I/O during tests.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest



# ── model_paths tests ────────────────────────────────────────────────────

class TestModelPaths:
    def test_cache_dirs_exist_after_ensure(self, tmp_path: Path):
        with mock.patch("backend.app.ml.registry.model_paths.MODELS_CACHE_DIR", tmp_path):
            from backend.app.ml.registry.model_paths import ensure_cache_dirs
            ensure_cache_dirs()
            assert tmp_path.exists()

    def test_is_model_cached_false_when_empty(self, tmp_path: Path):
        with mock.patch("backend.app.ml.registry.model_paths.MODELS_CACHE_DIR", tmp_path):
            cache_dir = tmp_path / "distilbert"
            with mock.patch("backend.app.ml.registry.model_paths.DISTILBERT_CACHE_DIR", cache_dir):
                from backend.app.ml.registry.model_paths import is_model_cached
                assert is_model_cached("distilbert") is False

    def test_is_model_cached_true_when_has_files(self, tmp_path: Path):
        cache_dir = tmp_path / "distilbert"
        cache_dir.mkdir()
        (cache_dir / "config.json").write_text("{}")
        with mock.patch("backend.app.ml.registry.model_paths.DISTILBERT_CACHE_DIR", cache_dir):
            from backend.app.ml.registry.model_paths import is_model_cached
            assert is_model_cached("distilbert") is True

    def test_get_cache_dir(self):
        from backend.app.ml.registry.model_paths import get_cache_dir, DISTILBERT_CACHE_DIR
        assert get_cache_dir("distilbert") == DISTILBERT_CACHE_DIR


# ── model_download tests ─────────────────────────────────────────────────

class TestModelDownload:
    def test_is_auto_download_enabled_default(self):
        # Default is true
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITA_AUTO_DOWNLOAD_MODELS", None)
            from backend.app.ml.registry.model_download import is_auto_download_enabled
            # Re-import to pick up env change - use _env_flag directly
            from backend.app.ml.registry.model_download import _env_flag
            assert _env_flag("VITA_AUTO_DOWNLOAD_MODELS", True) is True

    def test_is_auto_download_disabled(self):
        with mock.patch.dict(os.environ, {"VITA_AUTO_DOWNLOAD_MODELS": "false"}):
            from backend.app.ml.registry.model_download import _env_flag
            assert _env_flag("VITA_AUTO_DOWNLOAD_MODELS", True) is False

    @mock.patch("backend.app.ml.registry.model_download.is_distilbert_enabled", return_value=True)
    @mock.patch("backend.app.ml.registry.model_download.is_model_cached", return_value=True)
    def test_download_distilbert_skips_if_cached(self, mock_cached, mock_enabled):
        from backend.app.ml.registry.model_download import download_distilbert
        result = download_distilbert()
        assert result is True  # Returns True because it's cached

    @mock.patch("backend.app.ml.registry.model_download.is_biobert_enabled", return_value=False)
    def test_download_biobert_skips_if_disabled(self, mock_enabled):
        from backend.app.ml.registry.model_download import download_biobert
        result = download_biobert()
        assert result is False

    @mock.patch("backend.app.ml.registry.model_download.is_yamnet_enabled", return_value=False)
    def test_download_yamnet_skips_if_disabled(self, mock_enabled):
        from backend.app.ml.registry.model_download import download_yamnet
        result = download_yamnet()
        assert result is False

    @mock.patch("backend.app.ml.registry.model_download.download_distilbert", return_value=True)
    @mock.patch("backend.app.ml.registry.model_download.download_biobert", return_value=True)
    @mock.patch("backend.app.ml.registry.model_download.download_yamnet", return_value=False)
    def test_download_all_supported(self, mock_y, mock_b, mock_d):
        from backend.app.ml.registry.model_download import download_all_supported
        result = download_all_supported()
        assert result["distilbert"] is True
        assert result["biobert"] is True
        assert result["yamnet"] is False


# ── model_status tests ───────────────────────────────────────────────────

class TestModelStatus:
    def test_mark_and_query(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, is_loaded, is_active, get_model_status
        mark_model_loaded("test_model", loaded=True, active=True)
        assert is_loaded("test_model") is True
        assert is_active("test_model") is True
        status = get_model_status("test_model")
        assert status["loaded"] is True
        assert status["active"] is True
        assert "error" not in status  # no error key when no error

    def test_mark_with_error(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, get_model_status
        mark_model_loaded("err_model", loaded=False, active=False, error="bad weights")
        status = get_model_status("err_model")
        assert status["loaded"] is False
        assert status["error"] == "bad weights"

    def test_get_all_model_status(self):
        from backend.app.ml.registry.model_status import get_all_model_status
        all_status = get_all_model_status()
        assert isinstance(all_status, dict)

    def test_list_active_models(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, list_active_models, _ACTIVE
        # Use a real model name from ALL_MODELS
        mark_model_loaded("distilbert", loaded=True, active=True)
        active = list_active_models()
        assert "distilbert" in active


# ── ensure_downloaded integration ────────────────────────────────────────

class TestEnsureDownloaded:
    @mock.patch("backend.app.ml.registry.model_download.download_distilbert", return_value=True)
    @mock.patch("backend.app.ml.registry.model_download.is_auto_download_enabled", return_value=True)
    def test_ensure_downloaded_distilbert(self, mock_auto, mock_dl):
        from backend.app.ml.registry.model_download import ensure_downloaded
        assert ensure_downloaded("distilbert") is True
        mock_dl.assert_called_once()

    def test_ensure_downloaded_unknown_model(self):
        from backend.app.ml.registry.model_download import ensure_downloaded
        assert ensure_downloaded("unknown_model_xyz") is False
