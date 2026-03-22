"""
Tests for audio model loading with cache integration.

All model loads/downloads are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest



class TestYamnetCacheIntegration:
    def test_resolve_yamnet_path_returns_none_when_not_configured(self):
        with mock.patch("backend.app.ml.audio.audio_models.get_model_path", return_value=None):
            with mock.patch("backend.app.ml.registry.model_paths.is_model_cached", return_value=False):
                with mock.patch("backend.app.ml.registry.model_download.is_auto_download_enabled", return_value=False):
                    from backend.app.ml.audio.audio_models import _resolve_yamnet_path
                    assert _resolve_yamnet_path() is None

    def test_resolve_yamnet_path_from_cache(self):
        with mock.patch("backend.app.ml.audio.audio_models.get_model_path", return_value=None):
            with mock.patch("backend.app.ml.registry.model_paths.is_model_cached", return_value=True):
                with mock.patch("backend.app.ml.registry.model_paths.get_cache_dir", return_value=Path("/fake/yamnet")):
                    from backend.app.ml.audio.audio_models import _resolve_yamnet_path
                    result = _resolve_yamnet_path()
                    assert result is not None


class TestInferAudioModels:
    def test_fallback_when_no_models(self):
        """When no models load, should return librosa_pipeline fallback."""
        import backend.app.ml.audio.audio_models as am
        # Set LOADED=True so the early-return path fires,
        # returning (MODEL is not None) → False.
        am._YAMNET_LOADED = True
        am._YAMNET_MODEL = None

        audio = np.zeros(16000, dtype=np.float32)
        result = am.infer_audio_models(audio, 16000)
        assert result["model"] == "none"
        assert result["inference_source"] == "librosa_pipeline"
        assert result["model_cached"] is False


class TestCompareWithLibrosa:
    def test_no_model_available(self):
        from backend.app.ml.audio.audio_models import compare_with_librosa_pipeline
        model_result = {"model": "none", "confidence": 0.0, "model_available": False}
        out = compare_with_librosa_pipeline(model_result, "low", 0.7, 16.0)
        assert out["source"] == "librosa_pipeline"
        assert out["model_available"] is False

    def test_model_escalates_risk(self):
        from backend.app.ml.audio.audio_models import compare_with_librosa_pipeline
        model_result = {
            "model": "yamnet", "confidence": 0.8,
            "respiratory_detected": True,
            "health_concern": True,
            "model_available": True, "model_loaded": True,
            "labels": ["Cough"],
        }
        out = compare_with_librosa_pipeline(model_result, "low", 0.5, 16.0)
        assert out["source"] == "blended"
        assert out["risk"] in ("moderate", "high")
