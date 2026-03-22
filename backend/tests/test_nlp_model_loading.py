"""
Tests for NLP model loading with cache integration.

All model loads are mocked to avoid downloading real weights.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest



class TestNlpModelHelpers:
    def test_resolve_model_path_returns_none_when_not_configured(self):
        with mock.patch("backend.app.ml.nlp.nlp_models.get_model_path", return_value=None):
            with mock.patch("backend.app.ml.registry.model_paths.is_model_cached", return_value=False):
                from backend.app.ml.nlp.nlp_models import _resolve_model_path
                assert _resolve_model_path("biobert") is None

    def test_resolve_model_path_returns_registry_path(self):
        with mock.patch("backend.app.ml.nlp.nlp_models.get_model_path", return_value="/fake/path"):
            from backend.app.ml.nlp.nlp_models import _resolve_model_path
            assert _resolve_model_path("biobert") == "/fake/path"

    def test_maybe_auto_download_disabled(self):
        with mock.patch("backend.app.ml.registry.model_download.is_auto_download_enabled", return_value=False):
            from backend.app.ml.nlp.nlp_models import _maybe_auto_download
            assert _maybe_auto_download("biobert") is None


class TestInferNlpModels:
    def test_fallback_when_no_models(self):
        """When no models load, should return keyword_rules fallback."""
        import backend.app.ml.nlp.nlp_models as nlp
        # Reset load flags
        nlp._BIOBERT_LOADED = False
        nlp._DISTILBERT_LOADED = False
        nlp._BIOBERT_PIPELINE = None
        nlp._BIOBERT_MODEL_OBJ = None
        nlp._DISTILBERT_PIPELINE = None

        with mock.patch.object(nlp, "_load_biobert", return_value=False):
            with mock.patch.object(nlp, "_load_distilbert", return_value=False):
                with mock.patch.object(nlp, "_infer_biobert", return_value=None):
                    with mock.patch.object(nlp, "_infer_distilbert", return_value=None):
                        result = nlp.infer_nlp_models("I have a headache")
                        assert result["model"] == "none"
                        assert result["inference_source"] == "keyword_rules"
                        assert result["model_cached"] is False

    def test_distilbert_inference_returns_correct_fields(self):
        """When DistilBERT works, result should have expected fields."""
        import backend.app.ml.nlp.nlp_models as nlp
        fake_result = {
            "severity_score": 0.7,
            "confidence": 0.85,
            "model": "distilbert",
            "raw_label": "NEGATIVE",
            "model_available": True,
            "model_loaded": True,
            "model_cached": True,
            "inference_source": "distilbert",
        }
        with mock.patch.object(nlp, "_load_biobert", return_value=False):
            with mock.patch.object(nlp, "_infer_biobert", return_value=None):
                with mock.patch.object(nlp, "_load_distilbert", return_value=True):
                    with mock.patch.object(nlp, "_infer_distilbert", return_value=fake_result):
                        result = nlp.infer_nlp_models("severe chest pain")
                        assert result["model"] == "distilbert"
                        assert result["model_cached"] is True
                        assert "available_models" in result


class TestCompareWithDistilbert:
    def test_no_model_available(self):
        from backend.app.ml.nlp.nlp_models import compare_with_distilbert
        model_result = {"model": "none", "severity_score": None, "confidence": 0.0}
        out = compare_with_distilbert(model_result, 0.6, "moderate", 0.7)
        assert out["source"] == "distilbert_keywords"
        assert out["model_cached"] is False

    def test_model_wins(self):
        from backend.app.ml.nlp.nlp_models import compare_with_distilbert
        model_result = {
            "model": "biobert", "severity_score": 0.8, "confidence": 0.9,
            "available_models": ["biobert"],
        }
        out = compare_with_distilbert(model_result, 0.5, "moderate", 0.6)
        assert out["source"] == "biobert"
        assert out["model_cached"] is True
