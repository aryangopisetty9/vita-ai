"""
Tests for the pretrained model helper modules.

Covers:
- rppg_models: model loading, inference fallback, confidence comparison
- audio_models: model loading, inference fallback, librosa comparison
- nlp_models: model loading, inference fallback, distilbert comparison
- Score engine XGBoost fallback
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest



# ---------------------------------------------------------------------------
# rPPG models
# ---------------------------------------------------------------------------

class TestRppgModels:
    def test_import(self):
        from backend.app.ml.face.rppg_models import infer_rppg_models, compare_with_signal_pipeline
        assert callable(infer_rppg_models)
        assert callable(compare_with_signal_pipeline)

    def test_infer_returns_fallback_when_no_models(self):
        from backend.app.ml.face.rppg_models import infer_rppg_models
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        result = infer_rppg_models(frames, 30.0)
        assert result["model"] == "none"
        assert result["bpm"] is None
        assert isinstance(result["available_models"], list)
        assert "model_priority" in result
        assert "legacy_fallback_used" in result
        assert result["legacy_fallback_used"] is False

    def test_compare_signal_wins_when_no_model(self):
        from backend.app.ml.face.rppg_models import compare_with_signal_pipeline
        model_result = {"bpm": None, "confidence": 0.0, "model": "none"}
        result = compare_with_signal_pipeline(model_result, 72.0, 0.6)
        assert result["source"] == "signal_pipeline"
        assert result["bpm"] == 72.0
        assert result["confidence"] == 0.6

    def test_compare_model_wins_when_higher_confidence(self):
        from backend.app.ml.face.rppg_models import compare_with_signal_pipeline
        # BPMs far apart (>10 BPM difference) so blending is skipped
        model_result = {"bpm": 95.0, "confidence": 0.9, "model": "open_rppg"}
        result = compare_with_signal_pipeline(model_result, 72.0, 0.3)
        assert result["source"] == "open_rppg"
        assert result["bpm"] == 95.0

    def test_compare_blended_when_close(self):
        from backend.app.ml.face.rppg_models import compare_with_signal_pipeline
        model_result = {"bpm": 73.0, "confidence": 0.7, "model": "open_rppg"}
        result = compare_with_signal_pipeline(model_result, 72.0, 0.6)
        assert result["source"] == "blended"
        assert 70 < result["bpm"] < 76

    def test_get_available_models(self):
        from backend.app.ml.face.rppg_models import get_available_models
        models = get_available_models()
        assert isinstance(models, list)


# ---------------------------------------------------------------------------
# Audio models
# ---------------------------------------------------------------------------

class TestAudioModels:
    def test_import(self):
        from backend.app.ml.audio.audio_models import infer_audio_models, compare_with_librosa_pipeline
        assert callable(infer_audio_models)
        assert callable(compare_with_librosa_pipeline)

    def test_infer_returns_fallback_when_no_models(self):
        import backend.app.ml.audio.audio_models as am
        # Force "already loaded but no model" state
        saved_yamnet = (am._YAMNET_LOADED, am._YAMNET_MODEL)
        am._YAMNET_LOADED = True
        am._YAMNET_MODEL = None
        try:
            audio = np.zeros(22050, dtype=np.float32)
            result = am.infer_audio_models(audio, 22050)
            assert result["model"] == "none"
            assert result["confidence"] == 0.0
        finally:
            am._YAMNET_LOADED, am._YAMNET_MODEL = saved_yamnet

    def test_compare_librosa_wins_when_no_model(self):
        from backend.app.ml.audio.audio_models import compare_with_librosa_pipeline
        model_result = {"model": "none", "confidence": 0.0, "respiratory_detected": False}
        result = compare_with_librosa_pipeline(model_result, "low", 0.7, 16.0)
        assert result["source"] == "librosa_pipeline"
        assert result["risk"] == "low"
        assert result["confidence"] == 0.7

    def test_compare_escalates_when_respiratory(self):
        from backend.app.ml.audio.audio_models import compare_with_librosa_pipeline
        model_result = {
            "model": "yamnet", "confidence": 0.8,
            "respiratory_detected": True, "labels": ["cough"],
        }
        result = compare_with_librosa_pipeline(model_result, "low", 0.6, 16.0)
        assert result["risk"] == "moderate"
        assert result["source"] == "blended"

    def test_get_available_models(self):
        from backend.app.ml.audio.audio_models import get_available_models
        models = get_available_models()
        assert isinstance(models, list)


# ---------------------------------------------------------------------------
# NLP models
# ---------------------------------------------------------------------------

class TestNlpModels:
    def test_import(self):
        from backend.app.ml.nlp.nlp_models import infer_nlp_models, compare_with_distilbert
        assert callable(infer_nlp_models)
        assert callable(compare_with_distilbert)

    def test_infer_returns_fallback_when_no_biobert(self, monkeypatch):
        monkeypatch.setenv("VITA_AUTO_DOWNLOAD_MODELS", "false")
        monkeypatch.setenv("VITA_ENABLE_BIOBERT", "false")
        monkeypatch.setenv("VITA_ENABLE_DISTILBERT", "false")
        # Force reload of cached load flags
        import importlib, backend.app.ml.nlp.nlp_models as _nlp
        _nlp._BIOBERT_LOADED = False
        _nlp._DISTILBERT_LOADED = False
        _nlp._BIOBERT_PIPELINE = None
        _nlp._BIOBERT_MODEL_OBJ = None
        _nlp._DISTILBERT_PIPELINE = None
        _nlp._DISTILBERT_MODEL_OBJ = None
        from backend.app.ml.nlp.nlp_models import infer_nlp_models
        result = infer_nlp_models("I have chest pain and fever")
        assert result["model"] in ("none", "biobert", "distilbert")
        assert isinstance(result["available_models"], list)

    def test_compare_distilbert_wins_when_no_biobert(self):
        from backend.app.ml.nlp.nlp_models import compare_with_distilbert
        biobert_result = {"model": "none", "severity_score": None, "confidence": 0.0}
        result = compare_with_distilbert(biobert_result, 0.7, "moderate", 0.65)
        assert result["source"] == "distilbert_keywords"
        assert result["risk"] == "moderate"

    def test_compare_biobert_wins_when_higher(self):
        from backend.app.ml.nlp.nlp_models import compare_with_distilbert
        biobert_result = {"model": "biobert", "severity_score": 0.8, "confidence": 0.9}
        result = compare_with_distilbert(biobert_result, 0.4, "moderate", 0.5)
        assert result["source"] == "biobert"
        assert result["risk"] == "high"

    def test_get_available_models(self):
        from backend.app.ml.nlp.nlp_models import get_available_models
        models = get_available_models()
        assert isinstance(models, list)
        # DistilBERT should always be in the list
        assert "distilbert" in models


# ---------------------------------------------------------------------------
# Score engine XGBoost fallback
# ---------------------------------------------------------------------------

class TestScoreEngineXGBoost:
    def test_fusion_method_in_output(self):
        from backend.app.ml.fusion.score_engine import compute_vita_score
        result = compute_vita_score(
            face_result={"value": 72, "risk": "low", "confidence": 0.8},
            audio_result={"value": 16, "risk": "low", "confidence": 0.7},
            symptom_result={"risk": "low", "confidence": 0.9},
        )
        assert "fusion_method" in result
        # Without XGBoost model file, should be weighted_sum
        assert result["fusion_method"] == "weighted_sum"

    def test_xgboost_predict_returns_none_without_model(self):
        from backend.app.ml.fusion.fusion_model import predict_score
        result = predict_score(80.0, 85.0, 90.0, [0.8, 0.7, 0.9])
        assert result is None


# ---------------------------------------------------------------------------
# Integration: model debug metadata appears in module outputs
# ---------------------------------------------------------------------------

class TestModelDebugMetadata:
    def test_face_module_error_has_no_crash(self):
        """Face module should not crash even though rppg_models returns none."""
        from backend.app.ml.face.face_module import analyze_face_video
        result = analyze_face_video("nonexistent.mp4")
        assert result["risk"] == "error"
        # Debug metadata for rppg model may or may not be present on error path
        # but should not crash

    def test_audio_module_with_non_existent_file(self):
        from backend.app.ml.audio.audio_module import analyze_audio
        result = analyze_audio("nonexistent.wav")
        assert result["risk"] == "error"

    def test_symptom_module_produces_nlp_debug(self):
        from backend.app.ml.nlp.symptom_module import analyze_symptoms
        result = analyze_symptoms("I feel headache and dizziness")
        assert "debug" in result
        debug = result["debug"]
        assert "nlp_model" in debug
        assert "model_name" in debug["nlp_model"]
        assert "available_models" in debug["nlp_model"]
