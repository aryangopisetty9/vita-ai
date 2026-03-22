"""
Tests for full model integration: cache, download, loading, inference,
registry, status, health endpoint, and startup config.

All model downloads are mocked to avoid real network I/O.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest



# ═══════════════════════════════════════════════════════════════════════════
# Cache folder creation
# ═══════════════════════════════════════════════════════════════════════════

class TestCacheFolders:
    def test_ensure_cache_dirs_creates_all(self, tmp_path: Path):
        dirs = {
            "MODELS_CACHE_DIR": tmp_path,
            "DISTILBERT_CACHE_DIR": tmp_path / "distilbert",
            "BIOBERT_CACHE_DIR": tmp_path / "biobert",
            "YAMNET_CACHE_DIR": tmp_path / "yamnet_saved_model",
        }
        with mock.patch.multiple("backend.app.ml.registry.model_paths", **dirs):
            from backend.app.ml.registry.model_paths import ensure_cache_dirs
            ensure_cache_dirs()
            for d in dirs.values():
                assert d.exists()

    def test_is_model_cached_false_empty_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "distilbert"
        cache_dir.mkdir()
        with mock.patch("backend.app.ml.registry.model_paths.DISTILBERT_CACHE_DIR", cache_dir):
            from backend.app.ml.registry.model_paths import is_model_cached
            assert is_model_cached("distilbert") is False

    def test_is_model_cached_true_with_files(self, tmp_path: Path):
        cache_dir = tmp_path / "biobert"
        cache_dir.mkdir()
        (cache_dir / "config.json").write_text("{}")
        (cache_dir / "model.safetensors").write_text("fake")
        with mock.patch("backend.app.ml.registry.model_paths.BIOBERT_CACHE_DIR", cache_dir):
            from backend.app.ml.registry.model_paths import is_model_cached
            assert is_model_cached("biobert") is True


# ═══════════════════════════════════════════════════════════════════════════
# DistilBERT loading tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDistilBERTLoading:
    def test_load_disabled(self, monkeypatch):
        monkeypatch.setenv("VITA_ENABLE_DISTILBERT", "false")
        import backend.app.ml.nlp.nlp_models as nlp
        nlp._DISTILBERT_LOADED = False
        nlp._DISTILBERT_PIPELINE = None
        nlp._DISTILBERT_MODEL_OBJ = None
        with mock.patch("backend.app.ml.registry.model_download.is_distilbert_enabled", return_value=False):
            result = nlp._load_distilbert()
            assert result is False

    def test_load_returns_false_without_transformers(self):
        import backend.app.ml.nlp.nlp_models as nlp
        nlp._DISTILBERT_LOADED = False
        nlp._DISTILBERT_PIPELINE = None
        with mock.patch.object(nlp, "_HAS_TRANSFORMERS", False):
            result = nlp._load_distilbert()
            assert result is False

    def test_distilbert_inference_with_mock(self):
        import backend.app.ml.nlp.nlp_models as nlp
        fake_pipeline = mock.MagicMock()
        fake_pipeline.return_value = [{"label": "NEGATIVE", "score": 0.85}]
        nlp._DISTILBERT_PIPELINE = fake_pipeline
        result = nlp._infer_distilbert("I have chest pain")
        assert result is not None
        assert result["model"] == "distilbert"
        assert result["inference_source"] == "distilbert"
        assert result["model_loaded"] is True
        assert "severity_score" in result
        nlp._DISTILBERT_PIPELINE = None


# ═══════════════════════════════════════════════════════════════════════════
# BioBERT loading tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBioBERTLoading:
    def test_load_disabled(self, monkeypatch):
        monkeypatch.setenv("VITA_ENABLE_BIOBERT", "false")
        import backend.app.ml.nlp.nlp_models as nlp
        nlp._BIOBERT_LOADED = False
        nlp._BIOBERT_PIPELINE = None
        nlp._BIOBERT_MODEL_OBJ = None
        with mock.patch("backend.app.ml.registry.model_download.is_biobert_enabled", return_value=False):
            result = nlp._load_biobert()
            assert result is False

    def test_load_returns_false_without_transformers(self):
        import backend.app.ml.nlp.nlp_models as nlp
        nlp._BIOBERT_LOADED = False
        nlp._BIOBERT_PIPELINE = None
        nlp._BIOBERT_MODEL_OBJ = None
        with mock.patch.object(nlp, "_HAS_TRANSFORMERS", False):
            result = nlp._load_biobert()
            assert result is False

    def test_biobert_inference_returns_none_without_model(self):
        import backend.app.ml.nlp.nlp_models as nlp
        nlp._BIOBERT_MODEL_OBJ = None
        nlp._BIOBERT_TOKENIZER = None
        result = nlp._infer_biobert("test text")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# BioBERT → DistilBERT fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestNLPFallbackChain:
    def test_biobert_fallback_to_distilbert(self):
        import backend.app.ml.nlp.nlp_models as nlp
        fake_distilbert_result = {
            "severity_score": 0.6,
            "confidence": 0.75,
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
                    with mock.patch.object(nlp, "_infer_distilbert", return_value=fake_distilbert_result):
                        result = nlp.infer_nlp_models("stomach pain")
                        assert result["model"] == "distilbert"
                        assert result["inference_source"] == "distilbert"

    def test_distilbert_fallback_to_rules(self):
        import backend.app.ml.nlp.nlp_models as nlp
        with mock.patch.object(nlp, "_load_biobert", return_value=False):
            with mock.patch.object(nlp, "_infer_biobert", return_value=None):
                with mock.patch.object(nlp, "_load_distilbert", return_value=False):
                    with mock.patch.object(nlp, "_infer_distilbert", return_value=None):
                        result = nlp.infer_nlp_models("headache")
                        assert result["model"] == "none"
                        assert result["inference_source"] == "keyword_rules"
                        assert result["model_available"] is False


# ═══════════════════════════════════════════════════════════════════════════
# YAMNet loading / fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestYAMNetLoading:
    def test_load_disabled(self, monkeypatch):
        monkeypatch.setenv("VITA_ENABLE_YAMNET", "false")
        import backend.app.ml.audio.audio_models as am
        am._YAMNET_LOADED = False
        am._YAMNET_MODEL = None
        with mock.patch("backend.app.ml.registry.model_download.is_yamnet_enabled", return_value=False):
            result = am._load_yamnet()
            assert result is False

    def test_yamnet_fallback_to_librosa(self):
        import backend.app.ml.audio.audio_models as am
        # Reset loaded flags and model objects
        am._YAMNET_LOADED = False
        am._YAMNET_MODEL = None

        # Mock the entire _MODEL_CHAIN to ensure loaders return False
        fake_chain = [
            ("yamnet", lambda: False, lambda a, sr: None),
        ]
        with mock.patch.object(am, "_MODEL_CHAIN", fake_chain):
            audio = np.zeros(16000, dtype=np.float32)
            result = am.infer_audio_models(audio, 16000)
            assert result["model"] == "none"
            assert result["inference_source"] == "librosa_pipeline"

    def test_yamnet_inference_with_mock(self):
        import backend.app.ml.audio.audio_models as am
        fake_result = {
            "labels": ["Speech", "Music"],
            "top_label": "Speech",
            "confidence": 0.75,
            "respiratory_detected": False,
            "health_concern": False,
            "model": "yamnet",
            "model_available": True,
            "model_loaded": True,
            "model_cached": True,
            "inference_source": "yamnet",
        }

        def fake_loader():
            return True

        def fake_infer(a, sr):
            return fake_result

        fake_chain = [
            ("yamnet", fake_loader, fake_infer),
        ]
        with mock.patch.object(am, "_MODEL_CHAIN", fake_chain):
            audio = np.zeros(16000, dtype=np.float32)
            result = am.infer_audio_models(audio, 16000)
            assert result["model"] == "yamnet"
            assert result["inference_source"] == "yamnet"


# ═══════════════════════════════════════════════════════════════════════════
# Model status reporting
# ═══════════════════════════════════════════════════════════════════════════

class TestModelStatusReporting:
    def test_all_model_status_keys(self):
        from backend.app.ml.registry.model_status import get_all_model_status, ALL_MODELS
        status = get_all_model_status()
        for model_name in ALL_MODELS:
            assert model_name in status
            assert "cached" in status[model_name]
            assert "loaded" in status[model_name]
            assert "active" in status[model_name]

    def test_manual_models_have_note(self):
        from backend.app.ml.registry.model_status import get_model_status, MANUAL_MODELS
        for model_name in MANUAL_MODELS:
            status = get_model_status(model_name)
            assert "note" in status
            # Note gives actionable info: either "installed" or instructions
            note_lower = status["note"].lower()
            assert any(word in note_lower for word in ("manual", "weights", "train", "installed", "disabled"))

    def test_mark_and_clear_error(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, get_model_status
        mark_model_loaded("distilbert", loaded=False, active=False, error="test error")
        status = get_model_status("distilbert")
        assert status["error"] == "test error"
        mark_model_loaded("distilbert", loaded=True, active=True)
        status = get_model_status("distilbert")
        assert "error" not in status

    def test_list_active_models_returns_only_active(self):
        from backend.app.ml.registry.model_status import mark_model_loaded, list_active_models
        mark_model_loaded("biobert", loaded=True, active=True)
        mark_model_loaded("yamnet", loaded=True, active=False)
        active = list_active_models()
        assert "biobert" in active
        assert "yamnet" not in active


# ═══════════════════════════════════════════════════════════════════════════
# Startup auto-download config logic
# ═══════════════════════════════════════════════════════════════════════════

class TestStartupConfig:
    def test_auto_download_default_true(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITA_AUTO_DOWNLOAD_MODELS", None)
            from backend.app.ml.registry.model_download import _env_flag
            assert _env_flag("VITA_AUTO_DOWNLOAD_MODELS", True) is True

    def test_auto_download_disabled(self):
        with mock.patch.dict(os.environ, {"VITA_AUTO_DOWNLOAD_MODELS": "false"}):
            from backend.app.ml.registry.model_download import _env_flag
            assert _env_flag("VITA_AUTO_DOWNLOAD_MODELS", True) is False

    def test_preload_default_false(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITA_PRELOAD_MODELS", None)
            from backend.app.ml.registry.model_download import _env_flag
            assert _env_flag("VITA_PRELOAD_MODELS", False) is False

    def test_enable_flags(self):
        with mock.patch.dict(os.environ, {
            "VITA_ENABLE_BIOBERT": "true",
            "VITA_ENABLE_DISTILBERT": "true",
            "VITA_ENABLE_YAMNET": "true",
        }):
            from backend.app.ml.registry.model_download import is_biobert_enabled, is_distilbert_enabled, is_yamnet_enabled
            assert is_biobert_enabled() is True
            assert is_distilbert_enabled() is True
            assert is_yamnet_enabled() is True

    def test_disable_flags(self):
        with mock.patch.dict(os.environ, {
            "VITA_ENABLE_BIOBERT": "false",
            "VITA_ENABLE_DISTILBERT": "0",
            "VITA_ENABLE_YAMNET": "no",
        }):
            from backend.app.ml.registry.model_download import is_biobert_enabled, is_distilbert_enabled, is_yamnet_enabled
            assert is_biobert_enabled() is False
            assert is_distilbert_enabled() is False
            assert is_yamnet_enabled() is False


# ═══════════════════════════════════════════════════════════════════════════
# Health endpoint correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    @pytest.fixture(autouse=True)
    def _client(self):
        # Disable auto-download and preload during tests to avoid slow model loading
        with mock.patch.dict(os.environ, {
            "VITA_AUTO_DOWNLOAD_MODELS": "false",
            "VITA_PRELOAD_MODELS": "false",
        }):
            from fastapi.testclient import TestClient
            from backend.app.api.main import app
            self.client = TestClient(app)
            yield

    def test_health_returns_ok(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_models_key(self):
        resp = self.client.get("/health")
        data = resp.json()
        assert "models" in data

    def test_health_model_status_fields(self):
        resp = self.client.get("/health")
        data = resp.json()
        models = data.get("models", {})
        for model_name, status in models.items():
            assert "cached" in status
            assert "loaded" in status
            assert "active" in status

    def test_status_endpoint_has_all_sections(self):
        resp = self.client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "runtime_model_status" in data
        assert "fusion" in data
        assert "streaming" in data

    def test_models_endpoint(self):
        resp = self.client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "nlp_models" in data
        assert "audio_models" in data
        assert "rppg_models" in data


# ═══════════════════════════════════════════════════════════════════════════
# Model registry helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestModelRegistryHelpers:
    @mock.patch("backend.app.ml.registry.model_download.download_all_supported", return_value={"distilbert": True, "biobert": True, "yamnet": True})
    def test_ensure_supported_models_downloaded(self, mock_dl):
        from backend.app.ml.registry.model_registry import ensure_supported_models_downloaded
        results = ensure_supported_models_downloaded()
        assert isinstance(results, dict)
        mock_dl.assert_called_once()

    def test_load_supported_models_returns_dict(self):
        from backend.app.ml.registry.model_registry import load_supported_models
        with mock.patch("backend.app.ml.nlp.nlp_models.get_available_models", return_value=[]):
            with mock.patch("backend.app.ml.audio.audio_models.get_available_models", return_value=[]):
                loaded = load_supported_models()
                assert isinstance(loaded, dict)
                assert "biobert" in loaded
                assert "distilbert" in loaded
                assert "yamnet" in loaded

    def test_list_active_models_returns_list(self):
        from backend.app.ml.registry.model_registry import list_active_models
        active = list_active_models()
        assert isinstance(active, list)

    def test_refresh_registry(self):
        from backend.app.ml.registry.model_registry import refresh_registry, get_all_status
        refresh_registry()
        status = get_all_status()
        assert isinstance(status, dict)
        assert len(status) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Symptom module NLP integration debug fields
# ═══════════════════════════════════════════════════════════════════════════

class TestSymptomNLPDebug:
    def test_symptom_result_has_nlp_debug(self):
        import backend.app.ml.nlp.symptom_module as sm
        import backend.app.ml.nlp.nlp_models as nlp
        fake_nlp_result = {
            "severity_score": 0.5, "confidence": 0.6,
            "model": "none", "available_models": [],
            "model_available": False, "model_loaded": False,
            "model_cached": False, "inference_source": "keyword_rules",
        }
        # Mock both the NLP layer AND the _ensure_pipeline to avoid real model loads
        with mock.patch.object(sm, "infer_nlp_models", return_value=fake_nlp_result):
            sm._pipeline_loaded = True
            sm._pipeline = None
            sm._MODEL_NAME = "keyword-rules-only"
            result = sm.analyze_symptoms("I have a severe headache and dizziness")
            assert "debug" in result
            debug = result["debug"]
            assert "nlp_model" in debug
            nlp_debug = debug["nlp_model"]
            assert "model_name" in nlp_debug
            assert "inference_source" in nlp_debug
            assert "model_available" in nlp_debug
            assert "model_loaded" in nlp_debug
            assert "model_cached" in nlp_debug

    def test_symptom_result_has_model_used(self):
        import backend.app.ml.nlp.symptom_module as sm
        import backend.app.ml.nlp.nlp_models as nlp
        fake_nlp_result = {
            "severity_score": 0.5, "confidence": 0.6,
            "model": "none", "available_models": [],
            "model_available": False, "model_loaded": False,
            "model_cached": False, "inference_source": "keyword_rules",
        }
        with mock.patch.object(sm, "infer_nlp_models", return_value=fake_nlp_result):
            sm._pipeline_loaded = True
            sm._pipeline = None
            sm._MODEL_NAME = "keyword-rules-only"
            result = sm.analyze_symptoms("chest pain and fever")
            debug = result.get("debug", {})
            assert "model_used" in debug


# ═══════════════════════════════════════════════════════════════════════════
# Audio module model debug fields
# ═══════════════════════════════════════════════════════════════════════════

class TestAudioModuleDebug:
    def test_audio_result_has_model_debug(self, tmp_path: Path):
        """Audio module result should include model debug info when processing succeeds."""
        import soundfile as sf
        wav_path = tmp_path / "test.wav"
        # Create a synthetic 5-second audio file
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 0.3 * t)  # ~18 breaths/min
        sf.write(str(wav_path), audio, sr)

        # Mock at the import target in audio_module (where it's called from)
        fake_model_result = {
            "labels": [], "top_label": None, "confidence": 0.0,
            "respiratory_detected": False, "model": "none",
            "available_models": [],
            "model_available": False, "model_loaded": False,
            "model_cached": False,
            "inference_source": "librosa_pipeline",
        }
        import backend.app.ml.audio.audio_module as audio_mod
        with mock.patch.object(audio_mod, "infer_audio_models", return_value=fake_model_result):
            result = audio_mod.analyze_audio(str(wav_path))
            assert "debug" in result
            debug = result["debug"]
            assert "audio_model" in debug
            audio_model = debug["audio_model"]
            assert "model_used" in audio_model
            assert "yamnet_available" in audio_model
            assert "yamnet_loaded" in audio_model
            assert "yamnet_cached" in audio_model
            assert "inference_source" in audio_model
