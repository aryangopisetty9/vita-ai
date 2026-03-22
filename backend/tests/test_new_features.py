"""
Tests for new backend features: model registry, fusion model, validation,
clinical validation, session manager, and the /status endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest



# ═══════════════════════════════════════════════════════════════════════════
# Model Registry Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_import(self):
        from backend.app.ml.registry.model_registry import get_all_status, ModelInfo
        status = get_all_status()
        assert isinstance(status, dict)

    def test_model_info_fields(self):
        from backend.app.ml.registry.model_registry import ModelInfo
        mi = ModelInfo(name="test", category="rppg", env_var="TEST_VAR")
        assert mi.name == "test"
        assert mi.available is False
        assert mi.loaded is False

    def test_is_available_without_env(self):
        from backend.app.ml.registry.model_registry import is_available
        # With no env vars pointing to real files, all should be unavailable
        result = is_available("open_rppg")
        assert isinstance(result, bool)

    def test_get_models_by_category(self):
        from backend.app.ml.registry.model_registry import get_models_by_category
        rppg = get_models_by_category("rppg")
        assert isinstance(rppg, list)


# ═══════════════════════════════════════════════════════════════════════════
# Fusion Model Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFusionModel:
    def test_predict_with_auto_train(self):
        from backend.app.ml.fusion.fusion_model import predict_score
        result = predict_score(80.0, 70.0, 60.0, [0.8, 0.7, 0.6])
        # Fusion model auto-trains from synthetic data if no model file exists
        assert isinstance(result, float)
        assert 0 <= result <= 100

    def test_get_status(self):
        from backend.app.ml.fusion.fusion_model import get_fusion_status
        status = get_fusion_status()
        assert isinstance(status, dict)
        assert "available" in status
        assert "method" in status


# ═══════════════════════════════════════════════════════════════════════════
# Validation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestValidation:
    def test_video_valid_ext(self):
        from backend.app.core.validation import validate_video_upload
        from unittest.mock import MagicMock
        f = MagicMock()
        f.filename = "test.mp4"
        f.size = 1024
        assert validate_video_upload(f) is None

    def test_video_bad_ext(self):
        from backend.app.core.validation import validate_video_upload
        from unittest.mock import MagicMock
        f = MagicMock()
        f.filename = "test.txt"
        f.size = 1024
        err = validate_video_upload(f)
        assert err is not None
        assert "Unsupported" in err

    def test_audio_valid_ext(self):
        from backend.app.core.validation import validate_audio_upload
        from unittest.mock import MagicMock
        f = MagicMock()
        f.filename = "breath.wav"
        f.size = 1024
        assert validate_audio_upload(f) is None

    def test_audio_bad_ext(self):
        from backend.app.core.validation import validate_audio_upload
        from unittest.mock import MagicMock
        f = MagicMock()
        f.filename = "file.exe"
        f.size = 1024
        err = validate_audio_upload(f)
        assert err is not None

    def test_symptom_valid(self):
        from backend.app.core.validation import validate_symptom_text
        assert validate_symptom_text("I have a headache") is None

    def test_symptom_empty(self):
        from backend.app.core.validation import validate_symptom_text
        err = validate_symptom_text("")
        assert err is not None
        assert "empty" in err.lower()

    def test_symptom_too_long(self):
        from backend.app.core.validation import validate_symptom_text, MAX_TEXT_LENGTH
        err = validate_symptom_text("x" * (MAX_TEXT_LENGTH + 1))
        assert err is not None
        assert "long" in err.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Clinical Validation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestClinicalValidation:
    def test_emergency_detection(self):
        from backend.app.core.clinical_validation import check_symptom_severity
        result = check_symptom_severity(
            ["chest pain", "breathlessness"], "moderate", 0.7,
        )
        assert result["emergency_flag"] is True
        assert result["escalated_risk"] == "high"

    def test_no_escalation(self):
        from backend.app.core.clinical_validation import check_symptom_severity
        result = check_symptom_severity(
            ["mild headache"], "low", 0.6,
        )
        assert result["emergency_flag"] is False
        assert result["escalation_flag"] is False
        assert result["escalated_risk"] == "low"

    def test_consistency_no_modules(self):
        from backend.app.core.clinical_validation import check_cross_module_consistency
        result = check_cross_module_consistency()
        assert result["consistency_score"] == 1.0
        assert result["low_reliability_warning"] is False

    def test_consistency_contradiction(self):
        from backend.app.core.clinical_validation import check_cross_module_consistency
        result = check_cross_module_consistency(
            face_result={"risk": "high", "scan_quality": 0.8, "confidence": 0.7},
            symptom_result={"risk": "low", "confidence": 0.6},
        )
        assert result["consistency_score"] < 1.0
        assert len(result["consistency_warnings"]) > 0

    def test_calibrate_confidence(self):
        from backend.app.core.clinical_validation import calibrate_confidence
        # Good quality, deep model source → slight boost
        cal = calibrate_confidence(0.7, scan_quality=0.9, model_source="open_rppg")
        assert 0.0 <= cal <= 1.0
        # Low quality → lower confidence
        cal_low = calibrate_confidence(0.7, scan_quality=0.2)
        assert cal_low < cal

    def test_generate_clinical_notes(self):
        from backend.app.core.clinical_validation import generate_clinical_notes
        notes = generate_clinical_notes(
            "high",
            {"consistency_score": 0.7, "low_reliability_warning": True, "consistency_warnings": []},
        )
        assert "disclaimer" in notes
        assert "clinical_safety_note" in notes


# ═══════════════════════════════════════════════════════════════════════════
# Session Manager Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionManager:
    def test_create_and_close(self):
        from backend.app.services.session_manager import SessionManager
        mgr = SessionManager()
        s = mgr.create_session("test-client")
        assert s is not None
        assert s.session_id
        mgr.close_session(s.session_id)
        assert mgr.get_session(s.session_id) is None

    def test_record_frame(self):
        from backend.app.services.session_manager import SessionManager
        mgr = SessionManager()
        s = mgr.create_session()
        assert s.frames_received == 0
        mgr.record_frame(s.session_id)
        assert s.frames_received == 1
        mgr.close_session(s.session_id)

    def test_status(self):
        from backend.app.services.session_manager import SessionManager
        mgr = SessionManager()
        status = mgr.get_status()
        assert "active_sessions" in status
        assert "max_sessions" in status


# ═══════════════════════════════════════════════════════════════════════════
# Classification Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestClassificationMetrics:
    def test_accuracy(self):
        from backend.eval.metrics import classification_accuracy
        acc = classification_accuracy(
            ["low", "high", "moderate"], ["low", "high", "high"],
        )
        assert abs(acc - 2 / 3) < 0.01

    def test_report(self):
        from backend.eval.metrics import classification_report
        report = classification_report(
            ["low", "high", "moderate", "low"],
            ["low", "high", "high", "low"],
            labels=["low", "moderate", "high"],
        )
        assert "accuracy" in report
        assert "macro_avg" in report
        assert report["n_samples"] == 4


# ═══════════════════════════════════════════════════════════════════════════
# API Endpoint Tests (using TestClient)
# ═══════════════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def _client(self):
        from fastapi.testclient import TestClient
        from backend.app.api.main import app
        self.client = TestClient(app)

    def test_root(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "WS   /ws/audio-stream" in data["endpoints"]

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_status(self):
        resp = self.client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "fusion" in data
        assert "streaming" in data

    def test_models(self):
        resp = self.client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "rppg_models" in data
        assert "fusion_detail" in data


# ═══════════════════════════════════════════════════════════════════════════
# Score Engine Integration (with clinical validation)
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreEngineIntegration:
    def test_score_with_clinical_notes(self):
        from backend.app.ml.fusion.score_engine import compute_vita_score
        result = compute_vita_score(
            face_result={"heart_rate": 72, "risk": "low", "confidence": 0.8, "scan_quality": 0.9},
            symptom_result={"risk": "low", "confidence": 0.7, "detected_symptoms": ["headache"]},
        )
        assert "consistency" in result
        assert "clinical_notes" in result
        assert "fusion_method" in result
        assert result["clinical_notes"]["disclaimer"]

    def test_emergency_escalation_in_score(self):
        from backend.app.ml.fusion.score_engine import compute_vita_score
        result = compute_vita_score(
            symptom_result={
                "risk": "moderate",
                "confidence": 0.7,
                "detected_symptoms": ["chest pain", "breathlessness"],
            },
        )
        assert result["overall_risk"] == "high"
