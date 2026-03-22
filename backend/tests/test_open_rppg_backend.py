"""
Tests for the Open-rPPG backend wrapper (models/open_rppg_backend.py).

Covers:
- Availability check when package is / is not installed
- Singleton model loading & idempotency
- Inference with dummy frames (fallback when too few frames)
- Fallback result structure
- Status dict shape
- Safe HRV sanitisation
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest



# ═══════════════════════════════════════════════════════════════════════════
# Availability
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenRppgAvailability:
    def test_is_open_rppg_available_returns_bool(self):
        from backend.app.ml.face.open_rppg_backend import is_open_rppg_available
        assert isinstance(is_open_rppg_available(), bool)

    def test_get_supported_models_returns_list(self):
        from backend.app.ml.face.open_rppg_backend import get_supported_models
        models = get_supported_models()
        assert isinstance(models, list)

    def test_supported_models_not_empty_when_installed(self):
        from backend.app.ml.face.open_rppg_backend import is_open_rppg_available, get_supported_models
        if is_open_rppg_available():
            assert len(get_supported_models()) > 0
        else:
            pytest.skip("open-rppg not installed")


# ═══════════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenRppgStatus:
    def test_status_dict_shape(self):
        from backend.app.ml.face.open_rppg_backend import get_open_rppg_status
        status = get_open_rppg_status()
        assert isinstance(status, dict)
        for key in ("installed", "loaded", "active", "model_name",
                     "supported_models", "error"):
            assert key in status, f"Missing status key: {key}"
        assert isinstance(status["installed"], bool)
        assert isinstance(status["loaded"], bool)
        assert isinstance(status["active"], bool)
        assert isinstance(status["supported_models"], list)

    def test_status_installed_matches_availability(self):
        from backend.app.ml.face.open_rppg_backend import get_open_rppg_status, is_open_rppg_available
        status = get_open_rppg_status()
        assert status["installed"] == is_open_rppg_available()


# ═══════════════════════════════════════════════════════════════════════════
# Fallback result
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackResult:
    def test_fallback_result_keys(self):
        from backend.app.ml.face.open_rppg_backend import _fallback_result
        result = _fallback_result("test reason")
        assert result["bpm"] is None
        assert result["confidence"] == 0.0
        assert result["model"] == "none"
        assert result["open_rppg_active"] is False
        assert result["classical_fallback_used"] is True
        assert result["fallback_reason"] == "test reason"

    def test_fallback_with_partial_data(self):
        from backend.app.ml.face.open_rppg_backend import _fallback_result
        result = _fallback_result("reason", partial_data={"sqi": 0.5})
        assert result["sqi"] == 0.5
        assert result["bpm"] is None


# ═══════════════════════════════════════════════════════════════════════════
# HRV sanitisation
# ═══════════════════════════════════════════════════════════════════════════

class TestSafeHrv:
    def test_handles_nan_and_inf(self):
        from backend.app.ml.face.open_rppg_backend import _safe_hrv
        hrv = {"rmssd": float("nan"), "sdnn": float("inf"), "valid": 42.0}
        cleaned = _safe_hrv(hrv)
        assert cleaned["rmssd"] is None
        assert cleaned["sdnn"] is None
        assert cleaned["valid"] == 42.0

    def test_handles_numpy_arrays(self):
        from backend.app.ml.face.open_rppg_backend import _safe_hrv
        hrv = {"rr": np.array([0.8, 0.9, float("nan")])}
        cleaned = _safe_hrv(hrv)
        assert isinstance(cleaned["rr"], list)
        assert cleaned["rr"][2] is None

    def test_handles_non_dict(self):
        from backend.app.ml.face.open_rppg_backend import _safe_hrv
        assert _safe_hrv(None) == {}
        assert _safe_hrv("string") == {}


# ═══════════════════════════════════════════════════════════════════════════
# Inference fallback paths
# ═══════════════════════════════════════════════════════════════════════════

class TestInferOpenRppg:
    def test_too_few_frames_returns_fallback(self):
        from backend.app.ml.face.open_rppg_backend import infer_open_rppg
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5
        result = infer_open_rppg(frames, 30.0)
        assert result["bpm"] is None
        assert result["classical_fallback_used"] is True
        # May be "Too few frames" or "model not loaded" depending on state
        assert "fallback_reason" in result

    def test_empty_frames_returns_fallback(self):
        from backend.app.ml.face.open_rppg_backend import infer_open_rppg
        result = infer_open_rppg([], 30.0)
        assert result["bpm"] is None
        assert result["classical_fallback_used"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadOpenRppg:
    def test_load_returns_bool(self):
        from backend.app.ml.face.open_rppg_backend import load_open_rppg, is_open_rppg_available
        if not is_open_rppg_available():
            result = load_open_rppg()
            assert result is False
        else:
            # If installed, just check it returns a bool
            result = load_open_rppg()
            assert isinstance(result, bool)

    def test_load_when_not_installed_returns_false(self):
        """Simulate open-rppg not being installed."""
        import backend.app.ml.face.open_rppg_backend as backend
        saved = backend._HAS_OPEN_RPPG
        try:
            backend._HAS_OPEN_RPPG = False
            assert backend.load_open_rppg() is False
        finally:
            backend._HAS_OPEN_RPPG = saved
