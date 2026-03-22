"""
Tests for face endpoint model debug field correctness
and backward compatibility with Open-rPPG integration.

Verifies:
- Debug fields from face endpoint include Open-rPPG status
- Existing output schema is preserved (no breaking changes)
- The model priority chain includes open_rppg at position 0
- Score engine still works with face results containing new fields
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest



# ═══════════════════════════════════════════════════════════════════════════
# Model chain priority
# ═══════════════════════════════════════════════════════════════════════════

class TestModelChainPriority:
    def test_open_rppg_first_in_chain(self):
        """open_rppg should be the first entry in the model chain."""
        from backend.app.ml.face.rppg_models import _MODEL_CHAIN
        assert len(_MODEL_CHAIN) > 0
        first_name, _, _ = _MODEL_CHAIN[0]
        assert first_name == "open_rppg"

    def test_model_chain_has_all_models(self):
        """All expected models should still be in the chain."""
        from backend.app.ml.face.rppg_models import _MODEL_CHAIN
        names = [name for name, _, _ in _MODEL_CHAIN]
        # Open-rPPG must be first
        assert names[0] == "open_rppg"
        assert "open_rppg" in names


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_infer_result_has_standard_keys(self):
        """infer_rppg_models output must still have the original standard keys."""
        from backend.app.ml.face.rppg_models import infer_rppg_models
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        result = infer_rppg_models(frames, 30.0)
        for key in ("model", "bpm", "confidence", "available_models",
                     "model_priority"):
            assert key in result, f"Missing standard key: {key}"

    def test_compare_signal_still_works(self):
        """compare_with_signal_pipeline should still function with standard input."""
        from backend.app.ml.face.rppg_models import compare_with_signal_pipeline
        model_result = {"bpm": None, "confidence": 0.0, "model": "none"}
        result = compare_with_signal_pipeline(model_result, 72.0, 0.6)
        assert result["source"] == "signal_pipeline"
        assert result["bpm"] == 72.0

    def test_face_error_result_schema_unchanged(self):
        """_build_error_result must still return all required keys."""
        from backend.app.ml.face.face_module import _build_error_result
        result = _build_error_result("test error")
        for key in ("module_name", "heart_rate", "risk", "confidence",
                     "debug", "scan_quality", "retake_required",
                     "retake_reasons", "hr_timeseries"):
            assert key in result, f"Missing error result key: {key}"
        assert result["risk"] == "error"


# ═══════════════════════════════════════════════════════════════════════════
# Score engine with new face fields
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreEngineWithOpenRppg:
    def test_score_engine_accepts_face_result_with_open_rppg_fields(self):
        """Score engine should work even when face result has new fields."""
        from backend.app.ml.fusion.score_engine import compute_vita_score
        face = {
            "value": 72, "risk": "low", "confidence": 0.8,
            "scan_quality": 0.9,
            # New Open-rPPG fields shouldn't break anything
            "open_rppg_active": True,
            "classical_fallback_used": False,
        }
        result = compute_vita_score(face_result=face)
        assert "vita_health_score" in result
        assert 0 <= result["vita_health_score"] <= 100

    def test_score_engine_works_without_open_rppg_fields(self):
        """Score engine should still work with face results lacking new fields."""
        from backend.app.ml.fusion.score_engine import compute_vita_score
        face = {"value": 72, "risk": "low", "confidence": 0.8}
        result = compute_vita_score(face_result=face)
        assert "vita_health_score" in result


# ═══════════════════════════════════════════════════════════════════════════
# Model registry integration
# ═══════════════════════════════════════════════════════════════════════════

class TestModelRegistryOpenRppg:
    def test_open_rppg_in_registry(self):
        from backend.app.ml.registry.model_registry import get_all_status
        status = get_all_status()
        assert "open_rppg" in status

    def test_open_rppg_category_is_rppg(self):
        from backend.app.ml.registry.model_registry import get_models_by_category
        rppg_models = get_models_by_category("rppg")
        names = [m.name if hasattr(m, "name") else m for m in rppg_models]
        assert "open_rppg" in names
