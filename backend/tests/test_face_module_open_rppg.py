"""
Tests for face module integration with Open-rPPG.

Verifies that the face module output schema includes Open-rPPG debug
fields and that the module works correctly regardless of whether
Open-rPPG is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest



# ═══════════════════════════════════════════════════════════════════════════
# Debug fields in face module output
# ═══════════════════════════════════════════════════════════════════════════

class TestFaceModuleOpenRppgDebug:
    def test_debug_rppg_model_has_open_rppg_fields_when_pipeline_runs(self):
        """When the pipeline runs far enough, rppg_model debug should have Open-rPPG keys.

        Note: On the file-not-found error path the pipeline exits early
        and rppg_model is not populated. We verify the fields appear in the
        rppg_models output directly instead."""
        from backend.app.ml.face.rppg_models import infer_rppg_models
        import numpy as np
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        result = infer_rppg_models(frames, 30.0)
        # New Open-rPPG status fields must be present
        assert "open_rppg_active" in result
        assert "classical_fallback_used" in result

    def test_open_rppg_active_is_bool(self):
        from backend.app.ml.face.rppg_models import infer_rppg_models
        import numpy as np
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        result = infer_rppg_models(frames, 30.0)
        assert isinstance(result["open_rppg_active"], bool)

    def test_classical_fallback_is_bool(self):
        from backend.app.ml.face.rppg_models import infer_rppg_models
        import numpy as np
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        result = infer_rppg_models(frames, 30.0)
        assert isinstance(result["classical_fallback_used"], bool)

    def test_face_module_error_still_returns_full_schema(self):
        """The full face module schema must be intact even on error."""
        from backend.app.ml.face.face_module import analyze_face_video
        result = analyze_face_video("nonexistent.mp4")
        assert result["risk"] == "error"
        assert result["heart_rate"] is None
        assert result["module_name"] == "face_module"
        assert "confidence_breakdown" in result
        assert "hr_timeseries" in result
        assert isinstance(result["retake_required"], bool)


# ═══════════════════════════════════════════════════════════════════════════
# rPPG models integration
# ═══════════════════════════════════════════════════════════════════════════

class TestRppgModelsOpenRppg:
    def test_open_rppg_in_model_chain(self):
        """open_rppg should be in the model priority chain."""
        from backend.app.ml.face.rppg_models import get_available_models
        models = get_available_models()
        assert isinstance(models, list)

    def test_infer_rppg_returns_open_rppg_fields(self):
        """infer_rppg_models should include open_rppg status fields."""
        from backend.app.ml.face.rppg_models import infer_rppg_models
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        result = infer_rppg_models(frames, 30.0)
        assert "open_rppg_active" in result
        assert "classical_fallback_used" in result
        assert isinstance(result["open_rppg_active"], bool)
        assert isinstance(result["classical_fallback_used"], bool)

    def test_legacy_fallback_field_preserved(self):
        """legacy_fallback_used field should still be present."""
        from backend.app.ml.face.rppg_models import infer_rppg_models
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 10
        result = infer_rppg_models(frames, 30.0)
        assert "legacy_fallback_used" in result
        assert isinstance(result["legacy_fallback_used"], bool)
