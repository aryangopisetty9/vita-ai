"""
Tests for the score engine.

Validates the Vita Health Score computation, weight rebalancing when
modules are missing, and recommendation generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


from backend.app.ml.fusion.score_engine import (
    _normalise_breathing,
    _normalise_heart_rate,
    _normalise_symptoms,
    compute_vita_score,
)

# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "vita_health_score", "overall_risk", "confidence",
    "recommendations", "component_scores",
}


def _assert_valid_schema(result: dict) -> None:
    assert isinstance(result, dict)
    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key: {key}"
    # vita_health_score can be None when no modules provide data
    if result["vita_health_score"] is not None:
        assert 0 <= result["vita_health_score"] <= 100
    assert result["overall_risk"] in {"low", "moderate", "high", "unknown"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------

class TestNormalisers:
    def test_heart_rate_normal(self):
        score = _normalise_heart_rate({"value": 72, "risk": "low", "confidence": 0.8})
        assert 85 < score <= 100

    def test_heart_rate_high(self):
        score = _normalise_heart_rate({"value": 120, "risk": "moderate", "confidence": 0.5})
        assert score < 70

    def test_heart_rate_error(self):
        score = _normalise_heart_rate({"value": None, "risk": "error", "confidence": 0})
        assert score is None

    def test_heart_rate_unreliable(self):
        score = _normalise_heart_rate({"value": 72, "risk": "unreliable", "confidence": 0})
        assert score is not None
        assert 80 <= score <= 100

    def test_breathing_normal(self):
        score = _normalise_breathing({"value": 16, "risk": "low", "confidence": 0.7})
        assert score >= 80

    def test_breathing_error(self):
        score = _normalise_breathing({"value": None, "risk": "error", "confidence": 0})
        assert score is None

    def test_breathing_unreliable(self):
        # An unreliable breathing result should be excluded from scoring
        # (returns None) so it does not contribute a fake Vita score.
        score = _normalise_breathing({"value": 16, "risk": "unreliable", "confidence": 0})
        assert score is None

    def test_symptom_low(self):
        score = _normalise_symptoms({"risk": "low", "confidence": 0.9})
        assert score > 70

    def test_symptom_high(self):
        score = _normalise_symptoms({"risk": "high", "confidence": 0.9})
        assert score < 40

    def test_symptom_error(self):
        score = _normalise_symptoms({"risk": "error", "confidence": 0})
        assert score is None


# ---------------------------------------------------------------------------
# Score computation tests
# ---------------------------------------------------------------------------

class TestComputeVitaScore:
    def _make_face(self, bpm=72, risk="low", conf=0.85):
        return {"value": bpm, "risk": risk, "confidence": conf}

    def _make_audio(self, rate=16, risk="low", conf=0.75):
        return {"value": rate, "risk": risk, "confidence": conf}

    def _make_symptom(self, risk="low", conf=0.9):
        return {"risk": risk, "confidence": conf}

    def test_all_healthy(self):
        result = compute_vita_score(
            face_result=self._make_face(),
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        _assert_valid_schema(result)
        assert result["vita_health_score"] >= 70
        assert result["overall_risk"] == "low"

    def test_high_risk_symptoms(self):
        result = compute_vita_score(
            face_result=self._make_face(),
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(risk="high", conf=0.9),
        )
        _assert_valid_schema(result)
        # Score should drop but may not be "high" overall because
        # face and audio are healthy
        assert result["vita_health_score"] < 90

    def test_missing_face(self):
        result = compute_vita_score(
            face_result=None,
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        _assert_valid_schema(result)
        assert "heart_score" not in result["component_scores"]

    def test_missing_all(self):
        result = compute_vita_score()
        _assert_valid_schema(result)
        assert result["overall_risk"] == "unknown"

    def test_only_symptoms(self):
        result = compute_vita_score(symptom_result=self._make_symptom())
        _assert_valid_schema(result)
        assert "symptom_score" in result["component_scores"]

    def test_recommendations_include_disclaimer(self):
        result = compute_vita_score(symptom_result=self._make_symptom())
        _assert_valid_schema(result)
        disclaimers = [r for r in result["recommendations"] if "not a medical diagnosis" in r.lower()]
        assert len(disclaimers) >= 1


# ---------------------------------------------------------------------------
# Realism / anti-fallback tests
# ---------------------------------------------------------------------------

class TestRealismGuards:
    """Verify that missing or error module data is EXCLUDED from the
    fusion, not silently padded with neutral defaults."""

    def _make_face(self, bpm=72, risk="low", conf=0.85):
        return {"value": bpm, "risk": risk, "confidence": conf}

    def _make_audio(self, rate=16, risk="low", conf=0.75):
        return {"value": rate, "risk": risk, "confidence": conf}

    def _make_symptom(self, risk="low", conf=0.9):
        return {"risk": risk, "confidence": conf}

    def test_error_face_excluded_from_scores(self):
        """A face scan that returned error should NOT produce a heart_score."""
        result = compute_vita_score(
            face_result={"value": None, "risk": "error", "confidence": 0.0},
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        _assert_valid_schema(result)
        assert "heart_score" not in result["component_scores"]
        assert "face" in result.get("excluded_modules", [])

    def test_unreliable_face_excluded_from_scores(self):
        """An unreliable face scan should NOT contribute a heart_score."""
        result = compute_vita_score(
            face_result={"value": None, "risk": "unreliable", "confidence": 0.0},
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        assert "heart_score" not in result["component_scores"]

    def test_error_audio_excluded_from_scores(self):
        result = compute_vita_score(
            face_result=self._make_face(),
            audio_result={"value": None, "risk": "error", "confidence": 0.0},
            symptom_result=self._make_symptom(),
        )
        assert "breathing_score" not in result["component_scores"]
        assert "audio" in result.get("excluded_modules", [])

    def test_error_symptom_excluded_from_scores(self):
        result = compute_vita_score(
            face_result=self._make_face(),
            audio_result=self._make_audio(),
            symptom_result={"risk": "error", "confidence": 0.0},
        )
        assert "symptom_score" not in result["component_scores"]
        assert "symptom" in result.get("excluded_modules", [])

    def test_all_error_modules_returns_error(self):
        result = compute_vita_score(
            face_result={"value": None, "risk": "error", "confidence": 0},
            audio_result={"value": None, "risk": "error", "confidence": 0},
            symptom_result={"risk": "error", "confidence": 0},
        )
        assert result["overall_risk"] == "unknown"
        assert result["vita_health_score"] is None

    def test_different_inputs_produce_different_scores(self):
        """Two scans with different HR should NOT produce the same score."""
        r1 = compute_vita_score(
            face_result=self._make_face(bpm=68),
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        r2 = compute_vita_score(
            face_result=self._make_face(bpm=110),
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        assert r1["vita_health_score"] != r2["vita_health_score"]

    def test_low_confidence_not_inflated(self):
        """A module with 0.0 confidence should not be treated as 0.5."""
        result = compute_vita_score(
            face_result={"value": 72, "risk": "low", "confidence": 0.0},
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        assert result["confidence"] < 0.8  # should be pulled down by 0.0 face conf

    def test_excluded_modules_field_present(self):
        """Result should always include excluded_modules list."""
        result = compute_vita_score(
            face_result=self._make_face(),
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        assert "excluded_modules" in result
        assert result["excluded_modules"] == []

    def test_used_modules_and_final_weights_present(self):
        """used_modules and final_weights must be present and consistent."""
        result = compute_vita_score(
            face_result=self._make_face(),
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        assert "used_modules" in result
        assert "final_weights" in result
        assert set(result["used_modules"]) == {"face", "breathing", "symptom"}
        assert set(result["final_weights"].keys()) == {"face", "breathing", "symptom"}
        total = sum(result["final_weights"].values())
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total}"

    def test_used_modules_subset_when_face_missing(self):
        """When face is absent, used_modules must not include 'face'."""
        result = compute_vita_score(
            audio_result=self._make_audio(),
            symptom_result=self._make_symptom(),
        )
        assert "face" not in result["used_modules"]
        assert set(result["used_modules"]).issubset({"breathing", "symptom"})
        # Remaining weights must still sum to 1
        total = sum(result["final_weights"].values())
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total}"
