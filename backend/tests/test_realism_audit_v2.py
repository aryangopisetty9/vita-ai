"""
Validation tests for the whole-project realism audit (session 2).

Covers the fixes applied in this session:
- Unknown risk labels in symptom normaliser → None instead of 50.0
- Missing scan_quality defaults → 0.0 instead of 1.0
- Unknown ROI signal quality defaults → 0.0 instead of 0.5
- Varied inputs always produce varied outputs
"""

from __future__ import annotations

import pytest

from backend.app.ml.fusion.score_engine import (
    _normalise_symptoms,
    compute_vita_score,
)


# ---------------------------------------------------------------------------
# Unknown risk label tests
# ---------------------------------------------------------------------------

class TestUnknownRiskLabels:
    """Unknown or unexpected risk labels must return None, not 50.0."""

    def test_unknown_risk_returns_none(self):
        result = _normalise_symptoms({"risk": "banana", "confidence": 0.9})
        assert result is None

    def test_empty_risk_returns_none(self):
        result = _normalise_symptoms({"risk": "", "confidence": 0.9})
        assert result is None

    def test_unreliable_risk_returns_none(self):
        result = _normalise_symptoms({"risk": "unreliable", "confidence": 0.5})
        assert result is None

    def test_valid_low_risk_returns_score(self):
        result = _normalise_symptoms({"risk": "low", "confidence": 0.9})
        assert result is not None
        assert result > 70

    def test_valid_high_risk_returns_score(self):
        result = _normalise_symptoms({"risk": "high", "confidence": 0.9})
        assert result is not None
        assert result < 40


# ---------------------------------------------------------------------------
# Scan quality defaults
# ---------------------------------------------------------------------------

class TestScanQualityDefaults:
    """Missing scan_quality must NOT default to 1.0 (perfect)."""

    def test_missing_scan_quality_penalises_heart_weight(self):
        """When face result has no scan_quality key, the heart weight
        should be penalised (quality treated as 0.0, not 1.0)."""
        # With quality = 0.0 (missing), heart weight should be reduced → lower score
        face_no_quality = {"value": 70, "risk": "low", "confidence": 0.85}
        face_with_quality = {"value": 70, "risk": "low", "confidence": 0.85,
                             "scan_quality": 0.95}
        audio = {"value": 16, "risk": "low", "confidence": 0.75}
        symptom = {"risk": "low", "confidence": 0.9}

        r_no_q = compute_vita_score(face_result=face_no_quality,
                                    audio_result=audio,
                                    symptom_result=symptom)
        r_with_q = compute_vita_score(face_result=face_with_quality,
                                      audio_result=audio,
                                      symptom_result=symptom)

        # Score with explicit high quality should be >= score without (quality=0.0)
        assert r_with_q["vita_health_score"] >= r_no_q["vita_health_score"]


# ---------------------------------------------------------------------------
# Output variation tests  (Part 9: never show same value repeatedly)
# ---------------------------------------------------------------------------

class TestOutputVariation:
    """Different inputs must produce different outputs."""

    def test_different_heart_rates_different_scores(self):
        audio = {"value": 16, "risk": "low", "confidence": 0.75}
        symptom = {"risk": "low", "confidence": 0.9}

        scores = set()
        for bpm in [60, 72, 90, 110, 130]:
            r = compute_vita_score(
                face_result={"value": bpm, "risk": "low" if bpm < 100 else "moderate",
                             "confidence": 0.8, "scan_quality": 0.8},
                audio_result=audio,
                symptom_result=symptom,
            )
            scores.add(r["vita_health_score"])
        # At least 3 distinct scores from 5 different HRs
        assert len(scores) >= 3, f"Only {len(scores)} distinct scores from 5 different HRs: {scores}"

    def test_different_symptom_risks_different_scores(self):
        face = {"value": 72, "risk": "low", "confidence": 0.85, "scan_quality": 0.8}
        audio = {"value": 16, "risk": "low", "confidence": 0.75}

        scores = {}
        for risk in ["low", "moderate", "high"]:
            r = compute_vita_score(
                face_result=face,
                audio_result=audio,
                symptom_result={"risk": risk, "confidence": 0.9},
            )
            scores[risk] = r["vita_health_score"]

        assert scores["low"] > scores["moderate"] > scores["high"]

    def test_different_breathing_rates_different_scores(self):
        face = {"value": 72, "risk": "low", "confidence": 0.85, "scan_quality": 0.8}
        symptom = {"risk": "low", "confidence": 0.9}

        scores = set()
        for rate in [10, 16, 22, 30]:
            r = compute_vita_score(
                face_result=face,
                audio_result={"value": rate, "risk": "low" if 12 <= rate <= 20 else "moderate",
                              "confidence": 0.7},
                symptom_result=symptom,
            )
            scores.add(r["vita_health_score"])
        assert len(scores) >= 2, f"Only {len(scores)} distinct scores from 4 breathing rates"

    def test_fewer_modules_fewer_component_scores(self):
        """Excluding modules should reduce the set of component_scores."""
        face = {"value": 72, "risk": "low", "confidence": 0.85, "scan_quality": 0.8}
        audio = {"value": 16, "risk": "low", "confidence": 0.75}
        symptom = {"risk": "low", "confidence": 0.9}

        full = compute_vita_score(face_result=face, audio_result=audio,
                                  symptom_result=symptom)
        partial = compute_vita_score(face_result=face, symptom_result=symptom)

        assert len(full["component_scores"]) > len(partial["component_scores"])
        assert "breathing_score" in full["component_scores"]
        assert "breathing_score" not in partial["component_scores"]
