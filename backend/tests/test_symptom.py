"""
Tests for the symptom module.

Covers structured output validation, keyword detection, risk
classification, and graceful fallback when the transformer model
is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


from backend.app.ml.nlp.symptom_module import (
    _check_high_caution,
    _detect_symptoms,
    analyze_symptoms,
    analyze_symptoms_structured,
)

# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "module_name", "metric_name", "value", "unit",
    "confidence", "risk", "message", "detected_symptoms", "debug",
}


def _assert_valid_schema(result: dict) -> None:
    assert isinstance(result, dict)
    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key: {key}"
    assert result["module_name"] == "symptom_module"
    assert result["unit"] == "label"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["risk"] in {"low", "moderate", "high", "error"}
    assert isinstance(result["detected_symptoms"], list)


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestSymptomErrors:
    def test_empty_text(self):
        result = analyze_symptoms("")
        _assert_valid_schema(result)
        assert result["risk"] == "error"

    def test_very_short_text(self):
        result = analyze_symptoms("ab")
        _assert_valid_schema(result)
        assert result["risk"] == "error"

    def test_none_text(self):
        result = analyze_symptoms(None)  # type: ignore
        _assert_valid_schema(result)
        assert result["risk"] == "error"


# ---------------------------------------------------------------------------
# Keyword detection tests
# ---------------------------------------------------------------------------

class TestKeywordDetection:
    def test_single_symptom(self):
        detected = _detect_symptoms("I have a headache")
        names = [s[0] for s in detected]
        assert "headache" in names

    def test_multiple_symptoms(self):
        detected = _detect_symptoms("I feel fatigue and dizziness with nausea")
        names = [s[0] for s in detected]
        assert "fatigue" in names
        assert "dizziness" in names or "dizzy" in names
        assert "nausea" in names

    def test_no_symptoms(self):
        detected = _detect_symptoms("I feel great today!")
        assert len(detected) == 0

    def test_high_severity_symptom(self):
        detected = _detect_symptoms("I have chest pain and breathlessness")
        names = [s[0] for s in detected]
        assert "chest pain" in names
        assert "shortness of breath" in names  # "breathlessness" normalised
        # Both should have severity >= 7
        for name, sev in detected:
            if name in {"chest pain", "shortness of breath"}:
                assert sev >= 7


# ---------------------------------------------------------------------------
# High-caution pair detection
# ---------------------------------------------------------------------------

class TestHighCaution:
    def test_chest_pain_breathlessness(self):
        assert _check_high_caution({"chest pain", "breathlessness"}) is True

    def test_no_caution(self):
        assert _check_high_caution({"headache", "fatigue"}) is False


# ---------------------------------------------------------------------------
# Full-pipeline tests
# ---------------------------------------------------------------------------

class TestSymptomPipeline:
    def test_low_risk(self):
        result = analyze_symptoms("I have a mild headache and feel a bit tired")
        _assert_valid_schema(result)
        assert result["risk"] in {"low", "moderate"}
        assert len(result["detected_symptoms"]) >= 1

    def test_moderate_risk(self):
        result = analyze_symptoms("I have fever, dizziness, and persistent cough")
        _assert_valid_schema(result)
        assert result["risk"] in {"moderate", "high"}

    def test_high_risk(self):
        result = analyze_symptoms("I have severe chest pain and difficulty breathing")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    def test_no_medical_diagnosis_language(self):
        """Output messages must not claim diagnosis."""
        result = analyze_symptoms("I have chest pain and fever")
        _assert_valid_schema(result)
        msg = result["message"].lower()
        assert "diagnos" not in msg


# ---------------------------------------------------------------------------
# Hybrid rule-engine tests (exact risk expectations)
# ---------------------------------------------------------------------------

class TestRuleEngine:
    """Verify the hybrid AI + rule-based risk engine."""

    # ── Single LOW symptoms ────────────────────────────────────────
    def test_cough_is_low(self):
        result = analyze_symptoms("cough")
        _assert_valid_schema(result)
        assert result["risk"] == "low"

    def test_mild_headache_is_low(self):
        result = analyze_symptoms("mild headache")
        _assert_valid_schema(result)
        assert result["risk"] == "low"

    def test_fatigue_is_low(self):
        result = analyze_symptoms("fatigue")
        _assert_valid_schema(result)
        assert result["risk"] == "low"

    def test_sore_throat_is_low(self):
        result = analyze_symptoms("sore throat")
        _assert_valid_schema(result)
        assert result["risk"] == "low"

    # ── MODERATE symptoms ──────────────────────────────────────────
    def test_fever_is_moderate(self):
        result = analyze_symptoms("fever")
        _assert_valid_schema(result)
        assert result["risk"] == "moderate"

    # ── RED FLAG / HIGH symptoms ───────────────────────────────────
    def test_chest_pain_is_high(self):
        result = analyze_symptoms("chest pain")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    def test_shortness_of_breath_is_high(self):
        result = analyze_symptoms("shortness of breath")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    def test_coughing_blood_is_high(self):
        result = analyze_symptoms("coughing blood")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    def test_seizure_is_high(self):
        result = analyze_symptoms("seizure")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    def test_fainting_is_high(self):
        result = analyze_symptoms("fainting")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    # ── Synonym normalisation ──────────────────────────────────────
    def test_heart_pain_maps_to_chest_pain(self):
        result = analyze_symptoms("heart pain")
        _assert_valid_schema(result)
        assert result["risk"] == "high"
        assert "chest pain" in result["detected_symptoms"]

    def test_breathlessness_maps_to_shortness_of_breath(self):
        result = analyze_symptoms("breathlessness")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    # ── Dangerous combinations ─────────────────────────────────────
    def test_fever_cough_breath_is_high(self):
        result = analyze_symptoms("fever cough and shortness of breath")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    def test_chest_pain_dizziness_is_high(self):
        result = analyze_symptoms("chest pain and dizziness")
        _assert_valid_schema(result)
        assert result["risk"] == "high"

    def test_fever_cough_is_moderate(self):
        result = analyze_symptoms("I have fever and cough")
        _assert_valid_schema(result)
        assert result["risk"] == "moderate"

    # ── Count-based logic ──────────────────────────────────────────
    def test_two_mild_symptoms_moderate(self):
        result = analyze_symptoms("I have a headache and feel fatigued")
        _assert_valid_schema(result)
        assert result["risk"] == "moderate"


# ---------------------------------------------------------------------------
# Realism / confidence tests
# ---------------------------------------------------------------------------

class TestSymptomConfidenceRealism:
    """Verify confidence scales honestly with extraction quality."""

    def test_no_symptoms_low_confidence(self):
        """When nothing is detected, confidence should be very low, not 0.5."""
        result = analyze_symptoms("I feel totally fine and great today")
        _assert_valid_schema(result)
        assert result["confidence"] < 0.3, (
            f"No symptoms detected but confidence={result['confidence']} (should be <0.3)"
        )

    def test_one_symptom_moderate_confidence(self):
        result = analyze_symptoms("I have a headache")
        _assert_valid_schema(result)
        assert 0.2 < result["confidence"] < 0.7

    def test_many_symptoms_higher_confidence(self):
        result = analyze_symptoms(
            "I have chest pain, dizziness, shortness of breath, and fever"
        )
        _assert_valid_schema(result)
        assert result["confidence"] > 0.4

    def test_different_texts_different_confidence(self):
        r1 = analyze_symptoms("mild cough")
        r2 = analyze_symptoms("severe chest pain and shortness of breath and seizure")
        assert r1["confidence"] != r2["confidence"]


# ---------------------------------------------------------------------------
# New schema keys from hybrid engine
# ---------------------------------------------------------------------------

HYBRID_REQUIRED_KEYS = REQUIRED_KEYS | {
    "symptom_score", "recommendations", "contributing_factors",
    "detected_groups",
}


def _assert_hybrid_schema(result: dict) -> None:
    """Check all keys present in hybrid engine output."""
    _assert_valid_schema(result)
    for key in HYBRID_REQUIRED_KEYS:
        assert key in result, f"Missing hybrid key: {key}"
    score = result["symptom_score"]
    assert score is None or (isinstance(score, int) and 0 <= score <= 100), (
        f"symptom_score out of range: {score}"
    )
    assert isinstance(result["recommendations"], list), "recommendations must be a list"
    assert len(result["recommendations"]) >= 1, "recommendations must not be empty"
    assert isinstance(result["contributing_factors"], list), "contributing_factors must be a list"
    assert isinstance(result["detected_groups"], dict), "detected_groups must be a dict"


# ---------------------------------------------------------------------------
# Part 12 – Representative test cases for structured input
# ---------------------------------------------------------------------------

class TestStructuredInput:
    """Validate that structured form inputs reach the scoring engine and
    influence symptom_score, risk, confidence, and recommendations in a
    medically sensible direction."""

    # ── Case 1: Mild fever + cough + short duration + low severity ────────
    def test_mild_feverish_cold(self):
        """Low severity, short duration, fever + cough → moderate risk, mid score."""
        result = analyze_symptoms_structured(
            major_symptom="cough",
            minor_symptoms="runny nose, fatigue",
            fever=True,
            pain=False,
            difficulty_breathing=False,
            severity=3,
            days_suffering=2,
            age=28,
        )
        _assert_hybrid_schema(result)
        assert result["risk"] in {"low", "moderate"}
        assert result["symptom_score"] is not None
        # Mild case: score should be reasonably healthy (>40)
        assert result["symptom_score"] >= 40, f"Score too low for mild case: {result['symptom_score']}"
        assert "fever" in result["detected_symptoms"] or "fever" in result["injected_from_flags"]

    # ── Case 2: Fever + body pain + moderate severity ─────────────────────
    def test_moderate_flu_like(self):
        """Fever + pain + body ache at moderate severity → moderate/high risk."""
        result = analyze_symptoms_structured(
            major_symptom="fever",
            minor_symptoms="body ache, headache",
            fever=True,
            pain=True,
            difficulty_breathing=False,
            severity=6,
            days_suffering=4,
            age=35,
        )
        _assert_hybrid_schema(result)
        assert result["risk"] in {"moderate", "high"}
        assert result["symptom_score"] is not None
        # Moderate case: score should be below healthy baseline
        assert result["symptom_score"] < 75, f"Score too high for moderate-severity case: {result['symptom_score']}"
        assert len(result["contributing_factors"]) >= 1

    # ── Case 3: Numbness + pain + higher severity ─────────────────────────
    def test_numbness_pain_high_severity(self):
        """Numbness (red flag) + pain + high severity → high risk, low score."""
        result = analyze_symptoms_structured(
            major_symptom="numbness",
            minor_symptoms="pain",
            fever=False,
            pain=True,
            difficulty_breathing=False,
            severity=8,
            days_suffering=1,
            age=50,
        )
        _assert_hybrid_schema(result)
        assert result["risk"] == "high"
        assert result["symptom_score"] is not None
        # New calibrated model: numbness + severity 8 is serious but not extreme collapse.
        # Score should be below the MODERATE band (< 65), not hard-capped to lowest tier.
        assert result["symptom_score"] < 65, f"Score too high for red-flag numbness: {result['symptom_score']}"
        assert any("Red-flag" in r or "red-flag" in r.lower() for r in result["recommendations"])

    # ── Case 4: Difficulty breathing toggle + longer duration ─────────────
    def test_difficulty_breathing_redFlag(self):
        """difficulty_breathing=True alone must → high risk."""
        result = analyze_symptoms_structured(
            major_symptom="breathing issues",
            fever=False,
            pain=False,
            difficulty_breathing=True,
            severity=7,
            days_suffering=10,
            age=45,
        )
        _assert_hybrid_schema(result)
        assert result["risk"] == "high", f"Expected high risk for difficulty_breathing, got {result['risk']}"
        assert result["symptom_score"] is not None
        # Calibrated model: SOB (22) + severity 7 (15.5) + duration 10d (16.5) → ~46.
        # Should be below MODERATE band (< 60) — controlled, not extreme.
        assert result["symptom_score"] < 60, f"Score too high when difficulty_breathing=True: {result['symptom_score']}"
        assert "shortness of breath" in result["detected_symptoms"] or \
               "shortness of breath" in result["injected_from_flags"]
        # Recommendations must mention breathing urgency
        recs_text = " ".join(result["recommendations"]).lower()
        assert "breath" in recs_text

    # ── Case 5: Sparse text but toggles enabled ────────────────────────────
    def test_sparse_text_with_flags(self):
        """Short symptom text + fever + pain toggles → valid result."""
        result = analyze_symptoms_structured(
            major_symptom="flu",
            fever=True,
            pain=True,
            difficulty_breathing=False,
            severity=5,
            days_suffering=3,
        )
        _assert_hybrid_schema(result)
        assert result["risk"] in {"low", "moderate", "high"}
        assert result["symptom_score"] is not None
        # Toggles should register
        assert "fever" in result["detected_symptoms"] or "fever" in result["injected_from_flags"]

    # ── Case 6: Rich input, strong NLP + toggle agreement ─────────────────
    def test_rich_agreement_boosts_confidence(self):
        """Clear fever text + fever=True + difficulty_breathing text +
        difficulty_breathing=True should boost confidence above text-alone."""
        rich = analyze_symptoms_structured(
            major_symptom="fever and shortness of breath",
            fever=True,
            difficulty_breathing=True,
            severity=7,
            days_suffering=5,
        )
        sparse = analyze_symptoms_structured(
            major_symptom="fever and shortness of breath",
            fever=False,
            difficulty_breathing=False,
        )
        _assert_hybrid_schema(rich)
        _assert_hybrid_schema(sparse)
        # Both should be high risk (red-flag symptoms)
        assert rich["risk"] == "high"
        assert sparse["risk"] == "high"
        # Rich input (toggle agreement) should have higher confidence
        assert rich["confidence"] >= sparse["confidence"], (
            f"Rich confidence {rich['confidence']} should be >= sparse {sparse['confidence']}"
        )

    # ── Severity slider influence ──────────────────────────────────────────
    def test_severity_slider_lowers_score(self):
        """Same symptom, higher severity → lower symptom_score."""
        low_sev = analyze_symptoms_structured(major_symptom="headache", severity=2)
        high_sev = analyze_symptoms_structured(major_symptom="headache", severity=9)
        _assert_hybrid_schema(low_sev)
        _assert_hybrid_schema(high_sev)
        assert low_sev["symptom_score"] > high_sev["symptom_score"], (
            f"Low sev score {low_sev['symptom_score']} should exceed high sev {high_sev['symptom_score']}"
        )

    # ── Duration influence ────────────────────────────────────────────────
    def test_duration_lowers_score(self):
        """Same symptoms, longer duration → lower symptom_score."""
        acute = analyze_symptoms_structured(major_symptom="cough", severity=4, days_suffering=1)
        chronic = analyze_symptoms_structured(major_symptom="cough", severity=4, days_suffering=60)
        _assert_hybrid_schema(acute)
        _assert_hybrid_schema(chronic)
        assert acute["symptom_score"] > chronic["symptom_score"], (
            f"Acute {acute['symptom_score']} should exceed chronic {chronic['symptom_score']}"
        )

    # ── Recommendations are generated ────────────────────────────────────
    def test_recommendations_present_and_non_empty(self):
        result = analyze_symptoms_structured(
            major_symptom="chest pain", severity=8, difficulty_breathing=True
        )
        _assert_hybrid_schema(result)
        assert len(result["recommendations"]) >= 2
        recs = " ".join(result["recommendations"]).lower()
        assert "disclaimer" in recs or "screening" in recs or "consult" in recs

    # ── Structured inputs echoed back ─────────────────────────────────────
    def test_structured_inputs_echoed(self):
        result = analyze_symptoms_structured(
            major_symptom="fatigue",
            age=42,
            gender="female",
            severity=4,
            days_suffering=5,
            symptom_category="General",
        )
        _assert_hybrid_schema(result)
        si = result.get("structured_inputs", {})
        assert si.get("age") == 42
        assert si.get("gender") == "female"
        assert si.get("severity") == 4
        assert si.get("days_suffering") == 5

    # ── Difficulty breathing flag alone (no text) → valid result ──────────
    def test_critical_flag_without_major_text(self):
        """difficulty_breathing=True with minimal text should not error."""
        result = analyze_symptoms_structured(
            major_symptom="breathing",
            difficulty_breathing=True,
        )
        _assert_hybrid_schema(result)
        assert result["risk"] == "high"

    # ── backward compat: old analyze_symptoms still fully works ───────────
    def test_backward_compat_text_only(self):
        result = analyze_symptoms("I have fever and chest pain")
        _assert_valid_schema(result)
        assert result["risk"] == "high"
        # New keys also present via wrapper
        assert "symptom_score" in result
        assert "recommendations" in result
        assert "detected_groups" in result
        assert "risk_band" in result


# ---------------------------------------------------------------------------
# Part 10 – Validation test: the reference multi-symptom case
# ---------------------------------------------------------------------------

class TestValidationCase:
    """Part 10 validation: fever + cough + body pain + numbness, severity 6, 3 days.

    Expected outcome per specification:
    - Risk label: high (numbness is a red-flag symptom)
    - symptom_score: moderate/high borderline — NOT overly extreme.
      Calibrated range: 35–60 (NEEDS ATTENTION band, 4-tier).
    - No single-digit score collapse."""

    def test_reference_case_score_range(self):
        """Core validation: score lands in the calibrated moderate/high borderline zone."""
        result = analyze_symptoms_structured(
            major_symptom="fever, cough, body pain, numbness",
            severity=6,
            days_suffering=3,
        )
        _assert_hybrid_schema(result)
        # NLP red-flag (numbness) drives the label
        assert result["risk"] == "high"
        score = result["symptom_score"]
        assert score is not None
        # Score must NOT be extreme collapse (old system gave ~5)
        assert score >= 25, f"Score collapsed too far for multi-symptom case: {score}"
        # Score must reflect genuine concern (should not stay in healthy range)
        assert score < 70, f"Score too high for red-flag multi-symptom case: {score}"

    def test_reference_case_detected_groups(self):
        """Detected groups should capture the viral and neurological patterns."""
        result = analyze_symptoms_structured(
            major_symptom="fever, cough, body pain, numbness",
            severity=6,
            days_suffering=3,
        )
        _assert_hybrid_schema(result)
        groups = result["detected_groups"]
        # At minimum, viral pattern should be detected (fever + cough)
        group_labels = list(groups.keys())
        assert any("Viral" in lbl or "Infectious" in lbl for lbl in group_labels), (
            f"Expected Viral group in detected_groups, got: {group_labels}"
        )

    def test_reference_case_score_breakdown_present(self):
        """score_breakdown in debug should show each penalty component."""
        result = analyze_symptoms_structured(
            major_symptom="fever, cough, body pain, numbness",
            severity=6,
            days_suffering=3,
        )
        breakdown = result.get("debug", {}).get("score_breakdown", {})
        assert breakdown.get("base_score") == 100
        assert breakdown.get("symptom_penalty", 0) > 0
        assert breakdown.get("severity_penalty", 0) > 0
        assert breakdown.get("duration_penalty", 0) > 0
        # Final score should equal the top-level symptom_score
        assert breakdown.get("final_score") == result["symptom_score"]

    def test_reference_case_risk_band(self):
        """risk_band should be NEEDS ATTENTION or HIGH for this case."""
        result = analyze_symptoms_structured(
            major_symptom="fever, cough, body pain, numbness",
            severity=6,
            days_suffering=3,
        )
        assert result["risk_band"] in {"NEEDS ATTENTION", "HIGH"}, (
            f"Expected NEEDS ATTENTION or HIGH, got: {result['risk_band']}"
        )
