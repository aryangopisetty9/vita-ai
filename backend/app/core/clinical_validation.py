"""
Vita AI – Clinical Validation & Confidence Calibration
========================================================
Provides cross-module consistency checks, confidence calibration,
clinical safety notes, and escalation logic.

This module does NOT diagnose.  It adds guardrails and safety flags
to prevent overconfident or contradictory outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── High-concern symptom combinations ────────────────────────────────────
EMERGENCY_COMBINATIONS: List[Set[str]] = [
    {"chest pain", "breathlessness"},
    {"chest pain", "dizziness"},
    {"chest pain", "nausea"},
    {"breathlessness", "confusion"},
    {"fainting", "chest pain"},
    {"seizure"},
    {"unconsciousness"},
]

ESCALATION_COMBINATIONS: List[Set[str]] = [
    {"fever", "cough", "breathlessness"},
    {"dizziness", "fainting"},
    {"severe headache", "vision changes"},
    {"fever", "confusion"},
    {"chest tightness", "breathlessness"},
    {"rapid heartbeat", "breathlessness"},
]


def check_symptom_severity(
    detected_symptoms: List[str],
    current_risk: str,
    confidence: float,
) -> Dict[str, Any]:
    """Check symptom combinations for severity escalation.

    Returns additional flags and possibly escalated risk.
    """
    symptom_set = {s.lower().strip() for s in detected_symptoms}
    escalation_flag = False
    emergency_flag = False
    matched_combos: List[str] = []

    for combo in EMERGENCY_COMBINATIONS:
        if combo.issubset(symptom_set):
            emergency_flag = True
            matched_combos.append(f"EMERGENCY: {', '.join(sorted(combo))}")

    for combo in ESCALATION_COMBINATIONS:
        if combo.issubset(symptom_set):
            escalation_flag = True
            matched_combos.append(f"ESCALATION: {', '.join(sorted(combo))}")

    # Escalate risk if needed
    escalated_risk = current_risk
    if emergency_flag:
        escalated_risk = "high"
    elif escalation_flag and current_risk == "low":
        escalated_risk = "moderate"

    safety_note = None
    if emergency_flag:
        safety_note = (
            "The reported symptom combination may indicate a serious condition. "
            "Seek immediate medical attention or call emergency services."
        )
    elif escalation_flag:
        safety_note = (
            "The symptom combination warrants close monitoring. "
            "Consider consulting a healthcare professional promptly."
        )

    return {
        "escalated_risk": escalated_risk,
        "escalation_flag": escalation_flag,
        "emergency_flag": emergency_flag,
        "matched_combinations": matched_combos,
        "clinical_safety_note": safety_note,
    }


def check_cross_module_consistency(
    face_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
    symptom_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cross-check outputs from different modules for contradictions.

    Flags cases where module outputs disagree significantly.
    """
    warnings: List[str] = []
    consistency_score = 1.0
    low_reliability = False

    face_risk = _get_risk(face_result)
    audio_risk = _get_risk(audio_result)
    symptom_risk = _get_risk(symptom_result)

    risks = {k: v for k, v in [
        ("face", face_risk), ("audio", audio_risk), ("symptom", symptom_risk)
    ] if v is not None}

    if len(risks) >= 2:
        risk_values = list(risks.values())
        # Check for contradictions (e.g., one high and another low)
        if "high" in risk_values and "low" in risk_values:
            high_modules = [k for k, v in risks.items() if v == "high"]
            low_modules = [k for k, v in risks.items() if v == "low"]
            warnings.append(
                f"Contradiction: {', '.join(high_modules)} shows HIGH risk "
                f"but {', '.join(low_modules)} shows LOW risk. "
                "Results should be interpreted with caution."
            )
            consistency_score -= 0.3
            low_reliability = True

    # Check face quality vs confidence
    if face_result:
        face_quality = face_result.get("scan_quality") or 0.0
        face_conf = face_result.get("confidence", 0.0)
        if face_quality < 0.4 and face_conf > 0.7:
            warnings.append(
                "Face scan quality is low but confidence is high — "
                "heart rate estimate may be unreliable."
            )
            consistency_score -= 0.15
            low_reliability = True
        if face_result.get("retake_required"):
            warnings.append(
                "Face scan quality below threshold — retake recommended "
                "for more reliable results."
            )

    # Check audio quality
    if audio_result:
        audio_conf = audio_result.get("confidence", 0.0)
        if audio_conf < 0.3 and _get_risk(audio_result) in ("high", "moderate"):
            warnings.append(
                "Audio analysis has low confidence — breathing rate "
                "estimate may be unreliable."
            )
            consistency_score -= 0.1
            low_reliability = True

    # Check symptom confidence
    if symptom_result:
        sym_conf = symptom_result.get("confidence", 0.0)
        if sym_conf < 0.3 and _get_risk(symptom_result) == "high":
            warnings.append(
                "Symptom analysis has low confidence for a high-risk "
                "classification — manual review recommended."
            )
            consistency_score -= 0.1
            low_reliability = True

    consistency_score = max(consistency_score, 0.0)

    return {
        "consistency_score": round(consistency_score, 2),
        "low_reliability_warning": low_reliability,
        "consistency_warnings": warnings,
    }


def calibrate_confidence(
    raw_confidence: float,
    scan_quality: Optional[float] = None,
    signal_periodicity: Optional[float] = None,
    model_source: str = "classical_pipeline",
) -> float:
    """Adjust raw confidence using quality and source factors.

    Deep-model sources get a slight boost; low quality penalises.
    """
    cal = raw_confidence

    # Quality penalty
    if scan_quality is not None and scan_quality < 0.5:
        cal *= max(scan_quality / 0.5, 0.3)

    # Periodicity penalty
    if signal_periodicity is not None and signal_periodicity < 0.3:
        cal *= 0.7

    # Source boost (pretrained models are slightly more reliable)
    if model_source not in ("classical_pipeline", "none", "keyword-rules-only"):
        cal = min(cal * 1.1, 1.0)

    return round(max(min(cal, 1.0), 0.0), 3)


def generate_clinical_notes(
    overall_risk: str,
    consistency_result: Dict[str, Any],
    symptom_check: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate clinical-style safety notes and disclaimers."""
    notes: List[str] = []

    # Always add disclaimer
    notes.append(
        "This is a preliminary AI-based screening estimate. "
        "It does NOT constitute a medical diagnosis."
    )

    if overall_risk == "high":
        notes.append(
            "The screening suggests elevated health concern. "
            "Prompt consultation with a healthcare professional is recommended."
        )

    if consistency_result.get("low_reliability_warning"):
        notes.append(
            "Some module outputs have low reliability or show contradictions. "
            "Consider retaking scans in better conditions."
        )

    if symptom_check and symptom_check.get("emergency_flag"):
        notes.append(
            "IMPORTANT: Reported symptoms may indicate a medical emergency. "
            "If you are experiencing these symptoms, seek immediate medical help."
        )

    escalation_flag = False
    if symptom_check:
        escalation_flag = symptom_check.get("escalation_flag", False)

    return {
        "clinical_safety_note": " ".join(notes),
        "escalation_flag": escalation_flag or (symptom_check or {}).get("emergency_flag", False),
        "disclaimer": (
            "Vita AI is a screening tool only. It cannot diagnose diseases, "
            "prescribe treatments, or replace professional medical judgment."
        ),
    }


def _get_risk(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract risk label from a module result."""
    if result is None:
        return None
    return result.get("risk") or result.get("overall_risk")
