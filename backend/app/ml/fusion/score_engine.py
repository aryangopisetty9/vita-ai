"""
Vita AI – Score Engine
=======================
Combines outputs from the face, audio, and symptom modules into a
single **Vita Health Score** (0-100) with an overall risk label and
human-readable recommendations.

Scoring approach
----------------
1. Normalise each module's output to a 0-100 sub-score using
   domain-specific mapping functions.
2. Apply configurable weights (default 40 / 30 / 30).
3. If a module result is missing, redistribute its weight among the
   remaining modules.
4. Derive the overall risk from score thresholds.
5. Generate contextual recommendations.

Extension points
----------------
* Replace the weighted-sum fusion with an ML model (XGBoost / small
  neural net) trained on labeled multi-modal health data.
* Add new modules (e.g. skin-lesion analysis, SpO₂) by adding a
  normaliser and updating the weight map.
* Persist the score to a database for longitudinal dashboards.

Disclaimer
----------
The Vita Health Score is a **preliminary screening indicator** and
does **not** constitute a medical diagnosis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.core.config import (
    BREATHING_NORMAL_HIGH,
    BREATHING_NORMAL_LOW,
    DEFAULT_WEIGHTS,
    RPPG_NORMAL_HIGH,
    RPPG_NORMAL_LOW,
    SCORE_THRESHOLDS,
)
from backend.app.ml.fusion.fusion_model import predict_score as fusion_predict, get_fusion_status
from backend.app.core.clinical_validation import (
    check_symptom_severity,
    check_cross_module_consistency,
    calibrate_confidence,
    generate_clinical_notes,
)

logger = logging.getLogger(__name__)


def _apply_supportive_face_features(
    face_result: Dict[str, Any],
    scores: Dict[str, float],
) -> None:
    """Add low-weight supportive signals from face behavioral features.

    These features are treated as minor adjustments (±3 points max)
    to the heart score.  They never dominate the final Vita score.
    """
    adjustment = 0.0

    # Eye stability: very unstable eyes may indicate fatigue
    eye_stab = face_result.get("eye_stability")
    if eye_stab is not None:
        # 1.0 = stable → +1, 0.0 = unstable → -1
        adjustment += (eye_stab - 0.5) * 2.0  # range: -1 to +1

    # Skin signal stability: stable signal → small positive
    skin_stab = face_result.get("skin_signal_stability")
    if skin_stab is not None:
        adjustment += (skin_stab - 0.5) * 1.0  # range: -0.5 to +0.5

    # Facial tension: high tension → small negative
    tension = face_result.get("facial_tension_index")
    if tension is not None:
        adjustment -= tension * 1.5  # range: 0 to -1.5

    # Clamp total adjustment to ±3 points
    adjustment = float(np.clip(adjustment, -3.0, 3.0))
    scores["heart_score"] = float(np.clip(scores["heart_score"] + adjustment, 0, 100))


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation helpers (module output → 0-100 sub-score)
# ═══════════════════════════════════════════════════════════════════════════

def _normalise_heart_rate(result: Dict[str, Any]) -> Optional[float]:
    """Convert heart-rate result to a 0-100 score.

    Supports both legacy (value) and new (heart_rate) field names.
    Returns None when the module produced no usable result (error /
    unreliable / missing value) so that the fusion layer can exclude
    the module entirely instead of treating missing data as neutral.

    Scoring logic (for resting HR in adults):
    - 60-80 bpm  → 95-100  (ideal)
    - 50-60 / 80-100 → 70-95  (acceptable)
    - outside 50-100 → linearly decreasing
    """
    # Support both new and legacy field names
    bpm = result.get("heart_rate") or result.get("value")
    risk = result.get("risk", "")
    if bpm is None or risk == "error":
        return None  # no usable data — exclude from fusion

    bpm = float(bpm)
    ideal_low, ideal_high = 60.0, 80.0
    outer_low, outer_high = float(RPPG_NORMAL_LOW), float(RPPG_NORMAL_HIGH)

    if ideal_low <= bpm <= ideal_high:
        return 95.0 + 5.0 * (1.0 - abs(bpm - 70.0) / 10.0)
    elif outer_low <= bpm < ideal_low:
        return 70.0 + 25.0 * ((bpm - outer_low) / (ideal_low - outer_low))
    elif ideal_high < bpm <= outer_high:
        return 70.0 + 25.0 * ((outer_high - bpm) / (outer_high - ideal_high))
    elif bpm < outer_low:
        return max(20.0, 70.0 - (outer_low - bpm) * 2.0)
    else:
        return max(20.0, 70.0 - (bpm - outer_high) * 1.5)


def _normalise_breathing(result: Dict[str, Any]) -> Optional[float]:
    """Convert breathing-rate result to a 0-100 score.

    Returns None when the module has no usable result so the fusion
    layer can exclude it instead of padding with a neutral default.

    Normal adult respiratory rate: 12-20 breaths/min.
    """
    # Support both new field name (breathing_rate) and legacy (value)
    rate = result.get("breathing_rate") or result.get("value")
    risk = result.get("risk", "")
    reliability = result.get("reliability", "")
    retake = result.get("retake_required", False)
    if rate is None or risk in ("error", "unreliable") or reliability == "unreliable" or retake:
        return None

    rate = float(rate)
    low = float(BREATHING_NORMAL_LOW)
    high = float(BREATHING_NORMAL_HIGH)
    mid = (low + high) / 2.0

    if low <= rate <= high:
        # Within normal → 80-100 range, best at midpoint
        return 80.0 + 20.0 * (1.0 - abs(rate - mid) / (high - low))
    elif rate < low:
        return max(20.0, 80.0 - (low - rate) * 5.0)
    else:
        return max(20.0, 80.0 - (rate - high) * 5.0)


def _normalise_symptoms(result: Dict[str, Any]) -> Optional[float]:
    """Convert symptom result to a 0-100 score.

    If the symptom module returned a direct ``symptom_score`` (new hybrid
    engine), use it as-is for finer-grained fusion instead of the coarser
    risk-tier mapping.  Falls back to risk→score mapping for legacy results.

    Returns None when the module has no usable result.
    """
    risk = result.get("risk", "error")
    if risk == "error":
        return None

    # ── New path: direct continuous score from hybrid engine ─────────────
    direct = result.get("symptom_score")
    if direct is not None:
        try:
            val = float(direct)
            if 0.0 <= val <= 100.0:
                return round(val, 1)
        except (TypeError, ValueError):
            pass  # fall through to legacy mapping

    # ── Legacy path: map risk label → score ──────────────────────────────
    confidence = float(result.get("confidence", 0.0))
    base = {"low": 90.0, "moderate": 55.0, "high": 25.0}
    if risk not in base:
        return None
    score = base[risk]
    # Adjust slightly by confidence — low confidence pulls toward midpoint
    score = score * (0.7 + 0.3 * confidence)
    return round(float(np.clip(score, 0, 100)), 1)


def _module_reliability_factor(result: Dict[str, Any]) -> float:
    """Convert per-module reliability/confidence to a weight multiplier."""
    reliability = str(result.get("reliability", "")).lower().strip()
    if reliability == "high":
        return 1.0
    if reliability == "medium":
        return 0.8
    if reliability == "low":
        return 0.55
    if reliability == "unreliable":
        return 0.30

    # Backward-compatible fallback when reliability is not present.
    conf = result.get("confidence")
    try:
        conf_v = float(conf) if conf is not None else 0.0
    except (TypeError, ValueError):
        conf_v = 0.0
    return float(np.clip(0.25 + 0.75 * conf_v, 0.25, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Recommendation logic
# ═══════════════════════════════════════════════════════════════════════════

def _generate_recommendations(
    overall_risk: str,
    heart_score: Optional[float],
    breathing_score: Optional[float],
    symptom_score: Optional[float],
) -> List[str]:
    """Generate human-readable recommendations based on component scores."""
    recs: List[str] = []

    # Component-specific advice — only when the module actually contributed
    if heart_score is not None and heart_score < 70:
        recs.append("Your estimated heart rate is outside the typical resting range — consider resting and re-measuring.")
    if breathing_score is not None and breathing_score < 70:
        recs.append("Breathing pattern appears atypical — try a calm, controlled breathing exercise and re-test.")
    if symptom_score is not None and symptom_score < 60:
        recs.append("Some reported symptoms warrant attention — monitor closely for changes.")

    # Overall risk advice
    if overall_risk == "low":
        recs.extend([
            "Stay hydrated and maintain regular rest.",
            "Continue monitoring any changes in how you feel.",
        ])
    elif overall_risk == "moderate":
        recs.extend([
            "Consider re-testing later today to track changes.",
            "If symptoms persist or worsen, consult a healthcare professional.",
        ])
    elif overall_risk == "high":
        recs.extend([
            "Seeking timely medical attention is strongly recommended.",
            "If you experience chest pain, difficulty breathing, or confusion, contact emergency services.",
        ])

    # Always add disclaimer
    recs.append(
        "This is a preliminary screening estimate — it is not a medical diagnosis. "
        "Always consult a qualified healthcare professional for clinical advice."
    )
    return recs


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_vita_score(
    face_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
    symptom_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the combined Vita Health Score.

    Parameters
    ----------
    face_result : dict or None
        Output from ``analyze_face_video``.
    audio_result : dict or None
        Output from ``analyze_audio``.
    symptom_result : dict or None
        Output from ``analyze_symptoms``.

    Returns
    -------
    dict
        ``vita_health_score``, ``overall_risk``, ``confidence``,
        ``recommendations``, ``component_scores``.

    Integration notes
    -----------------
    * **Flutter / mobile**: Call ``POST /predict/final-score`` with
      module results collected from individual endpoints.
    * **Database**: Persist the returned score for trend tracking and
      dashboard analytics.
    * **ML fusion**: Replace the weighted-sum calculation with a
      trained XGBoost / neural fusion model.
    """
    # --- Normalise available modules ---
    scores: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    confidences: List[float] = []
    used_modules: List[str] = []
    excluded_modules: List[str] = []

    if face_result is not None:
        hr_score = _normalise_heart_rate(face_result)
        if hr_score is not None:
            scores["heart_score"] = hr_score
            weights["face"] = DEFAULT_WEIGHTS["heart"]
            weights["face"] *= _module_reliability_factor(face_result)

            # Quality-aware weight reduction: if face scan quality is low,
            # reduce the face weight so noisy features don't dominate.
            face_scan_quality = float(face_result.get("scan_quality") or 0.0)
            if face_scan_quality < 0.5:
                weights["face"] *= max(face_scan_quality / 0.5, 0.3)

            face_conf = face_result.get("confidence")
            confidences.append(float(face_conf) if face_conf is not None else 0.0)
            used_modules.append("face")

            # Supportive behavioral features (low weight, additive)
            _apply_supportive_face_features(face_result, scores)
        else:
            excluded_modules.append("face")

    if audio_result is not None:
        br_score = _normalise_breathing(audio_result)
        if br_score is not None:
            scores["breathing_score"] = br_score
            weights["breathing"] = DEFAULT_WEIGHTS["breathing"]
            weights["breathing"] *= _module_reliability_factor(audio_result)
            audio_conf = audio_result.get("confidence")
            confidences.append(float(audio_conf) if audio_conf is not None else 0.0)
            used_modules.append("breathing")
        else:
            excluded_modules.append("audio")

    if symptom_result is not None:
        sym_score = _normalise_symptoms(symptom_result)
        if sym_score is not None:
            scores["symptom_score"] = sym_score
            weights["symptom"] = DEFAULT_WEIGHTS["symptom"]
            weights["symptom"] *= _module_reliability_factor(symptom_result)
            sym_conf = symptom_result.get("confidence")
            confidences.append(float(sym_conf) if sym_conf is not None else 0.0)
            used_modules.append("symptom")
        else:
            excluded_modules.append("symptom")

    if not scores:
        return {
            "vita_health_score": None,
            "overall_risk": "unknown",
            "confidence": 0.0,
            "recommendations": [
                "No module results provided. Please run at least one analysis module."
            ],
            "component_scores": {},
            "used_modules": [],
            "excluded_modules": excluded_modules,
            "final_weights": {},
        }

    # --- Rebalance weights if some modules are missing ---
    total_weight = sum(weights.values())
    normalised_weights = {k: v / total_weight for k, v in weights.items()}

    # --- Weighted score (or trained fusion model if available) ---
    weight_key_map = {
        "face": "heart_score",
        "breathing": "breathing_score",
        "symptom": "symptom_score",
    }

    # Only call the ML fusion model when ALL three modules have real
    # scores.  If any module is excluded (None normalisation), the
    # model would receive a fake 50.0 placeholder — defeating the
    # purpose of the fixes.  Fall back to weighted-sum in that case.
    fusion_method = "weighted_sum"
    fused = None

    if not excluded_modules and len(scores) == 3:
        fused = fusion_predict(
            scores["heart_score"],
            scores["breathing_score"],
            scores["symptom_score"],
            confidences,
        )

    if fused is not None:
        vita_score = int(round(fused))
        fusion_method = get_fusion_status().get("method", "trained_model")
    else:
        vita_score = 0.0
        for wk, w in normalised_weights.items():
            score_key = weight_key_map[wk]
            vita_score += w * scores[score_key]
        vita_score = int(round(np.clip(vita_score, 0, 100)))

    # --- Overall risk ---
    if vita_score >= SCORE_THRESHOLDS["low"]:
        overall_risk = "low"
    elif vita_score >= SCORE_THRESHOLDS["moderate"]:
        overall_risk = "moderate"
    else:
        overall_risk = "high"

    # --- Aggregate confidence ---
    avg_confidence = round(float(np.mean(confidences)), 2) if confidences else 0.0

    # --- Recommendations ---
    recommendations = _generate_recommendations(
        overall_risk,
        scores.get("heart_score"),       # None when module was excluded
        scores.get("breathing_score"),
        scores.get("symptom_score"),
    )

    # If modules were excluded, inform the user
    if excluded_modules:
        module_labels = {"face": "heart-rate (face)", "audio": "breathing (audio)", "symptom": "symptom"}
        excluded_names = [module_labels.get(m, m) for m in excluded_modules]
        recommendations.insert(
            0,
            f"Note: {', '.join(excluded_names)} analysis could not produce a reliable "
            f"result and was excluded from the overall score. The score reflects only "
            f"the modules that returned usable data.",
        )

    # --- Clinical validation & consistency checks ---
    consistency = check_cross_module_consistency(
        face_result, audio_result, symptom_result,
    )

    # Check symptom severity escalation
    symptom_check = None
    if symptom_result:
        detected = symptom_result.get("detected_symptoms", [])
        symptom_check = check_symptom_severity(
            detected, overall_risk, avg_confidence,
        )
        # Apply escalation
        if symptom_check.get("escalated_risk") != overall_risk:
            overall_risk = symptom_check["escalated_risk"]
        if symptom_check.get("clinical_safety_note"):
            recommendations.insert(0, symptom_check["clinical_safety_note"])

    # Calibrate confidence
    scan_quality = None
    signal_period = None
    model_source = "classical_pipeline"
    if face_result:
        scan_quality = face_result.get("scan_quality")
        signal_period = face_result.get("signal_periodicity")
        model_source = face_result.get("inference_source", "classical_pipeline")

    calibrated_confidence = calibrate_confidence(
        avg_confidence,
        scan_quality=scan_quality,
        signal_periodicity=signal_period,
        model_source=model_source,
    )

    clinical_notes = generate_clinical_notes(
        overall_risk, consistency, symptom_check,
    )

    return {
        "vita_health_score": vita_score,
        "overall_risk": overall_risk,
        "confidence": calibrated_confidence,
        "recommendations": recommendations,
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "used_modules": used_modules,
        "excluded_modules": excluded_modules,
        "final_weights": {k: round(v, 4) for k, v in normalised_weights.items()},
        "fusion_method": fusion_method,
        "consistency": consistency,
        "clinical_notes": clinical_notes,
    }
