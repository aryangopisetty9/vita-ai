"""
Vita AI – Symptom Module
=========================
Hybrid symptom analysis engine combining NLP text extraction with
structured form inputs.

Pipeline
--------
A. NLP layer
   1. Attempt to load a lightweight HuggingFace transformer pipeline for
      text-tone classification (DistilBERT sentiment as severity-tone proxy).
   2. Run keyword / rule-based symptom extraction against a curated
      symptom-severity database.
   3. Apply synonym normalisation and misspelling correction.

B. Structured layer (new — used when form fields are provided)
   4. Inject boolean toggle flags (fever, pain, difficulty_breathing) as
      explicit detected symptoms when not already captured by NLP.
   5. Apply severity slider, days_suffering, age as continuous score
      adjustments anchored on the rule-based risk tier.

C. Fusion layer
   6. Combine NLP risk + structured adjustments into a continuous
      symptom_score (0-100, higher = healthier / lower concern).
   7. Compute confidence honestly from evidence quality + toggle agreement.
   8. Generate contextual, actionable recommendations.

If the transformer is unavailable, the engine falls back to keyword
rules — it never crashes.

Disclaimer
----------
This module provides a **preliminary screening estimate** only.
It does **not** diagnose any medical condition.  Users should
always consult a qualified healthcare professional.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional transformer import – loaded lazily on first call so that
# module import is fast and tests can collect without downloading models.
# ---------------------------------------------------------------------------
_pipeline = None        # Will hold the HF pipeline once loaded
_pipeline_loaded = False  # Track whether we've attempted to load
_MODEL_NAME = "keyword-rules-only"


def _ensure_pipeline() -> None:
    """Lazily load the HuggingFace pipeline on first use."""
    global _pipeline, _pipeline_loaded, _MODEL_NAME
    if _pipeline_loaded:
        return
    _pipeline_loaded = True
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,  # CPU
        )
        _MODEL_NAME = "distilbert-sentiment"
        logger.info("Loaded HuggingFace sentiment pipeline for symptom module.")
    except Exception as exc:
        _MODEL_NAME = "keyword-rules-only"
        logger.warning("HuggingFace pipeline unavailable (%s). Using keyword fallback.", exc)

from backend.app.core.config import HIGH_CAUTION_PAIRS, SYMPTOM_MIN_TEXT_LENGTH
from backend.app.ml.nlp.nlp_models import (
    compare_with_distilbert,
    infer_nlp_models,
)

# ---------------------------------------------------------------------------
# Synonym normalisation (AI-assisted phrases → canonical names)
# ---------------------------------------------------------------------------
SYMPTOM_SYNONYMS: Dict[str, str] = {
    "heart pain": "chest pain",
    "chest tightness": "chest pain",
    "chest pressure": "chest pain",
    "breathless": "shortness of breath",
    "breathlessness": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "can't breathe": "shortness of breath",
    "hard to breathe": "shortness of breath",
    "dizzy": "dizziness",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "tiredness": "fatigue",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "fatigued": "fatigue",
    "high temperature": "fever",
    "high fever": "fever",
    "throwing up": "vomiting",
    "passed out": "fainting",
    "fainted": "fainting",
    "unconscious": "unconsciousness",
    "loss of consciousness": "unconsciousness",
    "blacked out": "unconsciousness",
    "coughing up blood": "coughing blood",
    "hemoptysis": "coughing blood",
    "fits": "seizure",
    "convulsions": "seizure",
    "body ache": "muscle pain",
    "body pain": "muscle pain",
    "muscle ache": "muscle pain",
    "back pain": "muscle pain",
    "stomach ache": "nausea",
    "feeling sick": "nausea",
    "common cold": "congestion",
    "stuffy nose": "congestion",
    "shortness in breath": "shortness of breath",
    "short of breath": "shortness of breath",
    "loss of feeling": "numbness",
    "tingling": "numbness",
    "pins and needles": "numbness",
}

# ---------------------------------------------------------------------------
# Severity tiers  (LOW / MODERATE / HIGH-RED-FLAG)
# ---------------------------------------------------------------------------
LOW_SYMPTOMS: Set[str] = {
    "cough", "mild headache", "fatigue", "sore throat",
    "runny nose", "sneezing", "congestion", "bloating",
    "muscle pain", "insomnia", "loss of appetite",
}

MODERATE_SYMPTOMS: Set[str] = {
    "fever", "persistent cough", "vomiting", "headache",
    "dizziness", "diarrhea", "chills", "weakness",
    "joint pain", "rash", "swelling", "palpitations",
    "blurred vision", "nausea",
}

RED_FLAG_SYMPTOMS: Set[str] = {
    "chest pain", "shortness of breath", "fainting",
    "unconsciousness", "coughing blood", "seizure",
    "severe dizziness", "confusion", "numbness",
    "severe headache", "blood in stool", "blood in urine",
    "sudden weight loss",
}

# Dangerous combinations that escalate to HIGH regardless of individual tier
DANGEROUS_COMBOS_HIGH: List[Set[str]] = [
    {"fever", "cough", "shortness of breath"},
    {"chest pain", "dizziness"},
    {"chest pain", "shortness of breath"},
    {"chest pain", "nausea"},
    {"fainting", "chest pain"},
    {"shortness of breath", "confusion"},
]

DANGEROUS_COMBOS_MODERATE: List[Set[str]] = [
    {"fever", "cough"},
    {"dizziness", "fainting"},
    {"fever", "vomiting"},
    {"fever", "headache"},
]

# Legacy lookup kept for _detect_symptoms matching — severity score aids
# ordering but the rule engine above decides the final risk.
SYMPTOM_DB: Dict[str, int] = {
    # LOW tier
    "cough": 2, "mild headache": 2, "fatigue": 2, "sore throat": 2,
    "runny nose": 1, "sneezing": 1, "congestion": 2, "bloating": 2,
    "muscle pain": 3, "insomnia": 3, "loss of appetite": 3,
    # MODERATE tier
    "fever": 5, "persistent cough": 6, "vomiting": 5, "headache": 4,
    "dizziness": 4, "diarrhea": 4, "chills": 5, "weakness": 4,
    "joint pain": 4, "rash": 4, "swelling": 5, "palpitations": 6,
    "blurred vision": 5, "nausea": 4,
    # RED FLAG tier
    "chest pain": 9, "shortness of breath": 9, "fainting": 8,
    "unconsciousness": 10, "coughing blood": 10, "seizure": 10,
    "severe dizziness": 7, "confusion": 8, "numbness": 7,
    "severe headache": 7, "blood in stool": 8, "blood in urine": 8,
    "sudden weight loss": 7,
}

# ---------------------------------------------------------------------------
# Per-symptom score PENALTIES used in the base_score=100 model.
# Higher value = greater concern deduction from the healthy baseline.
# LOW tier: 3–7, MODERATE tier: 7–11, RED FLAG tier: 14–25.
# These are separate from SYMPTOM_DB (which drives detection/ordering only).
# ---------------------------------------------------------------------------
SYMPTOM_PENALTIES: Dict[str, float] = {
    # LOW tier — 3-7 pts
    "cough": 5.0,         "mild headache": 4.0, "fatigue": 5.0,
    "sore throat": 4.0,   "runny nose": 3.0,    "sneezing": 3.0,
    "congestion": 4.0,    "bloating": 4.0,      "muscle pain": 6.0,
    "insomnia": 4.0,      "loss of appetite": 5.0,
    # MODERATE tier — 7-11 pts
    "fever": 9.0,         "persistent cough": 10.0, "vomiting": 9.0,
    "headache": 8.0,      "dizziness": 8.0,         "diarrhea": 8.0,
    "chills": 8.0,        "weakness": 7.0,          "joint pain": 8.0,
    "rash": 7.0,          "swelling": 8.0,          "palpitations": 10.0,
    "blurred vision": 9.0, "nausea": 7.0,
    # RED FLAG tier — 14-25 pts
    "chest pain": 22.0,   "shortness of breath": 22.0, "fainting": 17.0,
    "unconsciousness": 25.0, "coughing blood": 25.0,  "seizure": 25.0,
    "severe dizziness": 14.0, "confusion": 20.0,       "numbness": 18.0,
    "severe headache": 14.0,  "blood in stool": 20.0,  "blood in urine": 20.0,
    "sudden weight loss": 14.0,
}

# ---------------------------------------------------------------------------
# Symptom clusters – groups of symptoms that form recognisable clinical
# patterns.  When 2+ symptoms from the same cluster co-occur, an extra
# cluster penalty is applied to reflect the significance of the pattern.
# ---------------------------------------------------------------------------
SYMPTOM_CLUSTERS: Dict[str, Set[str]] = {
    "viral": {
        "fever", "cough", "runny nose", "chills", "sore throat",
        "sneezing", "congestion", "fatigue",
    },
    "pain_related": {
        "chest pain", "muscle pain", "joint pain", "headache",
        "severe headache", "mild headache",
    },
    "neurological": {
        "numbness", "confusion", "dizziness", "severe dizziness",
        "fainting", "unconsciousness", "seizure", "blurred vision",
    },
    "respiratory": {
        "shortness of breath", "persistent cough", "coughing blood",
    },
    "gastrointestinal": {
        "nausea", "vomiting", "diarrhea", "bloating", "loss of appetite",
    },
    "systemic": {
        "fatigue", "weakness", "rash", "swelling", "palpitations",
        "sudden weight loss",
    },
}

CLUSTER_LABELS: Dict[str, str] = {
    "viral":           "Viral / Infectious",
    "pain_related":    "Pain-Related",
    "neurological":    "Neurological",
    "respiratory":     "Respiratory",
    "gastrointestinal": "Gastrointestinal",
    "systemic":        "Systemic / General",
}

# Symptoms that count as "pain present" for the pain=True toggle injection.
_PAIN_SYMPTOM_SET: Set[str] = SYMPTOM_CLUSTERS["pain_related"]


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_error_result(message: str, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "module_name": "symptom_module",
        "metric_name": "symptom_risk",
        "value": None,
        "unit": "label",
        "confidence": 0.0,
        "risk": "error",
        "risk_band": None,
        "symptom_score": None,
        "message": message,
        "recommendations": [
            "Please provide a symptom description so the system can assess your inputs.",
            "This screening is for informational purposes only and does not constitute "
            "a medical diagnosis. Always consult a qualified healthcare professional.",
        ],
        "detected_symptoms": [],
        "detected_groups": {},
        "contributing_factors": [],
        "severity_modifiers": [],
        "duration_info": None,
        "debug": debug or {},
    }


# ── Text preprocessing ──────────────────────────────────────────────────

# Common misspellings → correct form
_SPELLING_FIXES: Dict[str, str] = {
    "headach": "headache", "stomache": "stomach", "nausious": "nauseous",
    "nasuea": "nausea", "diarhea": "diarrhea", "diarrhoea": "diarrhea",
    "brething": "breathing", "breathng": "breathing",
    "chets": "chest", "dizzy ness": "dizziness", "fevar": "fever",
    "coff": "cough", "coughin": "coughing", "vommit": "vomit",
    "fatige": "fatigue", "tierd": "tired", "palpatations": "palpitations",
    "swolen": "swollen", "numbnes": "numbness",
}


def _preprocess_text(text: str) -> str:
    """Clean and normalise input text before symptom extraction.

    Steps: strip → lowercase → collapse whitespace → fix common
    misspellings → remove trailing/leading punctuation clutter.
    """
    text = text.strip().lower()
    # Collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing punctuation noise (keep internal)
    text = text.strip(".,;:!?-– ")
    # Spelling fixes (longest first)
    for wrong in sorted(_SPELLING_FIXES, key=len, reverse=True):
        pattern = r"\b" + re.escape(wrong) + r"\b"
        text = re.sub(pattern, _SPELLING_FIXES[wrong], text)
    return text


# ── Severity modifier extraction ────────────────────────────────────────

_SEVERITY_ESCALATE: Dict[str, str] = {
    "severe": "high", "very bad": "high", "extreme": "high",
    "worst": "high", "unbearable": "high", "excruciating": "high",
    "intense": "high", "terrible": "high", "worsening": "moderate",
    "getting worse": "moderate", "persistent": "moderate",
    "constant": "moderate", "chronic": "moderate", "recurring": "moderate",
}

_SEVERITY_DEESCALATE = {"mild", "slight", "minor", "little bit", "a bit"}


def _extract_severity_modifiers(text: str) -> List[Dict[str, str]]:
    """Detect severity modifiers in the text.

    Returns a list of ``{"modifier": ..., "effect": "escalate"|"deescalate"}``.
    """
    text_lower = text.lower()
    found: List[Dict[str, str]] = []
    for phrase, level in _SEVERITY_ESCALATE.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", text_lower):
            found.append({"modifier": phrase, "effect": "escalate", "to": level})
    for phrase in _SEVERITY_DEESCALATE:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text_lower):
            found.append({"modifier": phrase, "effect": "deescalate"})
    return found


# ── Duration extraction ─────────────────────────────────────────────────

_DURATION_PATTERNS = [
    (r"\bfor\s+(\d+)\s+(day|days|week|weeks|month|months|hour|hours)\b", "for"),
    (r"\bsince\s+(yesterday|last\s+week|last\s+night|this\s+morning)\b", "since"),
    (r"\b(\d+)\s+(day|days|week|weeks|month|months)\s+ago\b", "ago"),
    (r"\ball\s+(day|night|week)\b", "duration"),
    (r"\b(several|few|couple)\s+(day|days|week|weeks)\b", "vague"),
]


def _extract_duration(text: str) -> Optional[Dict[str, str]]:
    """Extract temporal duration info from the text if present."""
    text_lower = text.lower()
    for pattern, tag in _DURATION_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            return {"raw": m.group(0), "type": tag}
    return None


def _normalise_text(text: str) -> str:
    """Apply synonym normalisation so the keyword detector finds canonical names."""
    lower = text.lower()
    # Replace longest synonyms first to avoid partial matches.
    for alias in sorted(SYMPTOM_SYNONYMS, key=len, reverse=True):
        pattern = r"\b" + re.escape(alias) + r"\b"
        lower = re.sub(pattern, SYMPTOM_SYNONYMS[alias], lower)
    return lower


def _detect_symptoms(text: str) -> List[Tuple[str, int]]:
    """Match known symptoms in the text (case-insensitive).

    Applies synonym normalisation first so phrases like "heart pain" are
    mapped to canonical names ("chest pain") before matching.

    Returns a list of (symptom_name, severity_score) tuples, sorted by
    severity descending.
    """
    text_lower = _normalise_text(text)
    found: List[Tuple[str, int]] = []
    matched_spans: List[Tuple[int, int]] = []

    # Sort symptom keys longest-first so multi-word symptoms match before
    # their single-word substrings (e.g. "severe headache" before "headache").
    for symptom in sorted(SYMPTOM_DB, key=len, reverse=True):
        pattern = r"\b" + re.escape(symptom) + r"\b"
        match = re.search(pattern, text_lower)
        if match:
            # Avoid overlapping matches
            start, end = match.span()
            overlap = any(s <= start < e or s < end <= e for s, e in matched_spans)
            if not overlap:
                found.append((symptom, SYMPTOM_DB[symptom]))
                matched_spans.append((start, end))

    found.sort(key=lambda x: x[1], reverse=True)
    return found


def _check_high_caution(detected: Set[str]) -> bool:
    """Return True if any high-caution symptom pair is present."""
    for pair in HIGH_CAUTION_PAIRS:
        if pair.issubset(detected):
            return True
    return False


def _compute_risk(
    detected: List[Tuple[str, int]],
    transformer_negative_score: Optional[float],
) -> Tuple[str, float, str]:
    """Rule-based risk engine applied AFTER AI extraction.

    Priority order:
      1. Any RED FLAG symptom → HIGH
      2. Dangerous combination → HIGH or MODERATE
      3. Count-based: 1 mild → LOW, 2-3 mild → MODERATE
      4. Any MODERATE-tier symptom → MODERATE
    """
    if not detected:
        return (
            "low",
            0.15,
            "No specific symptoms detected. If you are feeling unwell, "
            "please describe your symptoms in more detail.",
        )

    names: Set[str] = {s[0] for s in detected}

    # ── STEP 1: Red-flag check ──────────────────────────────────
    has_red_flag = bool(names & RED_FLAG_SYMPTOMS)
    if has_red_flag:
        red_found = sorted(names & RED_FLAG_SYMPTOMS)
        confidence = _calc_confidence(detected, transformer_negative_score)
        return (
            "high",
            confidence,
            f"Reported symptoms include red-flag indicator(s): "
            f"{', '.join(red_found)}. Seek medical attention promptly.",
        )

    # ── STEP 2: Dangerous combinations ──────────────────────────
    for combo in DANGEROUS_COMBOS_HIGH:
        if combo.issubset(names):
            confidence = _calc_confidence(detected, transformer_negative_score)
            return (
                "high",
                confidence,
                "Reported symptom combination warrants immediate "
                "medical evaluation.",
            )

    for combo in DANGEROUS_COMBOS_MODERATE:
        if combo.issubset(names):
            confidence = _calc_confidence(detected, transformer_negative_score)
            return (
                "moderate",
                confidence,
                "Reported symptom combination suggests moderate concern. "
                "Monitor closely and consult a healthcare professional "
                "if symptoms persist or worsen.",
            )

    # ── STEP 3 & 4: Tier-based + count logic ────────────────────
    has_moderate = bool(names & MODERATE_SYMPTOMS)
    low_only = names - MODERATE_SYMPTOMS - RED_FLAG_SYMPTOMS

    if has_moderate:
        risk = "moderate"
        message = (
            "Reported symptoms suggest moderate concern. Monitor closely "
            "and consider consulting a healthcare professional if they persist."
        )
    elif len(low_only) >= 2:
        risk = "moderate"
        message = (
            "Multiple mild symptoms reported. Monitor and rest; "
            "consult a professional if they persist."
        )
    else:
        risk = "low"
        message = (
            "Reported symptoms appear mild. Continue monitoring "
            "and stay hydrated."
        )

    # Transformer tone is captured in confidence scoring but does NOT
    # override the rule engine's category-based risk decision.  Symptom
    # words are inherently \"negative\" to a generic sentiment model, so
    # using sentiment to escalate would produce false positives on mild
    # symptoms like \"cough\" or \"fatigue\".

    confidence = _calc_confidence(detected, transformer_negative_score)
    return risk, confidence, message


def _calc_confidence(
    detected: List[Tuple[str, int]],
    transformer_negative_score: Optional[float],
) -> float:
    """Confidence based on symptom count and AI agreement.

    Scales from 0.15 (no symptoms detected — uncertain) to ~0.95
    (many clear symptoms with AI agreement).  Previous formula started
    at 0.5 for zero symptoms, which overstated certainty when the
    extractor found nothing.
    """
    n = len(detected)
    # 0 symptoms → 0.15, 1 → 0.35, 3 → 0.55, 5+ → 0.75
    keyword_conf = min(1.0, 0.15 + n * 0.20) if n > 0 else 0.15

    if transformer_negative_score is not None:
        # Blend keyword + transformer, but only when keywords were found
        # does the transformer score carry meaningful weight.
        if n > 0:
            confidence = 0.6 * keyword_conf + 0.4 * transformer_negative_score
        else:
            # No keywords: transformer may react to tone, but we don't
            # trust it alone — minor contribution only.
            confidence = 0.15 + 0.10 * transformer_negative_score
    else:
        confidence = keyword_conf * 0.85

    return round(float(min(confidence, 1.0)), 2)


# ═══════════════════════════════════════════════════════════════════════════
# Structured-field helpers (Part B of the hybrid pipeline)
# ═══════════════════════════════════════════════════════════════════════════

# Risk rank used across several functions — define once at module level.
_RISK_RANK: Dict[str, int] = {"low": 0, "moderate": 1, "high": 2}


def _compute_symptom_penalty(detected_names: Set[str]) -> Tuple[float, List[str]]:
    """Compute the total symptom penalty using a top-N weighted scheme.

    The three highest-penalty symptoms count in full; every additional
    symptom contributes 50 % of its penalty.  This prevents mild co-occurring
    symptoms (e.g. sneezing + bloating + insomnia) from producing the same
    penalty as a single red-flag symptom.

    Returns ``(total_penalty, symptoms_ordered_by_penalty_descending)``.
    """
    ordered = sorted(
        detected_names,
        key=lambda n: SYMPTOM_PENALTIES.get(n, 5.0),
        reverse=True,
    )
    penalties = [SYMPTOM_PENALTIES.get(n, 5.0) for n in ordered]
    full   = sum(penalties[:3])
    partial = sum(p * 0.5 for p in penalties[3:])
    return round(full + partial, 1), ordered


def _compute_cluster_penalty(
    detected_names: Set[str],
) -> Tuple[float, Dict[str, List[str]]]:
    """Bonus penalty when 2+ symptoms form a recognisable clinical cluster.

    A 2-symptom cluster adds 5 pts; a 3+-symptom cluster adds 8 pts.
    Total cluster penalty is capped at 20 pts (prevents compounding).

    Returns ``(cluster_penalty, detected_groups)`` where ``detected_groups``
    maps each active cluster label to the list of its matching symptoms.
    """
    detected_groups: Dict[str, List[str]] = {}
    cluster_penalty = 0.0
    for cluster_id, cluster_syms in SYMPTOM_CLUSTERS.items():
        hits = sorted(detected_names & cluster_syms)
        if hits:
            detected_groups[CLUSTER_LABELS[cluster_id]] = hits
            if len(hits) >= 3:
                cluster_penalty += 8.0
            elif len(hits) >= 2:
                cluster_penalty += 5.0
    return round(min(cluster_penalty, 20.0), 1), detected_groups


def _compute_severity_penalty(severity: Optional[int]) -> Tuple[float, Optional[str]]:
    """Tiered severity penalty aligned to the 0–10 slider.

    1–3  → mild impact   (0–3 pts)
    4–6  → moderate impact (3.5–10.5 pts)
    7–10 → strong impact  (15.5–30.5 pts)

    Returns ``(penalty, human_readable_factor)``.
    """
    if severity is None:
        return 0.0, None
    if severity <= 3:
        pen = max(0.0, (3 - severity) * 1.5)
        label: Optional[str] = f"Mild severity ({severity}/10)" if severity <= 2 else None
    elif severity <= 6:
        pen = (severity - 3) * 3.5
        label = f"Moderate severity ({severity}/10)"
    else:
        pen = 10.5 + (severity - 6) * 5.0
        label = f"High severity ({severity}/10)"
    return round(pen, 1), label


def _compute_duration_penalty(days: Optional[int]) -> Tuple[float, Optional[str]]:
    """Duration penalty that scales with days suffering.

    1–2 days  → low impact   (1.5–3 pts)
    3–5 days  → moderate     (5–9 pts)
    6–14 days → stronger     (10.5–20 pts)
    >14 days  → strong+, capped at 28 pts

    Returns ``(penalty, human_readable_factor)``.
    """
    if days is None or days < 0:
        return 0.0, None
    if days == 0:
        return 0.0, None
    if days <= 2:
        pen = days * 1.5
        label: Optional[str] = f"Very recent onset ({days} day{'s' if days != 1 else ''})"
    elif days <= 5:
        pen = 3.0 + (days - 2) * 2.0
        label = f"Duration: {days} days"
    elif days <= 14:
        pen = 9.0 + (days - 5) * 1.5
        label = f"Extended duration ({days} days)"
    else:
        pen = min(20.0 + (days - 14) * 0.3, 28.0)
        label = f"Prolonged duration ({days} days)"
    return round(pen, 1), label


def _compute_age_penalty(age: Optional[int]) -> Tuple[float, Optional[str]]:
    """Modest contextual penalty for age brackets with higher screening concern.

    Very young (<5) and older adults (>75): −6 pts.
    Children (<12) and seniors (>65): −3 pts.
    Working-age adults: no penalty.
    """
    if age is None:
        return 0.0, None
    if age < 5 or age > 75:
        return 6.0, f"Age {age} (elevated screening concern)"
    if age < 12 or age > 65:
        return 3.0, f"Age {age} (mild elevated concern)"
    return 0.0, None


def _score_to_risk_band(score: int) -> str:
    """Map a 0–100 symptom_score to a 4-tier risk band label.

    80–100 → LOW
    60–79  → MODERATE
    40–59  → NEEDS ATTENTION
    <40    → HIGH
    """
    if score >= 80:
        return "LOW"
    if score >= 60:
        return "MODERATE"
    if score >= 40:
        return "NEEDS ATTENTION"
    return "HIGH"


def _compute_structured_confidence_boost(
    detected_names: Set[str],
    severity: Optional[int],
    days_suffering: Optional[int],
    fever: bool,
    difficulty_breathing: bool,
) -> float:
    """Return a small additional confidence boost from structured field completeness.

    Toggle-NLP agreement (e.g. user reports fever AND NLP detected "fever")
    increases confidence because two independent signals agree.  Simply
    providing structured fields (without text agreement) gives a small
    completeness boost.
    """
    boost = 0.0
    if severity is not None:
        boost += 0.05  # form more complete
    if days_suffering is not None:
        boost += 0.05  # form more complete
    if fever and "fever" in detected_names:
        boost += 0.10  # toggle agrees with NLP
    if difficulty_breathing and "shortness of breath" in detected_names:
        boost += 0.10  # toggle agrees with NLP
    return round(min(boost, 0.20), 2)


def _generate_recommendations(
    risk: str,
    symptom_score: int,
    detected_symptoms: List[str],
    severity: Optional[int],
    days_suffering: Optional[int],
    fever: bool,
    difficulty_breathing: bool,
    age: Optional[int],
    detected_groups: Optional[Dict[str, List[str]]] = None,
    symptom_category: Optional[str] = None,
) -> List[str]:
    """Generate actionable, context-aware recommendations.

    Honest design principles:
    - Escalate urgency language only when red flags or extreme inputs justify it.
    - Do NOT suggest specific diagnoses or treatments.
    - Always include the screening disclaimer.
    - Recommendations are ordered: most urgent first.
    - Include cluster group context so users understand symptom patterns.
    """
    recs: List[str] = []
    symptom_set = set(detected_symptoms)
    detected_groups = detected_groups or {}

    # ── Red-flag urgency (highest priority) ──────────────────────────────
    red_flags = symptom_set & RED_FLAG_SYMPTOMS
    if red_flags or difficulty_breathing:
        recs.append(
            "\u26a0\ufe0f Red-flag symptoms detected. Please seek medical attention "
            "promptly — do not rely on this app for urgent care decisions."
        )
        if difficulty_breathing:
            recs.append(
                "Difficulty breathing can indicate a serious respiratory or cardiac "
                "condition. Contact a clinician or emergency services immediately."
            )
        if "chest pain" in symptom_set:
            recs.append(
                "Chest pain alongside other symptoms may require urgent cardiac "
                "evaluation. Seek emergency care if pain is severe or spreading."
            )
        if "confusion" in symptom_set or "numbness" in symptom_set:
            recs.append(
                "Neurological symptoms (confusion, numbness) should be evaluated "
                "by a physician without delay."
            )
        if "seizure" in symptom_set or "unconsciousness" in symptom_set:
            recs.append(
                "Loss of consciousness or seizure requires emergency medical attention."
            )

    # ── Cluster pattern context ────────────────────────────────────────────
    # Tell the user which recognisable symptom groups were detected.
    if detected_groups:
        for group_label, group_syms in detected_groups.items():
            if len(group_syms) >= 2:
                syms_str = ", ".join(group_syms)
                recs.append(
                    f"{group_label} symptom pattern detected ({syms_str}). "
                    "This combination may indicate a related underlying condition."
                )

    # ── Severity guidance ─────────────────────────────────────────────────
    if severity is not None and severity >= 8:
        recs.append(
            f"Your self-reported severity is {severity}/10 (high). "
            "Early medical consultation is strongly recommended."
        )
    elif severity is not None and severity >= 6:
        recs.append(
            f"Moderate-to-high severity reported ({severity}/10). "
            "Consider seeing a healthcare professional soon."
        )

    # ── Duration guidance ─────────────────────────────────────────────────
    if days_suffering is not None and days_suffering > 14:
        recs.append(
            f"Symptoms persisting for {days_suffering} days warrant clinical "
            "evaluation; prolonged illness may require investigation."
        )
    elif days_suffering is not None and days_suffering > 7:
        recs.append(
            "Symptoms lasting over a week should be monitored closely. "
            "Consider a healthcare visit if not improving."
        )

    # ── Fever guidance ────────────────────────────────────────────────────
    if fever or "fever" in symptom_set:
        recs.append(
            "Stay well-hydrated and rest. If fever exceeds 39\u00b0C (102\u00b0F) "
            "or persists beyond 3 days, consult a doctor."
        )

    # ── Age-specific guidance ─────────────────────────────────────────────
    if age is not None and (age < 5 or age > 75) and risk in {"moderate", "high"}:
        recs.append(
            f"Given your age ({age}), these symptoms may warrant quicker medical "
            "attention compared with a healthy adult."
        )

    # ── Category hint ─────────────────────────────────────────────────────
    if symptom_category == "Respiratory" and risk in {"moderate", "high"}:
        recs.append(
            "For respiratory symptoms, avoid exposure to smoke, dust, or air "
            "pollutants and consider measuring your oxygen saturation if possible."
        )
    elif symptom_category == "Neurological" and risk in {"moderate", "high"}:
        recs.append(
            "For neurological symptoms, avoid driving or operating machinery "
            "until evaluated by a clinician."
        )

    # ── General risk-level guidance ───────────────────────────────────────
    if not recs or all("disclaimer" in r.lower() or "screening" in r.lower() for r in recs):
        # Only add generic guidance when nothing more specific was generated
        if risk == "high":
            recs.append(
                "Your reported symptoms suggest significant health concern. "
                "Seek a medical evaluation promptly."
            )
        elif risk == "moderate":
            recs.append(
                "Monitor your symptoms closely. Rest and stay hydrated. "
                "Consult a healthcare professional if symptoms worsen or persist."
            )
        else:
            recs.append(
                "Symptoms appear mild. Rest, stay hydrated, and monitor for any changes."
            )
    elif risk == "low" and not any(r.startswith("\u26a0") for r in recs):
        recs.append(
            "Symptoms appear mild. Rest, stay hydrated, and monitor for any changes."
        )
    elif risk == "moderate" and not any(r.startswith("\u26a0") for r in recs):
        recs.append(
            "Monitor symptoms closely. Consult a clinician if they persist or worsen."
        )

    # ── Always-present screening disclaimer ──────────────────────────────
    recs.append(
        "This screening is for informational purposes only and does not "
        "constitute a medical diagnosis. Always consult a qualified healthcare "
        "professional for proper evaluation."
    )

    return recs


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def analyze_symptoms(text: str) -> Dict[str, Any]:
    """Backward-compatible wrapper: analyse free text only.

    Delegates to :func:`analyze_symptoms_structured` so all improvements
    are available even for callers that only pass plain text (e.g. tests,
    legacy integrations).
    """
    return analyze_symptoms_structured(text=text)


def analyze_symptoms_structured(
    text: str = "",
    major_symptom: Optional[str] = None,
    minor_symptoms: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    days_suffering: Optional[int] = None,
    symptom_category: Optional[str] = None,
    fever: bool = False,
    pain: bool = False,
    difficulty_breathing: bool = False,
    severity: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyse symptoms using a hybrid NLP + structured-field engine.

    Parameters
    ----------
    text : str
        Legacy free-text description (used when major/minor fields absent).
    major_symptom : str, optional
        Primary symptom from the form (NLP runs on this first).
    minor_symptoms : str, optional
        Secondary symptoms from the form.
    age, gender : contextual modifiers (modest influence).
    days_suffering : int, optional
        Duration in days — longer duration ↑ screening concern.
    symptom_category : str, optional
        Self-reported category (Respiratory, etc.) used for recommendations.
    fever, pain, difficulty_breathing : bool
        Toggle flags — injected as detected symptoms when not found by NLP.
    severity : int, optional (0–10)
        Self-rated severity — strong influence on symptom_score.

    Returns
    -------
    dict with keys: risk, symptom_score (0-100), confidence, message,
        detected_symptoms, contributing_factors, recommendations, …
    """
    # ── Guard: normalise None inputs ────────────────────────────────────
    text = text or ""
    major_symptom = (major_symptom or "").strip()
    minor_symptoms = (minor_symptoms or "").strip()

    # ── Choose NLP text: prefer structured fields over concatenated label
    # text (which contains "Age: 25. Gender: Male. Major symptom: …" noise).
    nlp_parts: List[str] = []
    if major_symptom:
        nlp_parts.append(major_symptom)
    if minor_symptoms:
        nlp_parts.append(minor_symptoms)

    if nlp_parts:
        nlp_text = ". ".join(nlp_parts)
    elif text.strip():
        nlp_text = text.strip()
    else:
        nlp_text = ""

    # ── Minimum-input guard ──────────────────────────────────────────────
    # Proceed when: meaningful text OR critical flags (difficulty_breathing)
    # or a minimal symptom word is provided.
    has_text = len(nlp_text) >= SYMPTOM_MIN_TEXT_LENGTH
    has_critical_flags = difficulty_breathing or (fever and pain)

    if not has_text and not has_critical_flags:
        return _build_error_result(
            "Symptom description is too short to analyse.",
            {"text_length": len(nlp_text)},
        )

    # ── Preprocessing ────────────────────────────────────────────────────
    if has_text:
        processed_text = _preprocess_text(nlp_text)
    else:
        # Critical flags only — minimal text; still attempt extraction
        processed_text = nlp_text.strip()

    # Lazily load the transformer pipeline on first real call
    _ensure_pipeline()

    # ── A. NLP layer ─────────────────────────────────────────────────────

    severity_mods = _extract_severity_modifiers(processed_text)
    duration_info = _extract_duration(processed_text)

    transformer_score: Optional[float] = None
    model_used = _MODEL_NAME

    if _pipeline is not None and has_text:
        try:
            result = _pipeline(processed_text[:512])[0]
            if result["label"] == "NEGATIVE":
                transformer_score = float(result["score"])
            else:
                transformer_score = 1.0 - float(result["score"])
        except Exception as exc:
            logger.warning("Transformer inference failed: %s", exc)
            model_used = "keyword-rules-fallback"

    # Keyword symptom extraction on the clean NLP text
    detected = _detect_symptoms(processed_text)
    detected_names_set: Set[str] = {s[0] for s in detected}

    # ── B. Structured layer – inject toggle flags ────────────────────────
    # If a toggle was set but text didn't mention it, inject it explicitly
    # so the rule engine can factor it in.  This prevents systematic
    # under-detection when the user provides very short text.
    injected: List[str] = []
    if difficulty_breathing and "shortness of breath" not in detected_names_set:
        detected.append(("shortness of breath", SYMPTOM_DB["shortness of breath"]))
        injected.append("shortness of breath")
    if fever and "fever" not in detected_names_set:
        detected.append(("fever", SYMPTOM_DB["fever"]))
        injected.append("fever")
    # pain toggle: inject "muscle pain" as proxy when no pain-related symptom
    # was found in text.  This avoids generic "pain" matching arbitrary words.
    if pain and not (detected_names_set & _PAIN_SYMPTOM_SET):
        detected.append(("muscle pain", SYMPTOM_DB["muscle pain"]))
        injected.append("muscle pain")

    detected.sort(key=lambda x: x[1], reverse=True)
    detected_names_set = {s[0] for s in detected}
    symptom_names = [s[0] for s in detected]

    # ── Rule-based risk label (NLP-driven, backward-compatible) ──────────
    # The risk LABEL is determined solely by the NLP rule engine so that
    # "fever" → moderate, "chest pain" → high, etc. remain correct and all
    # existing tests continue to pass.  The continuous symptom_score is
    # computed independently using the new penalty model.
    risk, confidence, message = _compute_risk(detected, transformer_score)

    # ── Optional BioBERT escalation ───────────────────────────────────────
    nlp_result = infer_nlp_models(processed_text)
    nlp_comparison = compare_with_distilbert(
        nlp_result,
        distilbert_score=transformer_score,
        keyword_risk=risk,
        keyword_confidence=confidence,
    )
    nlp_model_name = nlp_comparison.get("model_name", "none")
    nlp_source = nlp_comparison.get("source", "distilbert_keywords")
    if nlp_source == "biobert":
        biobert_risk = nlp_comparison.get("risk", "low")
        if _RISK_RANK.get(biobert_risk, 0) > _RISK_RANK.get(risk, 0):
            risk = biobert_risk
            model_conf = float(nlp_comparison.get("confidence", confidence))
            confidence = round(float(np.clip(0.5 * confidence + 0.5 * model_conf, 0.0, 1.0)), 2)

    # Text severity modifiers can only escalate risk (never de-escalate the
    # rule engine, which already handled mild modifiers in the text).
    for mod in severity_mods:
        if mod["effect"] == "escalate":
            target = mod.get("to", "moderate")
            if _RISK_RANK.get(target, 0) > _RISK_RANK.get(risk, 0):
                risk = target

    # ── Structured escalation overrides ──────────────────────────────────
    # difficulty_breathing is an unambiguous red-flag regardless of text.
    if difficulty_breathing and _RISK_RANK.get("high", 0) > _RISK_RANK.get(risk, 0):
        risk = "high"

    # Severity 8+ escalates low → moderate (but does not override red-flag rules).
    if severity is not None and severity >= 8 and risk == "low":
        risk = "moderate"

    # ── Confidence ───────────────────────────────────────────────────────
    evidence_strength = min(1.0, 0.15 + 0.20 * len(detected))
    confidence = round(float(min(confidence, evidence_strength + 0.20)), 2)

    conf_boost = _compute_structured_confidence_boost(
        detected_names=detected_names_set,
        severity=severity,
        days_suffering=days_suffering,
        fever=fever,
        difficulty_breathing=difficulty_breathing,
    )
    confidence = round(min(confidence + conf_boost, 1.0), 2)

    # ── C. Fusion – base_score = 100 minus layered penalties ─────────────
    #
    # Penalty layers (applied in order, each explained separately):
    #  1. Per-symptom penalty   – what symptoms were found (top-N weighted)
    #  2. Cluster penalty       – bonus for recognisable clinical patterns
    #  3. Severity penalty      – how bad the user says it is
    #  4. Duration penalty      – how long symptoms have persisted
    #  5. Age penalty           – contextual modifier for vulnerable age groups
    #
    # Higher score = healthier / lower concern (0–100).
    # Score is NEVER anchored to the NLP risk tier — it is a fully
    # independent continuous signal that coexists with the risk label.

    sym_penalty, _ordered_syms = _compute_symptom_penalty(detected_names_set)
    cluster_pen, detected_groups = _compute_cluster_penalty(detected_names_set)
    sev_pen, sev_factor = _compute_severity_penalty(severity)
    dur_pen, dur_factor = _compute_duration_penalty(days_suffering)
    age_pen, age_factor = _compute_age_penalty(age)

    raw_score = 100.0 - sym_penalty - cluster_pen - sev_pen - dur_pen - age_pen
    symptom_score = int(round(float(np.clip(raw_score, 5, 98))))
    risk_band = _score_to_risk_band(symptom_score)

    # Build human-readable contributing_factors (most impactful listed first).
    contributing_factors: List[str] = []
    _factor_pairs = [
        (sym_penalty,   f"Symptom penalty: {', '.join(_ordered_syms[:3])} "
                        f"({sym_penalty:.0f} pts)"),
        (cluster_pen,   "Symptom cluster pattern" if cluster_pen > 0 else None),
        (sev_pen,       sev_factor),
        (dur_pen,       dur_factor),
        (age_pen,       age_factor),
    ]
    for _val, _label in sorted(_factor_pairs, key=lambda x: x[0], reverse=True):
        if _label and _val > 0:
            contributing_factors.append(_label)

    # Flag-agreement factors (qualitative, not scored separately)
    if fever and "fever" in detected_names_set and "fever" not in injected:
        contributing_factors.append("Fever confirmed (text + flag agreement)")
    if difficulty_breathing and "shortness of breath" in detected_names_set \
            and "shortness of breath" not in injected:
        contributing_factors.append("Difficulty breathing confirmed (text + flag agreement)")

    # Score breakdown for debug / explanation
    score_breakdown = {
        "base_score": 100,
        "symptom_penalty": sym_penalty,
        "cluster_penalty": cluster_pen,
        "severity_penalty": sev_pen,
        "duration_penalty": dur_pen,
        "age_penalty": age_pen,
        "raw_score": round(raw_score, 1),
        "final_score": symptom_score,
    }

    # ── Contextual recommendations ────────────────────────────────────────
    recommendations = _generate_recommendations(
        risk=risk,
        symptom_score=symptom_score,
        detected_symptoms=symptom_names,
        severity=severity,
        days_suffering=days_suffering,
        fever=fever,
        difficulty_breathing=difficulty_breathing,
        age=age,
        detected_groups=detected_groups,
        symptom_category=symptom_category,
    )

    return {
        "module_name": "symptom_module",
        "metric_name": "symptom_risk",
        "value": risk,
        "unit": "label",
        "confidence": confidence,
        "risk": risk,
        "risk_band": risk_band,
        "symptom_score": symptom_score,
        "message": message,
        "recommendations": recommendations,
        "detected_symptoms": symptom_names,
        "detected_groups": detected_groups,
        "contributing_factors": contributing_factors,
        "injected_from_flags": injected,
        "severity_modifiers": severity_mods,
        "duration_info": duration_info,
        "structured_inputs": {
            "age": age,
            "gender": gender,
            "days_suffering": days_suffering,
            "symptom_category": symptom_category,
            "severity": severity,
            "fever": fever,
            "pain": pain,
            "difficulty_breathing": difficulty_breathing,
        },
        "debug": {
            "text_length": len(nlp_text),
            "model_used": model_used,
            "transformer_negative_score": transformer_score,
            "keyword_matches": len(detected),
            "score_breakdown": score_breakdown,
            "nlp_model": {
                "model_name": nlp_model_name,
                "source": nlp_source,
                "selection_reason": nlp_comparison.get("selection_reason", ""),
                "available_models": nlp_result.get("available_models", []),
                "model_available": nlp_comparison.get("model_available", False),
                "model_loaded": nlp_comparison.get("model_loaded", False),
                "model_cached": nlp_comparison.get("model_cached", False),
                "inference_source": nlp_comparison.get("inference_source", "keyword_rules"),
            },
        },
    }
