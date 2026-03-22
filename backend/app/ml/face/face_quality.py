"""
Vita AI – Face Quality Scoring
================================
Frame-level and scan-level quality assessment used for:

* Gating: reject scans that are too noisy for reliable results.
* Confidence: build an explainable, factor-by-factor confidence score.
* Retake guidance: tell the user *why* the scan was poor.

All scores are 0-1, where 1 = excellent quality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame quality
# ═══════════════════════════════════════════════════════════════════════════

def frame_quality_score(
    face_detected: bool,
    brightness: float,
    motion_magnitude: float,
    brightness_quality: float,
) -> float:
    """Compute a single 0-1 quality score for one frame.

    Factors: face visibility, lighting quality, motion penalty.
    """
    if not face_detected:
        return 0.0

    # Motion penalty: gentle curve, 0 motion → 1.0, large motion → ~0.0
    motion_penalty = 1.0 / (1.0 + motion_magnitude * 0.5)

    return round(brightness_quality * motion_penalty, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Scan-level quality aggregation
# ═══════════════════════════════════════════════════════════════════════════

def compute_scan_quality(
    frame_qualities: List[float],
    tracking_ratio: float,
    motion_scores: List[float],
    brightness_scores: List[float],
    roi_agreement: float,
    signal_periodicity: float,
    scan_duration_sec: float,
    min_duration_sec: float = 10.0,
    overexposure_ratios: Optional[List[float]] = None,
    usable_frame_count: Optional[int] = None,
    total_frame_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Build an explainable quality breakdown for the whole scan.

    Returns
    -------
    {
        "scan_quality": 0-1 overall,
        "confidence_breakdown": {
            "tracking": ...,
            "lighting": ...,
            "motion": ...,
            "signal_periodicity": ...,
            "roi_agreement": ...,
            "scan_completeness": ...,
            "exposure_stability": ...,
            "overexposure": ...,
            "usable_frames": ...,
        },
        "retake_required": bool,
        "retake_reasons": [...],
    }
    """
    import numpy as np

    # --- Individual factors ---
    tracking_score = float(np.clip(tracking_ratio, 0, 1))

    lighting_score = float(np.mean(brightness_scores)) if brightness_scores else 0.5
    lighting_score = float(np.clip(lighting_score, 0, 1))

    if motion_scores:
        raw_motion = float(np.mean(motion_scores))
        # Invert: low motion = high quality
        motion_quality = float(np.clip(1.0 / (1.0 + raw_motion * 0.3), 0, 1))
    else:
        motion_quality = 0.5

    periodicity = float(np.clip(signal_periodicity, 0, 1))
    roi_agr = float(np.clip(roi_agreement, 0, 1))

    completeness = float(np.clip(scan_duration_sec / min_duration_sec, 0, 1))

    # --- Exposure stability (variance of brightness scores) ---
    if brightness_scores and len(brightness_scores) >= 4:
        b_arr = np.array(brightness_scores)
        b_std = float(np.std(b_arr))
        # Low variance = stable exposure.  std 0 → 1.0, std 0.3 → ~0.0
        exposure_stability = float(np.clip(1.0 - b_std / 0.3, 0, 1))
    else:
        exposure_stability = 0.5

    # --- Overexposure penalty ---
    if overexposure_ratios and len(overexposure_ratios) > 0:
        mean_oe = float(np.mean(overexposure_ratios))
        # 0% overexposure → 1.0, 30%+ → 0.0
        overexposure_score = float(np.clip(1.0 - mean_oe / 0.30, 0, 1))
    else:
        overexposure_score = 1.0

    # --- Usable frames ratio ---
    if usable_frame_count is not None and total_frame_count and total_frame_count > 0:
        usable_ratio = usable_frame_count / total_frame_count
        usable_score = float(np.clip(usable_ratio / 0.6, 0, 1))  # 60%+ usable → 1.0
    else:
        usable_score = tracking_score  # fallback

    # --- Weighted combination ---
    weights = {
        "tracking": 0.14,
        "lighting": 0.10,
        "motion": 0.14,
        "signal_periodicity": 0.20,
        "roi_agreement": 0.08,
        "scan_completeness": 0.06,
        "exposure_stability": 0.10,
        "overexposure": 0.10,
        "usable_frames": 0.08,
    }
    breakdown = {
        "tracking": round(tracking_score, 3),
        "lighting": round(lighting_score, 3),
        "motion": round(motion_quality, 3),
        "signal_periodicity": round(periodicity, 3),
        "roi_agreement": round(roi_agr, 3),
        "scan_completeness": round(completeness, 3),
        "exposure_stability": round(exposure_stability, 3),
        "overexposure": round(overexposure_score, 3),
        "usable_frames": round(usable_score, 3),
    }
    overall = sum(weights[k] * breakdown[k] for k in weights)
    overall = round(float(np.clip(overall, 0, 1)), 3)

    # --- Retake logic ---
    retake_required = False
    retake_reasons: List[str] = []

    if tracking_score < 0.5:
        retake_required = True
        retake_reasons.append("Face not detected in enough frames. Keep your face centered and visible.")
    if lighting_score < 0.4:
        retake_required = True
        retake_reasons.append("Lighting quality is too low. Move to a well-lit area and avoid strong shadows.")
    if motion_quality < 0.35:
        retake_required = True
        retake_reasons.append("Too much head movement detected. Please keep your head still during the scan.")
    if periodicity < 0.2:
        retake_required = True
        retake_reasons.append("Could not detect a clear pulse signal. Ensure good lighting on your face and retry for 30 seconds.")
    if completeness < 0.5:
        retake_required = True
        retake_reasons.append(f"Scan too short ({scan_duration_sec:.0f}s). Please hold still for at least {min_duration_sec:.0f} seconds.")
    if overexposure_score < 0.3:
        retake_required = True
        retake_reasons.append("Too much brightness washout detected. Reduce direct light on your face or move to diffuse lighting.")
    if exposure_stability < 0.25:
        retake_required = True
        retake_reasons.append("Unstable lighting detected. Avoid flickering lights and keep a steady light source.")
    if usable_score < 0.3:
        retake_required = True
        retake_reasons.append("Too few usable frames. Ensure good lighting, minimal movement, and scan for at least 20 seconds.")

    # Note: retake_required is communicated via the retake_required flag and
    # retake_reasons list. The scan_quality score reflects the true computed
    # quality and must NOT be capped to the threshold — doing so produces a
    # constant value that hides real variation.

    return {
        "scan_quality": overall,
        "confidence_breakdown": breakdown,
        "retake_required": retake_required,
        "retake_reasons": retake_reasons,
    }


def compute_roi_agreement(roi_bpms: Dict[str, float]) -> float:
    """Measure how well different ROIs agree on heart rate.

    If all ROIs give similar BPM, agreement → 1.0.
    High variance → low agreement → low confidence.
    """
    import numpy as np
    valid = [b for b in roi_bpms.values() if b > 0]
    if len(valid) < 2:
        return 0.5  # can't assess agreement with < 2 ROIs
    std = float(np.std(valid))
    # 0 std → 1.0; 20 bpm std → ~0.0
    return float(np.clip(1.0 - std / 20.0, 0, 1))
