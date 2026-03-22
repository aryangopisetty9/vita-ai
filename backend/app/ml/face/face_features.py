"""
Vita AI – Face Features
=========================
Landmark-derived behavioral and appearance features:

* **Blink detection** – EAR-based blink counting and rate.
* **Eye movement / stability** – iris-to-eye-centre displacement.
* **Facial muscle tension** – region-level landmark displacement trends.
* **Skin signal analysis** – redness/pallor proxies from cheek ROIs.

All features are *screening-level indicators only*.  They do NOT
constitute medical diagnoses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.ml.face.vision_utils import (
    LEFT_EYE,
    LEFT_IRIS,
    RIGHT_EYE,
    RIGHT_IRIS,
    REGION_LANDMARKS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Eye Aspect Ratio (EAR) & blink detection
# ═══════════════════════════════════════════════════════════════════════════

EAR_BLINK_THRESHOLD = 0.21
EAR_CONSEC_FRAMES = 2


def compute_ear(lm_list, indices: List[int], h: int, w: int) -> float:
    """6-point Eye Aspect Ratio for one eye."""
    pts = [(lm_list[i].x * w, lm_list[i].y * h) for i in indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hor = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if hor < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * hor))


def avg_ear(lm_list, h: int, w: int) -> float:
    """Average EAR of both eyes."""
    return (compute_ear(lm_list, LEFT_EYE, h, w) +
            compute_ear(lm_list, RIGHT_EYE, h, w)) / 2.0


class BlinkDetector:
    """Stateful blink counter using EAR thresholding."""

    def __init__(self, threshold: float = EAR_BLINK_THRESHOLD,
                 consec: int = EAR_CONSEC_FRAMES):
        self.threshold = threshold
        self.consec = consec
        self.blink_count = 0
        self._streak = 0

    def update(self, ear_value: float) -> None:
        if ear_value < self.threshold:
            self._streak += 1
        else:
            if self._streak >= self.consec:
                self.blink_count += 1
            self._streak = 0


def aggregate_blink_results(
    ear_values: List[float],
    blink_count: int,
    duration_sec: float,
) -> Dict[str, Any]:
    """Build the blink-analysis sub-dict for the face result."""
    if not ear_values:
        return {}
    blinks_per_min = round(blink_count / max(duration_sec / 60.0, 0.01), 1)
    return {
        "blink_count": blink_count,
        "blink_rate_per_min": blinks_per_min,
        "avg_ear": round(float(np.mean(ear_values)), 4),
        "min_ear": round(float(np.min(ear_values)), 4),
        "ear_std": round(float(np.std(ear_values)), 4),
        "normal_range_note": "Typical adult blink rate: 15-20 blinks/min.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Eye movement / gaze stability
# ═══════════════════════════════════════════════════════════════════════════

def iris_center(lm_list, iris_indices: List[int], h: int, w: int) -> Optional[Tuple[float, float]]:
    """(x, y) pixel coords of iris centre.  Needs refine_landmarks."""
    if iris_indices[0] >= len(lm_list):
        return None
    cx = float(np.mean([lm_list[i].x for i in iris_indices])) * w
    cy = float(np.mean([lm_list[i].y for i in iris_indices])) * h
    return (cx, cy)


def eye_center(lm_list, eye_indices: List[int], h: int, w: int) -> Tuple[float, float]:
    """Geometric centre of the eye outline."""
    cx = float(np.mean([lm_list[i].x for i in eye_indices])) * w
    cy = float(np.mean([lm_list[i].y for i in eye_indices])) * h
    return (cx, cy)


def compute_gaze_offset(lm_list, h: int, w: int) -> Optional[Tuple[float, float]]:
    """Average iris-to-eye-centre displacement (dx, dy) in pixels."""
    l_iris = iris_center(lm_list, LEFT_IRIS, h, w)
    r_iris = iris_center(lm_list, RIGHT_IRIS, h, w)
    if l_iris is None or r_iris is None:
        return None
    l_eye = eye_center(lm_list, LEFT_EYE, h, w)
    r_eye = eye_center(lm_list, RIGHT_EYE, h, w)
    avg_dx = ((l_iris[0] - l_eye[0]) + (r_iris[0] - r_eye[0])) / 2.0
    avg_dy = ((l_iris[1] - l_eye[1]) + (r_iris[1] - r_eye[1])) / 2.0
    return (avg_dx, avg_dy)


def aggregate_eye_movement(
    gaze_offsets: List[Tuple[float, float]],
    duration_sec: float,
) -> Dict[str, Any]:
    """Build eye-movement sub-dict."""
    if not gaze_offsets:
        return {}
    offsets = np.array(gaze_offsets)
    diffs = np.diff(offsets, axis=0)
    total = float(np.sum(np.linalg.norm(diffs, axis=1))) if len(diffs) > 0 else 0.0

    # Stability = inverse of variability (high std = low stability)
    std_dx = float(np.std(offsets[:, 0]))
    std_dy = float(np.std(offsets[:, 1]))
    raw_variability = (std_dx + std_dy) / 2.0
    eye_stability = float(np.clip(1.0 / (1.0 + raw_variability * 0.3), 0, 1))

    return {
        "total_gaze_displacement_px": round(total, 1),
        "avg_gaze_offset": {
            "dx": round(float(np.mean(offsets[:, 0])), 2),
            "dy": round(float(np.mean(offsets[:, 1])), 2),
        },
        "gaze_variability": {"std_dx": round(std_dx, 2), "std_dy": round(std_dy, 2)},
        "movement_intensity": round(total / max(duration_sec, 0.01), 1),
        "eye_stability": round(eye_stability, 3),
        "note": "Higher displacement = more eye movement. "
                "eye_stability is a 0-1 screening proxy (1 = very stable).",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Facial muscle-motion / tension
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_facial_motion(
    region_positions: Dict[str, List[np.ndarray]],
) -> Dict[str, Any]:
    """Compute per-region and overall facial motion intensity.

    Returns a ``facial_tension_index`` as a rough screening proxy —
    this is NOT a clinical stress assessment.
    """
    if not any(len(v) > 1 for v in region_positions.values()):
        return {}

    region_scores: Dict[str, float] = {}
    for region, positions_list in region_positions.items():
        if len(positions_list) < 2:
            region_scores[region] = 0.0
            continue
        displacements = []
        for i in range(1, len(positions_list)):
            diff = positions_list[i] - positions_list[i - 1]
            displacements.append(float(np.mean(np.linalg.norm(diff, axis=1))))
        region_scores[region] = round(float(np.mean(displacements)), 3)

    overall_motion = round(float(np.mean(list(region_scores.values()))), 3)

    # Tension index: brows + mouth motion relative to overall
    brow_motion = (region_scores.get("left_eyebrow", 0) +
                   region_scores.get("right_eyebrow", 0)) / 2.0
    mouth_motion = region_scores.get("mouth", 0)
    # Normalise to 0-1 (higher = more tension-like activity)
    tension_raw = (brow_motion + mouth_motion) / 2.0
    tension_index = float(np.clip(tension_raw / max(overall_motion + 0.01, 0.5), 0, 1))

    return {
        "region_motion_scores": region_scores,
        "overall_motion_intensity": overall_motion,
        "facial_tension_index": round(tension_index, 3),
        "unit": "avg_px_displacement_per_frame",
        "note": "facial_tension_index is a screening-level proxy. "
                "It is NOT a clinical stress measurement.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Skin colour / complexion analysis
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_skin_analysis(
    skin_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyse skin-colour stability and compute safe screening proxies.

    Outputs are framed as *weak supportive indicators* only.
    No ethnicity / identity inferences are made.
    """
    if not skin_samples:
        return {}

    rgbs = np.array([s["rgb"] for s in skin_samples], dtype=np.float64)
    hsvs = np.array([s["hsv"] for s in skin_samples], dtype=np.float64)

    avg_rgb = np.mean(rgbs, axis=0).astype(int).tolist()
    avg_hsv = np.mean(hsvs, axis=0).astype(int).tolist()

    # Colour consistency (low std → stable signal)
    rgb_std = float(np.mean(np.std(rgbs, axis=0)))
    stability = float(np.clip(1.0 / (1.0 + rgb_std * 0.05), 0, 1))

    # Redness proxy: (R - G) / R — higher in flushed / inflamed skin
    mean_r, mean_g = float(avg_rgb[0]), float(avg_rgb[1])
    if mean_r > 10:
        redness_proxy = round((mean_r - mean_g) / mean_r, 3)
    else:
        redness_proxy = 0.0

    # Pallor proxy: very high V + very low S in HSV → washed-out / pale
    mean_s = float(avg_hsv[1])
    mean_v = float(avg_hsv[2])
    # Low saturation + high value → pale
    pallor_proxy = round(float(np.clip((1.0 - mean_s / 255.0) * (mean_v / 255.0), 0, 1)), 3)

    return {
        "avg_rgb": avg_rgb,
        "avg_hsv": avg_hsv,
        "skin_signal_stability": round(stability, 3),
        "redness_proxy": redness_proxy,
        "pallor_proxy_low_confidence": pallor_proxy,
        "samples_used": len(skin_samples),
        "circulation_signal_quality": round(stability, 3),
        "note": "These are weak supportive screening indicators only. "
                "They do NOT diagnose any condition. "
                "Values are influenced by ambient lighting.",
    }


def aggregate_eye_color(
    eye_color_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate iris colour samples."""
    if not eye_color_samples:
        return {}
    avg_rgb = np.mean([s["rgb"] for s in eye_color_samples], axis=0).astype(int).tolist()
    avg_hsv = np.mean([s["hsv"] for s in eye_color_samples], axis=0).astype(int).tolist()
    return {
        "avg_rgb": avg_rgb,
        "avg_hsv": avg_hsv,
        "samples_used": len(eye_color_samples),
        "note": "Sampled from iris region. Accuracy depends on resolution and lighting.",
    }
