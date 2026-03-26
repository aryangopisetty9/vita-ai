"""
Vita AI – Face Module (Robust Pipeline)
=========================================
30-second face-scan pipeline that extracts:

1. **Heart rate** – multi-ROI rPPG with detrending, temporal
   normalisation, bandpass filtering, FFT peak detection, and
   quality-weighted ROI fusion.  Produces both a summary BPM and a
   sliding-window time-series for graph rendering.
2. **Blink detection** – EAR-based, with blink count and rate.
3. **Eye movement / gaze stability** – iris displacement tracking.
4. **Facial muscle tension** – per-region landmark displacement.
5. **Eye colour** – iris colour sampling.
6. **Skin colour & signal stability** – cheek-ROI colour analysis
   with redness/pallor screening proxies.
7. **Explainable confidence** – factor-by-factor breakdown.
8. **Quality gating & retake logic** – automatic retake
   recommendation with specific guidance when scan quality is low.

Robustness features
-------------------
* Multi-ROI extraction (forehead + left cheek + right cheek).
* Per-frame quality scoring (brightness + motion + face visibility).
* Motion-aware frame rejection / down-weighting.
* Lighting-aware normalisation and frame rejection.
* Signal periodicity checks.
* ROI-agreement cross-validation.
* Graceful degradation: partial results are still returned when
  some features fail.

Extension points
----------------
* Replace ``estimate_heart_rate_multi_roi`` with a deep-learning
  rPPG model (Open-rPPG).
* Add SpO₂ estimation with multi-wavelength camera input.
* Plug in a learned stress/fatigue model using EAR + blink + tension
  features as inputs.
* Replace ``cv2.VideoCapture(path)`` with a live frame iterator for
  real-time mobile streaming.

Disclaimer
----------
This module produces **preliminary screening estimates** only.
It is **not** a medical diagnostic system.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not installed – face module will use fallback mode.")

try:
    import mediapipe as mp  # type: ignore
    _HAS_MP = True
except ImportError:
    _HAS_MP = False
    logger.warning("MediaPipe not installed – advanced facial analytics unavailable.")

_FACE_LANDMARKER_MODEL = (
    Path(__file__).resolve().parents[4] / "models_cache" / "mediapipe" / "face_landmarker.task"
)

from backend.app.core.config import (
    FACE_MIN_FRAMES,
    RPPG_HIGH_HZ,
    RPPG_LOW_HZ,
    RPPG_NORMAL_HIGH,
    RPPG_NORMAL_LOW,
)

# Sub-modules
from backend.app.ml.face.face_features import (
    BlinkDetector,
    aggregate_blink_results,
    aggregate_eye_color,
    aggregate_eye_movement,
    aggregate_facial_motion,
    aggregate_skin_analysis,
    avg_ear,
    compute_gaze_offset,
)
from backend.app.ml.face.face_quality import (
    compute_roi_agreement,
    compute_scan_quality,
)
from backend.app.ml.face.rppg_utils import (
    ROI_NAMES,
    estimate_heart_rate_multi_roi,
    extract_frame_roi_signals,
    extract_frame_roi_rgb,
    extract_frame_overexposure,
)
from backend.app.ml.face.vision_utils import (
    LEFT_CHEEK,
    LEFT_IRIS,
    REGION_LANDMARKS,
    RIGHT_CHEEK,
    apply_clahe,
    brightness_quality as _brightness_quality,
    denoise_frame,
    extract_anchor_positions,
    extract_face_roi_haar,
    frame_blur_metrics,
    frame_brightness,
    frame_overexposure_ratio,
    gamma_correct,
    landmark_positions,
    sample_iris_color,
    sample_polygon_color,
    LandmarkSmoother,
    compute_roi_quality_metrics,
    extract_forehead_roi,
    extract_cheek_roi,
)
from backend.app.utils.signal_processing import (
    compute_hr_timeseries,
    median_filter_bpm,
    robust_hr_consensus,
)
from backend.app.ml.face.rppg_models import (
    compare_with_signal_pipeline,
    infer_rppg_models,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_COLOR_SAMPLE_INTERVAL = 10   # sample iris / skin colour every N frames
_MOTION_REJECT_THRESHOLD = 15.0  # px – reject frame if head motion > this
_HR_WINDOW_SEC = 5.0
_HR_STRIDE_SEC = 0.5
_MAX_MODEL_FRAMES = 300  # max frames to keep for model inference
_MIN_CONSENSUS_WINDOWS = 2  # lowered from 4 — web/mobile cameras often produce only 2-4 quality windows

# Conservative real-world tuning for average webcams/phones.
_LOW_BRIGHTNESS_MIN_QUALITY = 0.06
_BLUR_MIN_QUALITY = 0.05
_ROI_DRIFT_REJECT_PX = 22.0
_FRAME_OVEREXPOSURE_REJECT = 0.68
_MIN_TEMPORAL_COVERAGE_SEC = 4.5
_MAX_TEMPORAL_COVERAGE_SEC = 16.0
_TEMPORAL_COVERAGE_RATIO = 0.28
_UNSTABLE_SAMPLING_JITTER_CV = 0.40
_UNSTABLE_SAMPLING_MIN_EFF_FPS = 5.5
_FINAL_MIN_SIGNAL_PERIODICITY = 0.17
_FINAL_MIN_SIGNAL_STRENGTH = 0.07
_FINAL_MIN_VALID_WINDOWS = 2
_STRONG_MIN_PERIODICITY = 0.22
_STRONG_MIN_SIGNAL_STRENGTH = 0.10
_STRONG_MIN_VALID_WINDOWS = 3
_STRONG_MAX_STD_DEV = 11.0
_WEAK_MIN_PERIODICITY = 0.08
_WEAK_MIN_SIGNAL_STRENGTH = 0.03
_WEAK_MIN_VALID_WINDOWS = 1
_WEAK_MAX_STD_DEV = 22.0
_SEVERE_JITTER_CV = 0.65
_SEVERE_MIN_EFF_FPS = 4.0
_CATASTROPHIC_MOTION_MULT = 1.15
_CATASTROPHIC_LOW_BRIGHTNESS = 0.18
_CATASTROPHIC_OVEREXPOSURE = 0.75

# Diagnostics export controls.
_DIAG_EXPORT_ENV = "VITA_FACE_DIAGNOSTICS_EXPORT"
_DIAG_EXPORT_PATH_ENV = "VITA_FACE_DIAGNOSTICS_PATH"
_MIN_VALID_FPS = 8.0
_MAX_VALID_FPS = 60.0
_DEFAULT_FPS = 30.0
MIN_ANALYSIS_FPS = 12.0
LOW_FPS_SAFE_DEFAULT = 15.0

# Cross-run smoothing cache (process-local).
# Keep each scan independent: no state is carried across runs.


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_error_result(message: str, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a well-formed error result so callers never crash."""
    return {
        "module_name": "face_module",
        "scan_duration_sec": 0,
        "heart_rate": None,
        "heart_rate_unit": "bpm",
        "heart_rate_confidence": 0.0,
        "blink_rate": None,
        "blink_rate_unit": "blinks/min",
        "eye_stability": None,
        "facial_tension_index": None,
        "skin_signal_stability": None,
        "scan_quality": 0.0,
        "retake_required": True,
        "retake_reasons": [message],
        "message": message,
        "risk": "error",
        "confidence": 0.0,
        "hr_confidence": 0.0,
        "reliability": "unreliable",
        "hr_result_tier": "result_unavailable",
        "result_available": False,
        "retake_recommended": True,
        "estimated_from_weak_signal": False,
        "signal_strength": 0.0,
        "periodicity_score": 0.0,
        "valid_windows": 0,
        "method_used": "none",
        "warning": message,
        "debug": debug or {},
        # Extended sub-dicts
        "hr_timeseries": [],
        "blink_analysis": {},
        "eye_movement": {},
        "facial_motion": {},
        "eye_color": {},
        "skin_color": {},
        "confidence_breakdown": {},
        # Legacy compat
        "metric_name": "heart_rate",
        "value": None,
        "unit": "bpm",
    }


def _diag_export_enabled() -> bool:
    return os.getenv(_DIAG_EXPORT_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _diag_export_path() -> Path:
    raw = os.getenv(_DIAG_EXPORT_PATH_ENV, "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[4] / "temp" / "face_scan_diagnostics.jsonl"


def _stringify_map(value: Any) -> str:
    if isinstance(value, dict):
        try:
            return json.dumps(value, separators=(",", ":"), sort_keys=True)
        except Exception:
            return str(value)
    return "{}"


def _select_chosen_candidate(result: Dict[str, Any], debug: Dict[str, Any]) -> Dict[str, Any]:
    selected_roi = debug.get("selected_roi")
    selected_method = debug.get("selected_method")
    candidates = debug.get("all_candidate_scores") or []
    if isinstance(candidates, list):
        for c in candidates:
            if not isinstance(c, dict):
                continue
            if c.get("roi") == selected_roi and c.get("method") == selected_method:
                return c
    return {}


def _build_diag_record(result: Dict[str, Any], source: str) -> Dict[str, Any]:
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    timing = debug.get("timing") if isinstance(debug.get("timing"), dict) else {}
    chosen = _select_chosen_candidate(result, debug)
    hr_reasons = result.get("hr_rejection_reasons")
    if not isinstance(hr_reasons, list):
        hr_reasons = []

    acceptance = "accepted" if (result.get("heart_rate") is not None) else "rejected"
    return {
        "ts_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": source,
        "final_status": acceptance,
        "final_acceptance": debug.get("final_acceptance_reason", ""),
        "heart_rate": result.get("heart_rate"),
        "frame_count": debug.get("frames_processed", 0),
        "usable_frame_count": debug.get("usable_frame_count", debug.get("valid_frames", 0)),
        "effective_fps": timing.get("effective_fps", debug.get("effective_fps", 0.0)),
        "cadence_jitter_cv": timing.get("cadence_jitter_cv", debug.get("cadence_jitter_cv", 0.0)),
        "dropped_frame_estimate": timing.get("estimated_dropped_frames", 0),
        "low_brightness_reject_count": debug.get("low_brightness_reject_count", 0),
        "blur_reject_count": debug.get("blur_reject_count", 0),
        "motion_reject_count": debug.get("motion_reject_count", 0),
        "roi_instability_reject_count": debug.get("roi_instability_reject_count", 0),
        "roi_adaptive_weights": _stringify_map(debug.get("roi_adaptive_weights", {})),
        "method_adaptive_weights": _stringify_map(debug.get("method_adaptive_weights", {})),
        "candidate_periodicity": chosen.get("periodicity", result.get("periodicity_score", 0.0)),
        "candidate_signal_strength": chosen.get("signal_strength", result.get("signal_strength", 0.0)),
        "valid_windows": result.get("valid_windows", debug.get("valid_window_count", 0)),
        "window_std": debug.get("std_dev", 0.0),
        "chosen_roi": debug.get("selected_roi", ""),
        "chosen_method": debug.get("selected_method", result.get("method_used", "")),
        "final_reliability": result.get("reliability", "unreliable"),
        "hr_result_tier": result.get("hr_result_tier", "result_unavailable"),
        "result_available": bool(result.get("result_available", False)),
        "estimated_from_weak_signal": bool(result.get("estimated_from_weak_signal", False)),
        "retake_recommended": bool(result.get("retake_recommended", result.get("retake_required", False))),
        "hr_rejection_reasons": json.dumps(hr_reasons),
    }


def _export_diag_record(record: Dict[str, Any]) -> None:
    if not _diag_export_enabled():
        return
    out_path = _diag_export_path()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            fieldnames = [
                "ts_utc",
                "source",
                "final_status",
                "final_acceptance",
                "heart_rate",
                "frame_count",
                "usable_frame_count",
                "effective_fps",
                "cadence_jitter_cv",
                "dropped_frame_estimate",
                "low_brightness_reject_count",
                "blur_reject_count",
                "motion_reject_count",
                "roi_instability_reject_count",
                "roi_adaptive_weights",
                "method_adaptive_weights",
                "candidate_periodicity",
                "candidate_signal_strength",
                "valid_windows",
                "window_std",
                "chosen_roi",
                "chosen_method",
                "final_reliability",
                "hr_result_tier",
                "result_available",
                "estimated_from_weak_signal",
                "retake_recommended",
                "hr_rejection_reasons",
            ]
            exists = out_path.exists() and out_path.stat().st_size > 0
            with out_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow({k: record.get(k, "") for k in fieldnames})
        else:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:
        logger.debug("Face diagnostics export failed: %s", exc)


def _emit_result(result: Dict[str, Any], source: str) -> Dict[str, Any]:
    _export_diag_record(_build_diag_record(result, source=source))
    return result


def _classify_hr(bpm: float) -> Tuple[str, str]:
    """Map BPM to a risk label and human-readable message."""
    if bpm < RPPG_NORMAL_LOW:
        return "moderate", (
            f"Estimated heart rate ({bpm:.0f} bpm) is below typical resting range. "
            "This is a preliminary screening estimate — consult a doctor if concerned."
        )
    if bpm > RPPG_NORMAL_HIGH:
        return "moderate", (
            f"Estimated heart rate ({bpm:.0f} bpm) is above typical resting range. "
            "This is a preliminary screening estimate — consult a doctor if concerned."
        )
    return "low", (
        f"Estimated heart rate ({bpm:.0f} bpm) is within normal resting range."
    )


def _derive_reliability(
    confidence: float,
    periodicity: float,
    valid_windows: int,
    has_signal: bool,
) -> str:
    """Map signal evidence to reliability label."""
    if not has_signal:
        return "unreliable"
    if confidence >= 0.55 and periodicity >= 0.35 and valid_windows >= 3:
        return "high"
    if confidence >= 0.25 and periodicity >= 0.15 and valid_windows >= 2:
        return "medium"
    return "low"


def _evaluate_hr_result_tier(
    *,
    bpm: Optional[float],
    periodicity: float,
    signal_strength: float,
    valid_windows: int,
    std_dev: float,
    peak_support_count: int,
    selected_roi: Optional[str],
    timing_diag: Dict[str, Any],
    avg_motion: float,
    mean_brightness: float,
    mean_overexposure: float,
) -> Dict[str, Any]:
    """Classify HR outcome into result_available or result_unavailable."""
    if bpm is None or bpm <= 0:
        return {
            "tier": "result_unavailable",
            "result_available": False,
            "estimated_from_weak_signal": False,
            "reasons": ["no_viable_candidate"],
        }

    jitter_cv = float(timing_diag.get("cadence_jitter_cv", 0.0) or 0.0)
    eff_fps = float(timing_diag.get("effective_fps", 0.0) or 0.0)
    severe_sampling = bool(
        timing_diag.get("unstable_sampling", False)
        and (jitter_cv > _SEVERE_JITTER_CV or eff_fps < _SEVERE_MIN_EFF_FPS)
    )
    catastrophic_motion = avg_motion > (_MOTION_REJECT_THRESHOLD * _CATASTROPHIC_MOTION_MULT)
    catastrophic_lighting = mean_brightness < _CATASTROPHIC_LOW_BRIGHTNESS
    catastrophic_exposure = mean_overexposure > _CATASTROPHIC_OVEREXPOSURE
    catastrophic_scene = catastrophic_motion or catastrophic_lighting or catastrophic_exposure

    # Availability-first policy: if we have a current-scan candidate BPM,
    # only hard-fail on clearly catastrophic technical conditions.
    has_consistent_candidate = bool(
        selected_roi is not None
        or valid_windows >= 1
        or peak_support_count >= 1
        or (periodicity >= 0.02 and signal_strength >= 0.02)
    )
    minimal_signal = bool(periodicity >= 0.02 or signal_strength >= 0.02)

    strong_evidence = bool(
        has_consistent_candidate
        and periodicity >= _STRONG_MIN_PERIODICITY
        and signal_strength >= _STRONG_MIN_SIGNAL_STRENGTH
        and valid_windows >= _STRONG_MIN_VALID_WINDOWS
        and std_dev <= _STRONG_MAX_STD_DEV
        and not timing_diag.get("unstable_sampling", False)
        and peak_support_count >= 2
    )
    hard_failure = bool(
        severe_sampling
        or catastrophic_scene
        or (not has_consistent_candidate)
        or (not minimal_signal and valid_windows <= 0 and peak_support_count <= 0)
    )

    if not hard_failure:
        weak_flags: List[str] = []
        if not strong_evidence:
            weak_flags.append("retake_recommended")
            if periodicity < _STRONG_MIN_PERIODICITY:
                weak_flags.append("borderline_periodicity")
            if signal_strength < _STRONG_MIN_SIGNAL_STRENGTH:
                weak_flags.append("weak_signal")
            if valid_windows < _STRONG_MIN_VALID_WINDOWS:
                weak_flags.append("limited_windows")
            if timing_diag.get("unstable_sampling", False):
                weak_flags.append("unstable_sampling")
        return {
            "tier": "result_available",
            "result_available": True,
            "estimated_from_weak_signal": not strong_evidence,
            "reasons": weak_flags,
        }

    reject_reasons: List[str] = []
    if severe_sampling:
        reject_reasons.append("extreme_unstable_sampling")
    if catastrophic_motion:
        reject_reasons.append("severe_motion")
    if catastrophic_lighting:
        reject_reasons.append("catastrophic_low_light")
    if catastrophic_exposure:
        reject_reasons.append("catastrophic_overexposure")
    if not has_consistent_candidate:
        reject_reasons.append("no_consistent_candidate")
    if periodicity < _WEAK_MIN_PERIODICITY:
        reject_reasons.append("weak_periodicity")
    if signal_strength < _WEAK_MIN_SIGNAL_STRENGTH:
        reject_reasons.append("low_snr")
    if valid_windows < _WEAK_MIN_VALID_WINDOWS:
        reject_reasons.append("insufficient_windows")
    if std_dev > _WEAK_MAX_STD_DEV:
        reject_reasons.append("inconsistent_windows")
    if not reject_reasons:
        reject_reasons.append("insufficient_signal")
    return {
        "tier": "result_unavailable",
        "result_available": False,
        "estimated_from_weak_signal": False,
        "reasons": reject_reasons,
    }


def _stability_label(std_dev: float) -> str:
    """Map HR variability to a user-facing stability label."""
    if std_dev < 3.0:
        return "High"
    if std_dev < 7.0:
        return "Moderate"
    return "Low"


def _build_spatially_averaged_trace(roi_traces: Dict[str, List[float]]) -> List[float]:
    """Average ROI signals per-frame to suppress local noise."""
    available = [trace for trace in roi_traces.values() if len(trace) >= 8]
    if not available:
        return []
    min_len = min(len(trace) for trace in available)
    if min_len < 8:
        return []
    stacked = np.vstack([np.array(trace[-min_len:], dtype=np.float64) for trace in available])
    # Median across ROIs is robust to one noisy ROI.
    merged = np.median(stacked, axis=0)
    return merged.tolist()


def _compute_timing_diagnostics(
    timestamps: List[float],
    nominal_fps: float,
) -> Dict[str, Any]:
    """Summarise temporal cadence stability for timestamp-aware processing."""
    if len(timestamps) < 2:
        return {
            "coverage_sec": 0.0,
            "effective_fps": round(float(max(nominal_fps, 0.0)), 3),
            "cadence_jitter_cv": 0.0,
            "estimated_dropped_frames": 0,
            "unstable_sampling": True,
        }

    t = np.asarray(timestamps, dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return {
            "coverage_sec": 0.0,
            "effective_fps": round(float(max(nominal_fps, 0.0)), 3),
            "cadence_jitter_cv": 0.0,
            "estimated_dropped_frames": 0,
            "unstable_sampling": True,
        }
    t = np.sort(t)
    dts = np.diff(t)
    dts = dts[dts > 1e-4]
    if len(dts) == 0:
        return {
            "coverage_sec": 0.0,
            "effective_fps": round(float(max(nominal_fps, 0.0)), 3),
            "cadence_jitter_cv": 0.0,
            "estimated_dropped_frames": 0,
            "unstable_sampling": True,
        }

    coverage = float(t[-1] - t[0])
    med_dt = float(np.median(dts))
    mean_dt = float(np.mean(dts))
    dt_std = float(np.std(dts))
    eff_fps = float((len(t) - 1) / coverage) if coverage > 1e-6 else float(nominal_fps)
    jitter_cv = float(dt_std / max(mean_dt, 1e-6))
    expected = int(round(coverage * max(nominal_fps, 1e-6))) + 1
    dropped_est = int(max(expected - len(t), 0))
    unstable_sampling = bool(
        jitter_cv > _UNSTABLE_SAMPLING_JITTER_CV
        or eff_fps < _UNSTABLE_SAMPLING_MIN_EFF_FPS
    )

    return {
        "coverage_sec": round(max(coverage, 0.0), 3),
        "effective_fps": round(eff_fps, 3),
        "median_dt_sec": round(med_dt, 4),
        "cadence_jitter_cv": round(jitter_cv, 4),
        "estimated_dropped_frames": dropped_est,
        "unstable_sampling": unstable_sampling,
    }


def _sanitize_capture_fps(raw_fps: float) -> float:
    """Return a safe FPS value when container metadata is incorrect."""
    if not np.isfinite(raw_fps):
        return _DEFAULT_FPS
    if _MIN_VALID_FPS <= float(raw_fps) <= _MAX_VALID_FPS:
        return float(raw_fps)
    return _DEFAULT_FPS


def _apply_cross_run_smoothing(current_hr: float, channel: str) -> Tuple[float, Optional[float]]:
    """Return the current scan HR unchanged.

    Cross-run smoothing is intentionally disabled to prevent prior scans from
    influencing the current result.
    """
    _ = channel
    return round(current_hr, 1), None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def analyze_face_video(video_path: str) -> Dict[str, Any]:
    """Run the full face-scan pipeline on a video file.

    Parameters
    ----------
    video_path : str
        Path to a video file (mp4, avi, etc.).

    Returns
    -------
    dict
        Rich structured result with heart rate, behavioral features,
        confidence breakdown, quality gating, and retake guidance.

    Integration notes
    -----------------
    * **Flutter / mobile**: POST file to ``/predict/face``, render
      ``hr_timeseries`` as a line chart, show ``retake_required`` and
      ``retake_reasons`` in the UI.
    * **Database**: Persist the full dict for longitudinal tracking.
    * **Real-time**: Replace ``cv2.VideoCapture(path)`` with a live
      frame iterator; accumulate rolling windows.
    """
    # --- Pre-checks ---
    if not _HAS_CV2:
        return _emit_result(_build_error_result(
            "OpenCV is not installed. Cannot process video.",
            {"dependency_missing": "opencv-python"},
        ), source="file")
    if not video_path or not os.path.isfile(str(video_path)):
        return _emit_result(_build_error_result(
            "Video file not found or path is invalid.",
            {"video_path": video_path},
        ), source="file")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _emit_result(_build_error_result(
            "Could not open video file.",
            {"video_path": video_path},
        ), source="file")

    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS)
    fps = _sanitize_capture_fps(raw_fps)
    analysis_fps = LOW_FPS_SAFE_DEFAULT if fps < MIN_ANALYSIS_FPS else fps

    # --- Set up face detector ---
    face_mesh = None
    haar_cascade = None
    use_iris = False

    if _HAS_MP and _FACE_LANDMARKER_MODEL.exists():
        _base_options = mp.tasks.BaseOptions(
            model_asset_path=str(_FACE_LANDMARKER_MODEL)
        )
        _fl_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=_base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(_fl_options)
        use_iris = True
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar_cascade = cv2.CascadeClassifier(cascade_path)

    # --- Per-frame accumulators ---
    roi_traces: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
    roi_timestamps: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
    roi_rgb_traces: Dict[str, List[np.ndarray]] = {name: [] for name in ROI_NAMES}
    roi_overexposure: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
    roi_skin_quality: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
    ear_values: List[float] = []
    gaze_offsets: List[Tuple[float, float]] = []
    region_positions: Dict[str, List[np.ndarray]] = {r: [] for r in REGION_LANDMARKS}
    model_frames: List[np.ndarray] = []  # for pretrained rPPG models

    skin_samples: List[Dict[str, Any]] = []
    eye_color_samples: List[Dict[str, Any]] = []

    frame_qualities: List[float] = []
    motion_magnitudes: List[float] = []
    brightness_scores: List[float] = []
    overexposure_ratios: List[float] = []
    roi_drift_scores: List[float] = []
    blur_scores: List[float] = []
    valid_timestamps: List[float] = []

    frames_read = 0
    face_detected_count = 0
    valid_frame_count = 0
    overexposed_reject_count = 0
    low_brightness_reject_count = 0
    blur_reject_count = 0
    motion_reject_count = 0
    roi_instability_reject_count = 0

    blink_detector = BlinkDetector()
    prev_anchors: Optional[np.ndarray] = None
    landmark_smoother = LandmarkSmoother(alpha=0.6)
    prev_frame_ts: Optional[float] = None
    prev_raw_ts: Optional[float] = None

    # ══════════════════════════════════════════════════════════════
    # Frame loop
    # ══════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        h, w = frame.shape[:2]

        raw_ts_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        has_good_raw_ts = (
            np.isfinite(raw_ts_sec)
            and raw_ts_sec > 0
            and (prev_raw_ts is None or raw_ts_sec > prev_raw_ts + 1e-4)
        )
        if has_good_raw_ts:
            frame_ts_sec = raw_ts_sec
            prev_raw_ts = raw_ts_sec
        else:
            frame_ts_sec = float(frames_read / max(fps, 1e-6))
        if prev_frame_ts is not None and frame_ts_sec <= prev_frame_ts:
            frame_ts_sec = prev_frame_ts + (1.0 / max(fps, 1e-6))
        prev_frame_ts = frame_ts_sec

        # --- Frame-level brightness ---
        bright = frame_brightness(frame)
        bq = _brightness_quality(bright)
        brightness_scores.append(bq)

        # --- Frame-level overexposure ---
        oe = frame_overexposure_ratio(frame)
        overexposure_ratios.append(oe)

        # --- Frame-level blur quality ---
        blur_meta = frame_blur_metrics(frame)
        blur_q = float(blur_meta.get("blur_quality", 0.0))
        blur_scores.append(blur_q)

        # --- Denoise for low-quality webcam noise ---
        frame = denoise_frame(frame)

        # --- CLAHE + adaptive gamma for lighting normalisation ---
        frame = apply_clahe(frame)
        frame = gamma_correct(frame, bright)

        face_detected = False
        motion_mag = 0.0

        # --- MediaPipe path ---
        if face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_ts_sec * 1000)
            results = face_mesh.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                face_detected = True
                face_detected_count += 1
                lm_list = results.face_landmarks[0]

                # Head motion estimation (before smoothing)
                curr_anchors = extract_anchor_positions(lm_list, h, w)
                if prev_anchors is not None:
                    motion_mag = float(np.mean(np.linalg.norm(curr_anchors - prev_anchors, axis=1)))
                prev_anchors = curr_anchors
                motion_magnitudes.append(motion_mag)

                # ROI drift tracking
                drift = landmark_smoother.roi_drift(lm_list, h, w)
                roi_drift_scores.append(drift)

                # Stabilise landmarks (EMA smoothing reduces jitter)
                landmark_smoother.smooth_to_landmarks(lm_list, h, w)

                # Frame quality check
                drift_penalty = float(np.clip(1.0 / (1.0 + max(drift, 0.0) * 0.08), 0.0, 1.0))
                fq = _brightness_quality(bright) * blur_q * (1.0 / (1.0 + motion_mag * 0.5)) * drift_penalty
                frame_qualities.append(fq)

                # Skip very dark frames (insufficient skin contrast).
                if bq < _LOW_BRIGHTNESS_MIN_QUALITY:
                    low_brightness_reject_count += 1
                    continue

                # Skip strongly blurred frames.
                if blur_q < _BLUR_MIN_QUALITY:
                    blur_reject_count += 1
                    continue

                # Skip if ROI geometry is unstable (landmark drift spike).
                if drift > _ROI_DRIFT_REJECT_PX:
                    roi_instability_reject_count += 1
                    continue

                # Skip frame if motion too large (don't corrupt pulse signal)
                if motion_mag > _MOTION_REJECT_THRESHOLD:
                    motion_reject_count += 1
                    continue

                # Skip frame if heavily overexposed (blown-out ROIs)
                if oe > _FRAME_OVEREXPOSURE_REJECT:
                    overexposed_reject_count += 1
                    continue

                valid_frame_count += 1

                # Collect frames for pretrained model inference
                if len(model_frames) < _MAX_MODEL_FRAMES:
                    model_frames.append(frame.copy())

                # 1) Multi-ROI green-channel extraction
                ts_sec = frame_ts_sec
                valid_timestamps.append(ts_sec)
                roi_vals = extract_frame_roi_signals(frame, lm_list, h, w)
                for name in ROI_NAMES:
                    v = roi_vals.get(name)
                    if v is not None:
                        roi_traces[name].append(v)
                        roi_timestamps[name].append(ts_sec)

                # 1b) Multi-ROI RGB extraction (for POS projection)
                roi_rgb = extract_frame_roi_rgb(frame, lm_list, h, w)
                for name in ROI_NAMES:
                    v = roi_rgb.get(name)
                    if v is not None:
                        roi_rgb_traces[name].append(v)

                # 1c) Per-ROI overexposure tracking
                roi_oe = extract_frame_overexposure(frame, lm_list, h, w)
                for name in ROI_NAMES:
                    roi_overexposure[name].append(roi_oe.get(name, 0.0))

                # 1d) Per-ROI skin quality metrics (sampled every 5th valid frame)
                if valid_frame_count % 5 == 0:
                    fh_roi = extract_forehead_roi(frame, lm_list, h, w)
                    if fh_roi is not None:
                        q = compute_roi_quality_metrics(fh_roi)
                        roi_skin_quality["forehead"].append(q["quality_score"])
                    lc = extract_cheek_roi(frame, lm_list, LEFT_CHEEK, h, w)
                    if lc is not None:
                        q = compute_roi_quality_metrics(lc[0], lc[1])
                        roi_skin_quality["left_cheek"].append(q["quality_score"])
                    rc = extract_cheek_roi(frame, lm_list, RIGHT_CHEEK, h, w)
                    if rc is not None:
                        q = compute_roi_quality_metrics(rc[0], rc[1])
                        roi_skin_quality["right_cheek"].append(q["quality_score"])

                # 2) EAR / blink
                ear = avg_ear(lm_list, h, w)
                ear_values.append(round(ear, 4))
                blink_detector.update(ear)

                # 3) Gaze offset
                if use_iris:
                    offset = compute_gaze_offset(lm_list, h, w)
                    if offset is not None:
                        gaze_offsets.append(offset)

                # 4) Region landmarks for motion
                for region, indices in REGION_LANDMARKS.items():
                    pos = landmark_positions(lm_list, indices, h, w)
                    region_positions[region].append(pos)

                # 5) Colour sampling
                if frames_read % _COLOR_SAMPLE_INTERVAL == 0:
                    sc = sample_polygon_color(frame, lm_list, LEFT_CHEEK, h, w)
                    if sc:
                        skin_samples.append(sc)
                    # Also sample right cheek
                    sc2 = sample_polygon_color(frame, lm_list, RIGHT_CHEEK, h, w)
                    if sc2:
                        skin_samples.append(sc2)
                    ec = sample_iris_color(frame, lm_list, LEFT_IRIS, h, w)
                    if ec:
                        eye_color_samples.append(ec)

        # --- Haar fallback (limited features) ---
        elif haar_cascade is not None:
            roi = extract_face_roi_haar(frame, haar_cascade)
            if roi is not None and roi.size > 0:
                face_detected = True
                face_detected_count += 1
                valid_frame_count += 1
                roi_traces["forehead"].append(float(np.mean(roi[:, :, 1])))
                frame_qualities.append(bq)

        if not face_detected:
            frame_qualities.append(0.0)

    cap.release()
    if face_mesh is not None:
        face_mesh.close()

    # ══════════════════════════════════════════════════════════════
    # Post-processing
    # ══════════════════════════════════════════════════════════════
    duration_sec = frames_read / fps if fps > 0 else 0.0

    debug_info: Dict[str, Any] = {
        "frames_processed": frames_read,
        "valid_frames": valid_frame_count,
        "dropped_frames": max(frames_read - valid_frame_count, 0),
        "frames_with_face": face_detected_count,
        "fps": round(analysis_fps, 1),
        "capture_fps": round(fps, 1),
        "raw_fps": round(raw_fps, 3),
        "duration_sec": round(duration_sec, 2),
    }
    timing_diag = _compute_timing_diagnostics(valid_timestamps, fps)
    timing_eff_fps = float(timing_diag.get("effective_fps", fps))
    if timing_eff_fps < MIN_ANALYSIS_FPS:
        analysis_fps = LOW_FPS_SAFE_DEFAULT
    debug_info["analysis_fps"] = round(analysis_fps, 3)
    duration_from_coverage = float(timing_diag.get("coverage_sec", 0.0))
    if duration_from_coverage > duration_sec:
        duration_sec = duration_from_coverage
        debug_info["duration_sec"] = round(duration_sec, 2)
    debug_info["timing"] = timing_diag
    debug_info["raw_effective_fps"] = timing_diag["effective_fps"]
    debug_info["effective_fps"] = round(analysis_fps, 3)
    debug_info["cadence_jitter_cv"] = timing_diag["cadence_jitter_cv"]
    debug_info["low_brightness_reject_count"] = low_brightness_reject_count
    debug_info["blur_reject_count"] = blur_reject_count
    debug_info["motion_reject_count"] = motion_reject_count
    debug_info["roi_instability_reject_count"] = roi_instability_reject_count

    # --- Guard: too few frames ---
    if frames_read < FACE_MIN_FRAMES:
        return _emit_result(_build_error_result(
            f"Video too short ({frames_read} frames). Need at least {FACE_MIN_FRAMES}.",
            debug_info,
        ), source="file")
    if face_detected_count < FACE_MIN_FRAMES:
        return _emit_result(_build_error_result(
            "Face not detected in enough frames for reliable estimation.",
            debug_info,
        ), source="file")
    # Guard: too few *usable* frames (passes motion + overexposure filters)
    _MIN_USABLE_FRAMES = max(FACE_MIN_FRAMES // 3, 10)
    if valid_frame_count < _MIN_USABLE_FRAMES:
        return _emit_result(_build_error_result(
            f"Too few usable frames ({valid_frame_count}) after filtering motion "
            f"and overexposure. Need at least {_MIN_USABLE_FRAMES}. "
            "Please rescan in stable lighting and keep your head still.",
            debug_info,
        ), source="file")
    min_temporal_coverage = max(
        _MIN_TEMPORAL_COVERAGE_SEC,
        min(_MAX_TEMPORAL_COVERAGE_SEC, duration_sec * _TEMPORAL_COVERAGE_RATIO),
    )
    if float(timing_diag.get("coverage_sec", 0.0)) < min_temporal_coverage:
        return _emit_result(_build_error_result(
            "Insufficient temporal coverage after frame-quality filtering. "
            "Please hold still for longer under steady lighting.",
            {
                **debug_info,
                "required_coverage_sec": round(min_temporal_coverage, 2),
            },
        ), source="file")

    # --- Pretrained rPPG model inference (optional layer; attempt early) ---
    try:
        model_result = infer_rppg_models(model_frames, analysis_fps)
    except Exception as exc:
        logger.warning("Pretrained rPPG model inference failed; using classical pipeline: %s", exc)
        model_result = {
            "available_models": [],
            "model_priority": [],
            "legacy_fallback_used": True,
            "open_rppg_active": False,
            "classical_fallback_used": True,
            "fallback_reason": str(exc),
        }

    if not model_result.get("open_rppg_active", False):
        logger.warning(
            "Pretrained rPPG backend inactive; classical fallback in use. reason=%s",
            model_result.get("fallback_reason"),
        )

    # --- 1. Multi-ROI heart rate estimation ---
    roi_weights = {}
    for name in ROI_NAMES:
        # Higher weight for forehead (best rPPG region)
        base_w = 1.5 if name == "forehead" else 1.0
        roi_weights[name] = base_w

    # Compute mean skin coverage per ROI for quality-weighted competition
    roi_skin_cov: Dict[str, float] = {}
    for name in ROI_NAMES:
        sq = roi_skin_quality.get(name, [])
        roi_skin_cov[name] = float(np.mean(sq)) if sq else 1.0

    hr_result = estimate_heart_rate_multi_roi(
        roi_traces, roi_weights, analysis_fps, RPPG_LOW_HZ, RPPG_HIGH_HZ,
        roi_rgb_traces=roi_rgb_traces,
        roi_overexposure=roi_overexposure,
        roi_skin_coverage=roi_skin_cov,
        roi_timestamps=roi_timestamps,
    )
    bpm = hr_result["bpm"]
    hr_quality = hr_result["quality"]
    roi_agreement = hr_result["roi_agreement"]
    periodicity = hr_result["periodicity"]
    competition_signal_strength = float(hr_result.get("signal_strength", 0.0))
    selected_roi = hr_result.get("selected_roi")
    selected_method = hr_result.get("selected_method")

    debug_info["signal_quality"] = hr_quality
    debug_info["roi_quality_scores"] = hr_result["per_roi_quality"]
    debug_info["per_roi_bpm"] = hr_result["per_roi_bpm"]
    debug_info["roi_agreement"] = roi_agreement
    debug_info["signal_periodicity"] = periodicity
    debug_info["pos_used"] = hr_result.get("pos_used", False)
    debug_info["rois_dropped"] = hr_result.get("rois_dropped", [])
    debug_info["roi_adaptive_weights"] = hr_result.get("roi_adaptive_weights", {})
    debug_info["method_adaptive_weights"] = hr_result.get("method_adaptive_weights", {})
    debug_info["candidate_rejections"] = hr_result.get("candidate_rejections", [])
    debug_info["resampled_candidate_count"] = hr_result.get("resampled_candidate_count", 0)
    debug_info["resampled_candidate_ratio"] = hr_result.get("resampled_candidate_ratio", 0.0)
    debug_info["candidate_effective_fps"] = hr_result.get("effective_fps_summary", {})
    debug_info["overexposed_reject_count"] = overexposed_reject_count

    # --- Pretrained rPPG model comparison (optional layer) ---
    comparison = compare_with_signal_pipeline(
        model_result,
        signal_bpm=bpm if bpm else 0.0,
        signal_quality=hr_quality,
    )
    debug_info["rppg_model"] = {
        "model_used": comparison.get("model_name", "none"),
        "source": comparison.get("source", "signal_pipeline"),
        "selection_reason": comparison.get("selection_reason", ""),
        "available_models": model_result.get("available_models", []),
        "model_available": comparison.get("model_available", False),
        "model_loaded": comparison.get("model_loaded", False),
        "inference_source": comparison.get("inference_source", "classical_pipeline"),
        "blend_strategy": comparison.get("source", "signal_pipeline"),
        "model_priority": model_result.get("model_priority", []),
        "legacy_fallback_used": model_result.get("legacy_fallback_used", False),
        "open_rppg_active": model_result.get("open_rppg_active", False),
        "open_rppg_backend_name": model_result.get("open_rppg_backend_name"),
        "classical_fallback_used": model_result.get("classical_fallback_used", True),
        "fallback_reason": model_result.get("fallback_reason"),
    }
    # If model produced a better estimate, use it
    if comparison["source"] != "signal_pipeline" and comparison["bpm"] and comparison["bpm"] > 0:
        bpm = comparison["bpm"]
        hr_quality = comparison["confidence"]

    # Motion / lighting scores for debug
    if motion_magnitudes:
        avg_motion = float(np.mean(motion_magnitudes))
        debug_info["motion_score"] = round(avg_motion, 2)
    else:
        avg_motion = 0.0
        debug_info["motion_score"] = 0.0

    if brightness_scores:
        debug_info["lighting_score"] = round(float(np.mean(brightness_scores)), 3)
        b_arr = np.array(brightness_scores)
        debug_info["brightness_std"] = round(float(np.std(b_arr)), 3)
    else:
        debug_info["lighting_score"] = 0.5
        debug_info["brightness_std"] = 0.0

    if overexposure_ratios:
        debug_info["mean_overexposure"] = round(float(np.mean(overexposure_ratios)), 4)
    else:
        debug_info["mean_overexposure"] = 0.0

    # --- Quality gating ---
    tracking_ratio = face_detected_count / max(frames_read, 1)
    quality_result = compute_scan_quality(
        frame_qualities=frame_qualities,
        tracking_ratio=tracking_ratio,
        motion_scores=motion_magnitudes,
        brightness_scores=brightness_scores,
        roi_agreement=roi_agreement,
        signal_periodicity=periodicity,
        # Avoid false "scan too short" when timing coverage confirms a long scan.
        scan_duration_sec=max(duration_sec, float(timing_diag.get("coverage_sec", 0.0))),
        min_duration_sec=10.0,
        overexposure_ratios=overexposure_ratios,
        usable_frame_count=valid_frame_count,
        total_frame_count=frames_read,
    )
    scan_quality = quality_result["scan_quality"]
    retake_required = quality_result["retake_required"]
    retake_reasons = quality_result["retake_reasons"]
    confidence_breakdown = quality_result["confidence_breakdown"]

    debug_info["tracking_score"] = confidence_breakdown.get("tracking", 0)

    # Recovery fallback is intentionally disabled in final scoring paths.
    # We only report BPM that passes the primary pipeline evidence gates.
    debug_info["recovery_triggered"] = False
    debug_info["recovery"] = None
    debug_info["recovery_attempt_log"] = []

    # --- HR timeseries (spatially-averaged ROI signal for better weak-signal stability) ---
    averaged_trace = _build_spatially_averaged_trace(roi_traces)
    best_roi = max(roi_traces, key=lambda k: len(roi_traces[k]))
    source_trace = averaged_trace if averaged_trace else roi_traces[best_roi]
    hr_timeseries = compute_hr_timeseries(
        source_trace,
        analysis_fps,
        _HR_WINDOW_SEC,
        _HR_STRIDE_SEC,
        RPPG_LOW_HZ,
        RPPG_HIGH_HZ,
        min_window_quality=0.06,
    ) if source_trace else []

    # Temporal smoothing: remove transient spikes from the BPM trace
    if len(hr_timeseries) >= 3:
        hr_timeseries = median_filter_bpm(hr_timeseries, kernel=5)

    # Robust multi-window consensus: remove outliers and estimate a stable HR.
    hr_window_consensus = robust_hr_consensus(
        hr_timeseries,
        periodicity=periodicity,
        min_windows=_MIN_CONSENSUS_WINDOWS,
    )
    if hr_window_consensus["has_consensus"]:
        bpm = hr_window_consensus["heart_rate"]

    hr_filtered_values = hr_window_consensus.get("filtered_hr_values", [])
    hr_min = round(min(hr_filtered_values), 1) if hr_filtered_values else None
    hr_max = round(max(hr_filtered_values), 1) if hr_filtered_values else None
    median_hr = hr_window_consensus.get("median_hr")
    std_dev = float(hr_window_consensus.get("std_dev", 0.0))
    stability = _stability_label(std_dev)

    # --- Determine final HR result ---
    valid_windows = int(hr_window_consensus.get("valid_window_count", 0))
    method_used = "multi_roi_primary"
    if hr_window_consensus.get("has_consensus"):
        method_used = f"window_consensus_{hr_window_consensus.get('method', 'median')}"

    suppression_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    rejection_codes: List[str] = []
    warning: Optional[str] = None
    confidence_caps_applied: List[str] = []
    # Prefer competition engine signal_strength; fall back to old formula
    if competition_signal_strength > 0.0:
        signal_strength = float(np.clip(competition_signal_strength, 0.0, 1.0))
    else:
        signal_strength = float(np.clip(0.6 * max(hr_quality, 0.0) + 0.4 * max(periodicity, 0.0), 0.0, 1.0))

    model_used = bool(comparison.get("source") != "signal_pipeline")

    peak_support_count = int(hr_window_consensus.get("dominant_cluster_size", 0))
    mean_brightness = float(np.mean(brightness_scores)) if brightness_scores else 0.5
    mean_overexposure = float(np.mean(overexposure_ratios)) if overexposure_ratios else 0.0
    tier_eval = _evaluate_hr_result_tier(
        bpm=bpm,
        periodicity=periodicity,
        signal_strength=signal_strength,
        valid_windows=valid_windows,
        std_dev=std_dev,
        peak_support_count=peak_support_count,
        selected_roi=selected_roi,
        timing_diag=timing_diag,
        avg_motion=avg_motion,
        mean_brightness=mean_brightness,
        mean_overexposure=mean_overexposure,
    )
    hr_result_tier = str(tier_eval.get("tier", "result_unavailable"))
    result_available = bool(tier_eval.get("result_available", False))
    estimated_from_weak_signal = bool(tier_eval.get("estimated_from_weak_signal", False))
    has_min_signal = result_available
    rejection_codes = list(tier_eval.get("reasons", []))
    _soft_penalty = 0.0
    if result_available and not estimated_from_weak_signal:
        if std_dev > 9.0:
            _soft_penalty += 0.12
        if valid_windows < 4:
            _soft_penalty += 0.10
    elif result_available and estimated_from_weak_signal:
        _soft_penalty += 0.22
        warning = "Estimated from weak signal. Retake recommended."
    else:
        risk = "unreliable"
        if not rejection_codes:
            rejection_codes.append("insufficient_signal")
        message = (
            "Unable to estimate heart rate: signal evidence is insufficient. "
            "Please retry with steadier posture and more even lighting."
        )
        heart_rate = None
        hr_confidence = 0.0
        retake_required = True
        warning = "No measurable pulse signal."
        suppression_reason = (
            f"Insufficient signal evidence (periodicity={periodicity:.3f}, "
            f"valid_windows={valid_windows}, quality={hr_quality:.3f})"
        )
        if "pulse signal" not in " ".join(retake_reasons).lower():
            retake_reasons.append(
                "Could not detect a measurable pulse signal. Ensure your face is well lit and stable."
            )
    if has_min_signal:
        bpm = float(np.clip(bpm, 35.0, 220.0))
        risk, message = _classify_hr(bpm)
        heart_rate = round(bpm, 1)
        snr_factor = float(hr_window_consensus.get("snr_proxy", hr_quality))
        peak_quality_factor = float(hr_window_consensus.get("peak_quality_score", 0.0))
        cluster_tightness = float(hr_window_consensus.get("cluster_tightness", 0.0))
        run_delta = None
        heart_rate, run_delta = _apply_cross_run_smoothing(float(heart_rate), channel="file")
        bpm = heart_rate
        valid_window_ratio = float(np.clip(valid_windows / 8.0, 0.0, 1.0))
        if periodicity <= 0.0:
            periodicity_effect = 0.15
            confidence_cap_due_to_periodicity = 0.5
            confidence_caps_applied.append("periodicity_zero_cap_0.50")
        else:
            periodicity_effect = max(periodicity, 0.2)
            confidence_cap_due_to_periodicity = 1.0
        hr_confidence = (
            0.23 * signal_strength
            + 0.23 * peak_quality_factor
            + 0.18 * snr_factor
            + 0.14 * cluster_tightness
            + 0.12 * valid_window_ratio
            + 0.10 * periodicity_effect
        )
        # Cross-run consistency adjustment
        if run_delta is not None:
            if run_delta <= 3.0:
                hr_confidence = min(1.0, hr_confidence + 0.05)
            elif run_delta >= 8.0:
                hr_confidence = max(0.0, hr_confidence - 0.05)
        # Apply tier-based penalty.
        hr_confidence = max(0.0, hr_confidence - _soft_penalty)
        # Clamp based on signal strength
        if signal_strength < 0.25:
            hr_confidence = float(np.clip(hr_confidence, 0.0, 0.45))
        elif signal_strength < 0.4:
            hr_confidence = float(np.clip(hr_confidence, 0.0, 0.55))
        elif signal_strength >= 0.65:
            hr_confidence = float(np.clip(hr_confidence, 0.0, 0.92))
        else:
            hr_confidence = float(np.clip(hr_confidence, 0.0, 0.80))
        dominant_ratio = float(hr_window_consensus.get("dominant_cluster_ratio", 0.0))
        if std_dev < 3.0:
            hr_confidence = min(1.0, hr_confidence + 0.06)
        if dominant_ratio >= 0.70:
            hr_confidence = min(1.0, hr_confidence + 0.05)
        if std_dev < 12.0 and dominant_ratio > 0.30:
            hr_confidence = min(1.0, hr_confidence + 0.10)
        hr_confidence = float(np.clip(hr_confidence, 0.0, 1.0))
        hr_confidence = min(hr_confidence, confidence_cap_due_to_periodicity)

        # Usable scans should not collapse to near-zero confidence.
        if not estimated_from_weak_signal and valid_windows >= 4 and periodicity >= 0.40:
            hr_confidence = max(hr_confidence, 0.55)

        # Tier-aware confidence shaping.
        if estimated_from_weak_signal:
            hr_confidence = float(np.clip(hr_confidence, 0.10, 0.38))
            retake_required = True
            if "Low confidence" not in " ".join(retake_reasons):
                retake_reasons.append("Low confidence estimated result; retake recommended.")
        elif bool(hr_window_consensus.get("unstable", False)):
            hr_confidence = float(np.clip(hr_confidence, 0.10, 0.30))
            warning = "High variability detected - result may not be reliable."
        elif periodicity < 0.30 or valid_windows < 4:
            hr_confidence = float(np.clip(hr_confidence, 0.10, 0.40))
            warning = "Low confidence result: estimated from weak signal."
        elif valid_windows < _MIN_CONSENSUS_WINDOWS:
            warning = (
                "Window count is limited for weak signal. "
                "Use a longer recording for higher stability."
            )

    reliability = _derive_reliability(
        confidence=hr_confidence,
        periodicity=periodicity,
        valid_windows=valid_windows,
        has_signal=has_min_signal,
    )
    if estimated_from_weak_signal and has_min_signal:
        reliability = "low"
    if reliability == "low" and warning is None and heart_rate is not None:
        warning = "Low confidence result: estimated from weak signal."

    if heart_rate is not None:
        # Do not show missing-pulse warnings when we have a computed HR.
        if warning and "pulse" in warning.lower() and "no" in warning.lower():
            warning = "Low signal strength - estimate may vary."
        elif warning is None and (stability == "Low" or signal_strength < 0.35):
            warning = "Low signal strength - estimate may vary."
        retake_reasons = [
            r for r in retake_reasons if "could not detect" not in str(r).lower()
        ]

    if reliability == "low":
        retake_required = True
        if "low confidence" not in " ".join(retake_reasons).lower():
            retake_reasons.append("Low confidence result from weak signal; retake for better accuracy.")

    # Availability-first override: stable, high-window scans should always surface BPM.
    usable_scan_override = valid_windows >= 6 and median_hr is not None
    severe_motion_or_no_face = avg_motion > _MOTION_REJECT_THRESHOLD or face_detected_count <= 0

    if valid_windows >= 6:
        hr_result_tier = "result_available"
        if not severe_motion_or_no_face:
            retake_required = False

    if usable_scan_override:
        result_available = True
        has_min_signal = True
        estimated_from_weak_signal = False
        if heart_rate is None:
            heart_rate = round(float(median_hr), 1)
        hr_confidence = max(float(hr_confidence), 0.55)
        reliability = "medium"
        warning = None
        rejection_codes = []
        if not severe_motion_or_no_face:
            retake_required = False
            retake_reasons = [
                r for r in retake_reasons
                if "low confidence" not in str(r).lower()
                and "retake" not in str(r).lower()
            ]

    # For low-window but usable scans, apply only mild confidence penalty.
    if has_min_signal and valid_windows < 6:
        hr_confidence = float(np.clip(hr_confidence * 0.85, 0.10, 1.0))
        confidence_caps_applied.append("low_window_penalty_x0.85")
        if not severe_motion_or_no_face:
            result_available = True
            hr_result_tier = "result_available"
            estimated_from_weak_signal = False
            retake_required = False
            retake_reasons = [
                r for r in retake_reasons
                if "low confidence" not in str(r).lower()
                and "retake" not in str(r).lower()
            ]

    # Empty/unsupported classical evidence should reduce trust, not block availability.
    no_classical_support = (
        len(hr_result.get("all_candidate_scores", [])) == 0 or float(roi_agreement) == 0.0
    )
    if no_classical_support and has_min_signal:
        hr_confidence = float(np.clip(hr_confidence * 0.85, 0.10, 1.0))
        confidence_caps_applied.append("no_classical_support_penalty_x0.85")
        if "no_classical_support" not in rejection_codes:
            rejection_codes.append("no_classical_support")

    # Hard signal reality gating: availability is kept, trust is downgraded.
    signal_invalid = (
        periodicity < 0.1
        and signal_strength < 0.25
        and float(roi_agreement) == 0.0
    )

    if model_used and signal_invalid and has_min_signal:
        hr_confidence = float(np.clip(hr_confidence * 0.8, 0.10, 1.0))
        confidence_caps_applied.append("model_on_invalid_penalty_x0.80")
    if model_used and periodicity <= 0.0 and has_min_signal:
        hr_confidence = min(hr_confidence, 0.5)
        confidence_caps_applied.append("model_periodicity_zero_cap_0.50")

    if signal_invalid:
        if heart_rate is None:
            if median_hr is not None:
                heart_rate = round(float(median_hr), 1)
            elif bpm is not None and float(bpm) > 0:
                heart_rate = round(float(bpm), 1)
        result_available = True
        if valid_windows >= 6:
            hr_result_tier = "result_available"
        estimated_from_weak_signal = True
        reliability = "low"
        hr_confidence = min(float(hr_confidence), 0.45)
        confidence_caps_applied.append("signal_invalid_cap_0.45")
        warning = "Very weak physiological signal. Estimate may be unreliable."
        if "no_signal_structure" not in rejection_codes:
            rejection_codes.append("no_signal_structure")

    # --- 2. Blink ---
    blink_results = aggregate_blink_results(
        ear_values, blink_detector.blink_count, duration_sec,
    )
    blink_rate = blink_results.get("blink_rate_per_min")

    # --- 3. Eye movement ---
    eye_movement = aggregate_eye_movement(gaze_offsets, duration_sec)
    eye_stability = eye_movement.get("eye_stability")

    # --- 4. Facial motion / tension ---
    facial_motion = aggregate_facial_motion(region_positions)
    tension_index = facial_motion.get("facial_tension_index")

    # --- 5. Eye colour ---
    eye_color = aggregate_eye_color(eye_color_samples)

    # --- 6. Skin analysis ---
    skin_analysis = aggregate_skin_analysis(skin_samples)
    skin_stability = skin_analysis.get("skin_signal_stability")

    # ══════════════════════════════════════════════════════════════
    # Assemble final result
    # ══════════════════════════════════════════════════════════════
    debug_info["signal_strength"] = round(signal_strength, 3)
    debug_info["periodicity_score"] = round(periodicity, 3)
    debug_info["valid_windows"] = valid_windows
    debug_info["method_used"] = method_used
    debug_info["hr_values"] = hr_window_consensus.get("hr_values", [])
    debug_info["median_hr"] = hr_window_consensus.get("median_hr")
    debug_info["std_dev"] = hr_window_consensus.get("std_dev", 0.0)
    debug_info["variance"] = hr_window_consensus.get("variance", 0.0)
    debug_info["valid_window_count"] = hr_window_consensus.get("valid_window_count", 0)
    debug_info["selected_windows"] = hr_window_consensus.get("selected_windows", [])
    debug_info["rejected_windows"] = hr_window_consensus.get("rejected_windows", [])
    debug_info["peak_quality_score"] = hr_window_consensus.get("peak_quality_score", 0.0)
    debug_info["dominant_cluster_ratio"] = hr_window_consensus.get("dominant_cluster_ratio", 0.0)
    debug_info["dominant_cluster_std"] = hr_window_consensus.get("dominant_cluster_std", 0.0)
    debug_info["cluster_tightness"] = hr_window_consensus.get("cluster_tightness", 0.0)
    debug_info["heart_rate_range"] = [hr_min, hr_max] if hr_min is not None and hr_max is not None else None
    debug_info["stability"] = stability
    debug_info["final_method_used"] = method_used
    debug_info["final_acceptance_reason"] = (
        f"{hr_result_tier}: periodicity={periodicity:.3f}, valid_windows={valid_windows}, signal_strength={signal_strength:.3f}"
        if has_min_signal
        else f"result_unavailable: reasons={','.join(rejection_codes)}"
    )
    debug_info["hr_result_tier"] = hr_result_tier
    debug_info["result_available"] = result_available
    debug_info["estimated_from_weak_signal"] = estimated_from_weak_signal
    debug_info["signal_invalid"] = signal_invalid
    debug_info["model_used"] = model_used
    debug_info["confidence_caps_applied"] = confidence_caps_applied
    debug_info["hr_rejection_codes"] = rejection_codes
    debug_info["suppression_reason"] = suppression_reason
    debug_info["rejection_reason"] = rejection_reason
    debug_info["peak_support_count"] = peak_support_count
    debug_info["usable_frame_count"] = valid_frame_count
    # Competition engine fields
    debug_info["selected_roi"] = selected_roi
    debug_info["selected_method"] = selected_method
    debug_info["competition_signal_strength"] = round(competition_signal_strength, 3)
    debug_info["all_candidate_scores"] = hr_result.get("all_candidate_scores", [])
    debug_info["roi_scores"] = hr_result.get("roi_scores", {})
    debug_info["method_scores"] = hr_result.get("method_scores", {})
    debug_info["roi_skin_coverage"] = {k: round(v, 3) for k, v in roi_skin_cov.items()}
    if roi_drift_scores:
        debug_info["mean_roi_drift"] = round(float(np.mean(roi_drift_scores)), 3)

    cap_suffix = " (capped)" if confidence_caps_applied else ""
    print(
        f"[SCAN SUMMARY] fps={analysis_fps:.2f}, effective_fps={debug_info.get('effective_fps', analysis_fps):.2f}, "
        f"windows={valid_windows}, periodicity={periodicity:.3f}, signal={signal_strength:.3f}, "
        f"signal_invalid={signal_invalid}, model={comparison.get('model_name', 'none')}, "
        f"bpm={median_hr}, confidence={hr_confidence:.2f}{cap_suffix}, caps={confidence_caps_applied}"
    )

    return {
        # --- Primary fields ---
        "module_name": "face_module",
        "scan_duration_sec": round(duration_sec, 2),

        # Heart rate
        "heart_rate": heart_rate,
        "stable_heart_rate": median_hr if median_hr is not None else heart_rate,
        "stability": stability,
        "heart_rate_unit": "bpm",
        "heart_rate_confidence": hr_confidence,

        # Behavioral features
        "blink_rate": blink_rate,
        "blink_rate_unit": "blinks/min",
        "eye_stability": round(eye_stability, 3) if eye_stability is not None else None,
        "facial_tension_index": round(tension_index, 3) if tension_index is not None else None,
        "skin_signal_stability": round(skin_stability, 3) if skin_stability is not None else None,

        # Quality & gating
        "scan_quality": scan_quality,
        "retake_required": retake_required,
        "retake_reasons": retake_reasons,
        "confidence": hr_confidence,
        "hr_confidence": hr_confidence,
        "reliability": reliability,
        "hr_result_tier": hr_result_tier,
        "result_available": result_available,
        "retake_recommended": retake_required,
        "estimated_from_weak_signal": estimated_from_weak_signal,
        "signal_strength": round(signal_strength, 3),
        "periodicity_score": round(periodicity, 3),
        "valid_windows": valid_windows,
        "method_used": method_used,
        "warning": warning,
        "hr_rejection_reasons": rejection_codes,
        "confidence_breakdown": confidence_breakdown,

        # Risk & message
        "risk": risk,
        "message": message,

        # Detailed debug
        "debug": debug_info,

        # Sub-dicts for detailed analytics / charting
        "hr_timeseries": hr_timeseries,
        "blink_analysis": blink_results,
        "eye_movement": eye_movement,
        "facial_motion": facial_motion,
        "eye_color": eye_color,
        "skin_color": skin_analysis,

        # Legacy compatibility fields
        "metric_name": "heart_rate",
        "value": heart_rate,
        "unit": "bpm",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Real-time streaming pipeline
# ═══════════════════════════════════════════════════════════════════════════

# Streaming constants
_STREAM_INTERIM_EVERY_N = 15      # emit an interim result every N valid frames
_STREAM_MIN_FRAMES_ESTIMATE = 30  # need at least this many frames before any HR estimate
_STREAM_TARGET_DURATION = 30.0    # seconds; scanner will auto-finalise after this
_STREAM_MAX_BUFFER = 600          # hard cap on stored frames for pretrained model inference


class FaceStreamProcessor:
    """Stateful per-frame processor for real-time face scanning.

    Usage pattern
    -------------
    ::

        proc = FaceStreamProcessor(fps=30.0)
        for frame in camera:
            event = proc.push_frame(frame)
            if event:
                send_to_client(event)          # "interim" or "final"
            if proc.is_done:
                break
        final = proc.finalise()
        send_to_client({"event": "final", "data": final})

    Frame format: BGR numpy array (same as ``cv2.VideoCapture.read``).

    WebSocket protocol notes
    ------------------------
    * Each event dict has an ``"event"`` key: ``"interim"`` (mid-scan
      rolling estimate) or ``"final"`` (complete result after target
      duration).
    * ``"interim"`` events carry lightweight fields so clients can
      update a live BPM gauge cheaply.
    * ``"final"`` events have the identical schema as
      ``analyze_face_video`` for drop-in compatibility.
    * Clients can send any frame size; the processor does not resize.

    Integration notes (Flutter / web)
    ----------------------------------
    * Connect to ``ws://<host>/ws/face-scan``.
    * Send raw JPEG bytes for each camera frame.
    * Parse incoming JSON: ``event == "interim"`` → update gauge;
      ``event == "final"`` → show full result; ``event == "error"`` →
      show error message.
    * On ``final``, close the WebSocket.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = max(fps, 1.0)
        self._frame_count = 0          # total frames received
        self._valid_frame_count = 0    # frames with face detected
        self._last_interim_frame = 0   # last frame that triggered an interim emit

        # Rolling buffers (same accumulators as analyze_face_video)
        self._roi_traces: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
        self._roi_timestamps: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
        self._roi_rgb_traces: Dict[str, List[np.ndarray]] = {name: [] for name in ROI_NAMES}
        self._roi_overexposure: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
        self._ear_values: List[float] = []
        self._gaze_offsets: List[Tuple[float, float]] = []
        self._region_positions: Dict[str, List[np.ndarray]] = {r: [] for r in REGION_LANDMARKS}
        self._model_frames: List[np.ndarray] = []
        self._skin_samples: List[Dict[str, Any]] = []
        self._eye_color_samples: List[Dict[str, Any]] = []
        self._frame_qualities: List[float] = []
        self._motion_magnitudes: List[float] = []
        self._brightness_scores: List[float] = []
        self._overexposure_ratios: List[float] = []
        self._blur_scores: List[float] = []
        self._valid_timestamps: List[float] = []
        self._blink_detector = BlinkDetector()
        self._prev_anchors: Optional[np.ndarray] = None
        self._overexposed_reject_count = 0
        self._low_brightness_reject_count = 0
        self._blur_reject_count = 0
        self._motion_reject_count = 0
        self._roi_instability_reject_count = 0
        self._roi_skin_quality: Dict[str, List[float]] = {name: [] for name in ROI_NAMES}
        self._roi_drift_scores: List[float] = []
        self._landmark_smoother = LandmarkSmoother(alpha=0.6)

        # Lazily init MediaPipe / Haar
        self._face_mesh = None
        self._haar_cascade = None
        self._use_iris = False
        self._init_detector()

        self.is_done = False
        self._finalised_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _init_detector(self) -> None:
        if _HAS_MP and _FACE_LANDMARKER_MODEL.exists():
            try:
                _base_options = mp.tasks.BaseOptions(
                    model_asset_path=str(_FACE_LANDMARKER_MODEL)
                )
                _fl_options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=_base_options,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(_fl_options)
                self._use_iris = True
            except Exception as exc:
                logger.warning(
                    "MediaPipe FaceLandmarker init failed (%s) – falling back to Haar.", exc
                )
                self._face_mesh = None
        if self._face_mesh is None and _HAS_CV2:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)

    # ------------------------------------------------------------------
    def push_frame(self, frame: np.ndarray, timestamp_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Feed one BGR frame into the processor.

        Returns an ``"interim"`` event dict every
        ``_STREAM_INTERIM_EVERY_N`` valid frames, or ``None`` if no
        emit is due.  When the target scan duration is reached the
        method returns a ``"final"`` event and sets ``self.is_done``.
        """
        if self.is_done:
            return None

        self._frame_count += 1
        h, w = frame.shape[:2]

        bright = frame_brightness(frame)
        bq = _brightness_quality(bright)
        self._brightness_scores.append(bq)

        # Overexposure + denoise + CLAHE + gamma
        oe = frame_overexposure_ratio(frame)
        self._overexposure_ratios.append(oe)
        blur_meta = frame_blur_metrics(frame)
        blur_q = float(blur_meta.get("blur_quality", 0.0))
        self._blur_scores.append(blur_q)
        frame = denoise_frame(frame)
        frame = apply_clahe(frame)
        frame = gamma_correct(frame, bright)

        face_detected = False
        motion_mag = 0.0

        # ── MediaPipe path ────────────────────────────────────────
        if self._face_mesh is not None and _HAS_CV2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((self._frame_count / self._fps) * 1000)
            results = self._face_mesh.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                face_detected = True
                lm_list = results.face_landmarks[0]

                # Landmark stabilization via EMA smoothing
                drift = self._landmark_smoother.roi_drift(lm_list, h, w)
                self._roi_drift_scores.append(drift)
                self._landmark_smoother.smooth_to_landmarks(lm_list, h, w)

                curr_anchors = extract_anchor_positions(lm_list, h, w)
                if self._prev_anchors is not None:
                    motion_mag = float(
                        np.mean(np.linalg.norm(curr_anchors - self._prev_anchors, axis=1))
                    )
                self._prev_anchors = curr_anchors
                self._motion_magnitudes.append(motion_mag)

                fq = bq * (1.0 / (1.0 + motion_mag * 0.5))
                drift_penalty = float(np.clip(1.0 / (1.0 + max(drift, 0.0) * 0.08), 0.0, 1.0))
                fq = fq * blur_q * drift_penalty
                self._frame_qualities.append(fq)

                if bq < _LOW_BRIGHTNESS_MIN_QUALITY:
                    self._low_brightness_reject_count += 1
                elif blur_q < _BLUR_MIN_QUALITY:
                    self._blur_reject_count += 1
                elif drift > _ROI_DRIFT_REJECT_PX:
                    self._roi_instability_reject_count += 1
                elif motion_mag > _MOTION_REJECT_THRESHOLD:
                    self._motion_reject_count += 1
                elif oe > _FRAME_OVEREXPOSURE_REJECT:
                    self._overexposed_reject_count += 1
                else:
                    self._valid_frame_count += 1

                    if len(self._model_frames) < _STREAM_MAX_BUFFER:
                        self._model_frames.append(frame.copy())

                    curr_ts = (
                        float(timestamp_sec)
                        if timestamp_sec is not None
                        else float(self._frame_count / max(self._fps, 1e-6))
                    )
                    self._valid_timestamps.append(curr_ts)

                    roi_vals = extract_frame_roi_signals(frame, lm_list, h, w)
                    for name in ROI_NAMES:
                        v = roi_vals.get(name)
                        if v is not None:
                            self._roi_traces[name].append(v)
                            self._roi_timestamps[name].append(curr_ts)

                    roi_rgb = extract_frame_roi_rgb(frame, lm_list, h, w)
                    for name in ROI_NAMES:
                        v = roi_rgb.get(name)
                        if v is not None:
                            self._roi_rgb_traces[name].append(v)

                    roi_oe = extract_frame_overexposure(frame, lm_list, h, w)
                    for name in ROI_NAMES:
                        self._roi_overexposure[name].append(roi_oe.get(name, 0.0))

                    # Sample ROI skin quality every 5th valid frame
                    if self._valid_frame_count % 5 == 0:
                        fh_roi = extract_forehead_roi(frame, lm_list, h, w)
                        if fh_roi is not None and fh_roi.size > 0:
                            q = compute_roi_quality_metrics(fh_roi)
                            self._roi_skin_quality["forehead"].append(q["quality_score"])

                        lc = extract_cheek_roi(frame, lm_list, LEFT_CHEEK, h, w)
                        if lc is not None:
                            q = compute_roi_quality_metrics(lc[0], lc[1])
                            self._roi_skin_quality["left_cheek"].append(q["quality_score"])

                        rc = extract_cheek_roi(frame, lm_list, RIGHT_CHEEK, h, w)
                        if rc is not None:
                            q = compute_roi_quality_metrics(rc[0], rc[1])
                            self._roi_skin_quality["right_cheek"].append(q["quality_score"])

                    ear = avg_ear(lm_list, h, w)
                    self._ear_values.append(round(ear, 4))
                    self._blink_detector.update(ear)

                    if self._use_iris:
                        offset = compute_gaze_offset(lm_list, h, w)
                        if offset is not None:
                            self._gaze_offsets.append(offset)

                    for region, indices in REGION_LANDMARKS.items():
                        pos = landmark_positions(lm_list, indices, h, w)
                        self._region_positions[region].append(pos)

                    if self._frame_count % _COLOR_SAMPLE_INTERVAL == 0:
                        sc = sample_polygon_color(frame, lm_list, LEFT_CHEEK, h, w)
                        if sc:
                            self._skin_samples.append(sc)
                        sc2 = sample_polygon_color(frame, lm_list, RIGHT_CHEEK, h, w)
                        if sc2:
                            self._skin_samples.append(sc2)
                        ec = sample_iris_color(frame, lm_list, LEFT_IRIS, h, w)
                        if ec:
                            self._eye_color_samples.append(ec)

        # ── Haar fallback ─────────────────────────────────────────
        elif self._haar_cascade is not None and _HAS_CV2:
            roi = extract_face_roi_haar(frame, self._haar_cascade)
            if roi is not None and roi.size > 0:
                face_detected = True
                self._valid_frame_count += 1
                self._roi_traces["forehead"].append(float(np.mean(roi[:, :, 1])))
                self._frame_qualities.append(bq)
                self._motion_magnitudes.append(0.0)

        if not face_detected:
            self._frame_qualities.append(0.0)

        # ── Auto-finalise once target duration reached ────────────
        elapsed = self._frame_count / self._fps
        if elapsed >= _STREAM_TARGET_DURATION:
            self.is_done = True
            return self._build_final_event()

        # ── Periodic interim emit ─────────────────────────────────
        frames_since_last = self._valid_frame_count - self._last_interim_frame
        if (
            self._valid_frame_count >= _STREAM_MIN_FRAMES_ESTIMATE
            and frames_since_last >= _STREAM_INTERIM_EVERY_N
        ):
            self._last_interim_frame = self._valid_frame_count
            return self._build_interim_event()

        return None

    # ------------------------------------------------------------------
    def _quick_hr_estimate(self) -> Tuple[Optional[float], float]:
        """Fast HR estimate on current buffer.  Returns (bpm, quality)."""
        roi_weights = {n: (1.5 if n == "forehead" else 1.0) for n in ROI_NAMES}
        try:
            hr = estimate_heart_rate_multi_roi(
                self._roi_traces, roi_weights, self._fps, RPPG_LOW_HZ, RPPG_HIGH_HZ,
                roi_rgb_traces=self._roi_rgb_traces,
                roi_overexposure=self._roi_overexposure,
                roi_timestamps=self._roi_timestamps,
            )
            return hr["bpm"], hr["quality"]
        except Exception:
            return None, 0.0

    # ------------------------------------------------------------------
    def _build_interim_event(self) -> Dict[str, Any]:
        """Lightweight mid-scan payload for updating the live gauge."""
        bpm, quality = self._quick_hr_estimate()
        duration = self._frame_count / self._fps
        tracking_ratio = (
            self._valid_frame_count / max(self._frame_count, 1)
        )
        progress_pct = round(min(duration / _STREAM_TARGET_DURATION * 100, 99), 1)

        return {
            "event": "interim",
            "data": {
                "heart_rate": round(bpm, 1) if bpm else None,
                "heart_rate_unit": "bpm",
                "confidence": round(quality, 3),
                "scan_quality": round(quality * tracking_ratio, 3),
                "frames_processed": self._frame_count,
                "valid_frames": self._valid_frame_count,
                "tracking_ratio": round(tracking_ratio, 3),
                "elapsed_sec": round(duration, 1),
                "progress_pct": progress_pct,
                "blink_count": self._blink_detector.blink_count,
            },
        }

    # ------------------------------------------------------------------
    def _build_final_event(self) -> Dict[str, Any]:
        """Run full post-processing and build the complete final result."""
        result = self._compute_final_result()
        self._finalised_result = result
        return {"event": "final", "data": result}

    # ------------------------------------------------------------------
    def _compute_final_result(self) -> Dict[str, Any]:
        """Mirror the post-processing block of analyze_face_video."""
        duration_sec = self._frame_count / self._fps

        _MIN_STREAM_VALID_FRAMES = max(FACE_MIN_FRAMES // 3, 10)
        if self._frame_count < FACE_MIN_FRAMES or self._valid_frame_count < _MIN_STREAM_VALID_FRAMES:
            return _emit_result(_build_error_result(
                "Not enough valid frames captured for a reliable estimate.",
                {"frames": self._frame_count, "valid": self._valid_frame_count},
            ), source="stream")
        timing_diag = _compute_timing_diagnostics(self._valid_timestamps, self._fps)
        _MIN_USABLE_FRAMES = max(FACE_MIN_FRAMES // 3, 10)
        if self._valid_frame_count < _MIN_USABLE_FRAMES:
            return _emit_result(_build_error_result(
                f"Too few usable frames ({self._valid_frame_count}) after filtering "
                f"motion and overexposure. Need at least {_MIN_USABLE_FRAMES}. "
                "Please rescan in stable lighting and keep your head still.",
                {"frames": self._frame_count, "valid": self._valid_frame_count},
            ), source="stream")
        min_temporal_coverage = max(
            _MIN_TEMPORAL_COVERAGE_SEC,
            min(_MAX_TEMPORAL_COVERAGE_SEC, duration_sec * _TEMPORAL_COVERAGE_RATIO),
        )
        if float(timing_diag.get("coverage_sec", 0.0)) < min_temporal_coverage:
            return _emit_result(_build_error_result(
                "Insufficient temporal coverage after frame-quality filtering. "
                "Please hold still for longer under steady lighting.",
                {
                    "frames": self._frame_count,
                    "valid": self._valid_frame_count,
                    "timing": timing_diag,
                    "required_coverage_sec": round(min_temporal_coverage, 2),
                },
            ), source="stream")

        # HR
        roi_weights = {n: (1.5 if n == "forehead" else 1.0) for n in ROI_NAMES}
        # Compute skin coverage for competition engine
        _roi_skin_cov: Dict[str, float] = {}
        for _n in ROI_NAMES:
            _sq = self._roi_skin_quality.get(_n, [])
            _roi_skin_cov[_n] = float(np.mean(_sq)) if _sq else 1.0
        hr_result = estimate_heart_rate_multi_roi(
            self._roi_traces, roi_weights, self._fps, RPPG_LOW_HZ, RPPG_HIGH_HZ,
            roi_rgb_traces=self._roi_rgb_traces,
            roi_overexposure=self._roi_overexposure,
            roi_skin_coverage=_roi_skin_cov,
            roi_timestamps=self._roi_timestamps,
        )
        bpm = hr_result["bpm"]
        hr_quality = hr_result["quality"]
        roi_agreement = hr_result["roi_agreement"]
        periodicity = hr_result["periodicity"]
        _comp_signal_strength = float(hr_result.get("signal_strength", 0.0))
        _selected_roi = hr_result.get("selected_roi")
        _selected_method = hr_result.get("selected_method")
        _roi_adaptive_weights = hr_result.get("roi_adaptive_weights", {})
        _method_adaptive_weights = hr_result.get("method_adaptive_weights", {})
        _candidate_rejections = hr_result.get("candidate_rejections", [])
        _resampled_candidate_count = int(hr_result.get("resampled_candidate_count", 0))
        _resampled_candidate_ratio = float(hr_result.get("resampled_candidate_ratio", 0.0))
        _candidate_effective_fps = hr_result.get("effective_fps_summary", {})

        # Optional model inference
        model_result = infer_rppg_models(self._model_frames, self._fps)
        comparison = compare_with_signal_pipeline(
            model_result, signal_bpm=bpm or 0.0, signal_quality=hr_quality
        )
        if comparison["source"] != "signal_pipeline" and (comparison["bpm"] or 0) > 0:
            bpm = comparison["bpm"]
            hr_quality = comparison["confidence"]

        tracking_ratio = self._valid_frame_count / max(self._frame_count, 1)
        quality_result = compute_scan_quality(
            frame_qualities=self._frame_qualities,
            tracking_ratio=tracking_ratio,
            motion_scores=self._motion_magnitudes,
            brightness_scores=self._brightness_scores,
            roi_agreement=roi_agreement,
            signal_periodicity=periodicity,
            scan_duration_sec=duration_sec,
            min_duration_sec=10.0,
            overexposure_ratios=self._overexposure_ratios,
            usable_frame_count=self._valid_frame_count,
            total_frame_count=self._frame_count,
        )
        scan_quality = quality_result["scan_quality"]
        retake_required = quality_result["retake_required"]
        retake_reasons = quality_result["retake_reasons"]
        confidence_breakdown = quality_result["confidence_breakdown"]

        # Recovery fallback is intentionally disabled in final scoring paths.
        _stream_recovery_triggered = False
        _stream_recovery_succeeded = False
        _stream_recovery = None

        best_roi = max(self._roi_traces, key=lambda k: len(self._roi_traces[k]))
        averaged_trace = _build_spatially_averaged_trace(self._roi_traces)
        source_trace = averaged_trace if averaged_trace else self._roi_traces[best_roi]
        hr_timeseries = compute_hr_timeseries(
            source_trace,
            self._fps,
            _HR_WINDOW_SEC,
            _HR_STRIDE_SEC,
            RPPG_LOW_HZ,
            RPPG_HIGH_HZ,
            min_window_quality=0.10,
        ) if source_trace else []
        if len(hr_timeseries) >= 3:
            hr_timeseries = median_filter_bpm(hr_timeseries, kernel=5)

        hr_window_consensus = robust_hr_consensus(
            hr_timeseries,
            periodicity=periodicity,
            min_windows=_MIN_CONSENSUS_WINDOWS,
        )
        if hr_window_consensus["has_consensus"]:
            bpm = hr_window_consensus["heart_rate"]

        hr_filtered_values = hr_window_consensus.get("filtered_hr_values", [])
        hr_min = round(min(hr_filtered_values), 1) if hr_filtered_values else None
        hr_max = round(max(hr_filtered_values), 1) if hr_filtered_values else None
        median_hr = hr_window_consensus.get("median_hr")
        std_dev = float(hr_window_consensus.get("std_dev", 0.0))
        stability = _stability_label(std_dev)

        valid_windows = int(hr_window_consensus.get("valid_window_count", 0))
        method_used = "multi_roi_primary"
        if hr_window_consensus.get("has_consensus"):
            method_used = f"window_consensus_{hr_window_consensus.get('method', 'median')}"
        signal_strength = (
            float(np.clip(_comp_signal_strength, 0.0, 1.0))
            if _comp_signal_strength > 0.0
            else float(np.clip(0.6 * max(hr_quality, 0.0) + 0.4 * max(periodicity, 0.0), 0.0, 1.0))
        )
        peak_support_count = int(hr_window_consensus.get("dominant_cluster_size", 0))
        avg_motion = float(np.mean(self._motion_magnitudes)) if self._motion_magnitudes else 0.0
        mean_brightness = float(np.mean(self._brightness_scores)) if self._brightness_scores else 0.5
        mean_overexposure = float(np.mean(self._overexposure_ratios)) if self._overexposure_ratios else 0.0
        tier_eval = _evaluate_hr_result_tier(
            bpm=bpm,
            periodicity=periodicity,
            signal_strength=signal_strength,
            valid_windows=valid_windows,
            std_dev=std_dev,
            peak_support_count=peak_support_count,
            selected_roi=_selected_roi,
            timing_diag=timing_diag,
            avg_motion=avg_motion,
            mean_brightness=mean_brightness,
            mean_overexposure=mean_overexposure,
        )
        hr_result_tier = str(tier_eval.get("tier", "result_unavailable"))
        result_available = bool(tier_eval.get("result_available", False))
        estimated_from_weak_signal = bool(tier_eval.get("estimated_from_weak_signal", False))
        has_min_signal = result_available
        rejection_codes: List[str] = list(tier_eval.get("reasons", []))
        warning: Optional[str] = None
        _soft_penalty = 0.0
        if result_available and not estimated_from_weak_signal:
            if std_dev > 9.0:
                _soft_penalty += 0.10
        elif result_available and estimated_from_weak_signal:
            _soft_penalty += 0.20
            warning = "Estimated from weak signal. Retake recommended."

        if not has_min_signal:
            risk = "unreliable"
            message = (
                "Unable to estimate heart rate: signal evidence is insufficient. "
                "Please retry with steadier posture and better lighting."
            )
            heart_rate = None
            hr_confidence = 0.0
            retake_required = True
            warning = "No measurable pulse signal."
            suppression_reason = (
                f"Insufficient signal evidence (periodicity={periodicity:.3f}, "
                f"valid_windows={valid_windows}, quality={hr_quality:.3f})"
            )
        else:
            suppression_reason = None
            bpm = float(np.clip(bpm, 35.0, 220.0))
            risk, message = _classify_hr(bpm)
            heart_rate = round(bpm, 1)
            window_count_factor = float(hr_window_consensus.get("window_count_score", np.clip(valid_windows / 8.0, 0.0, 1.0)))
            stability_factor = float(hr_window_consensus.get("stability_score", 0.0))
            snr_factor = float(hr_window_consensus.get("snr_proxy", hr_quality))
            peak_quality_factor = float(hr_window_consensus.get("peak_quality_score", 0.0))
            cluster_tightness = float(hr_window_consensus.get("cluster_tightness", 0.0))
            run_delta = None
            heart_rate, run_delta = _apply_cross_run_smoothing(float(heart_rate), channel="stream")
            bpm = heart_rate
            base_conf = (
                0.10 * tracking_ratio
                + 0.08 * roi_agreement
                + 0.10 * periodicity
                + 0.10 * scan_quality
                + 0.16 * window_count_factor
                + 0.14 * snr_factor
                + 0.12 * peak_quality_factor
                + 0.20 * signal_strength
            )
            mean_oe = float(np.mean(self._overexposure_ratios)) if self._overexposure_ratios else 0.0
            oe_penalty = float(np.clip(mean_oe / 0.40, 0.0, 1.0))
            usable_ratio = self._valid_frame_count / max(self._frame_count, 1)
            usable_penalty = float(np.clip(1.0 - usable_ratio, 0.0, 1.0))
            hr_confidence = round(
                base_conf
                * (0.60 + 0.40 * stability_factor)
                * (0.65 + 0.35 * peak_quality_factor)
                * (0.65 + 0.35 * cluster_tightness)
                * (1.0 - 0.30 * oe_penalty)
                * (1.0 - 0.25 * usable_penalty),
                3,
            )
            if run_delta is not None:
                if run_delta <= 3.0:
                    hr_confidence = min(1.0, hr_confidence + 0.08)
                elif run_delta >= 8.0:
                    hr_confidence = max(0.0, hr_confidence - 0.08)
            hr_confidence = max(0.0, hr_confidence - _soft_penalty)
            dominant_ratio = float(hr_window_consensus.get("dominant_cluster_ratio", 0.0))
            if std_dev < 3.0:
                hr_confidence = min(1.0, hr_confidence + 0.06)
            if dominant_ratio >= 0.70:
                hr_confidence = min(1.0, hr_confidence + 0.05)
            hr_confidence = float(np.clip(hr_confidence, 0.0, 1.0))
            if estimated_from_weak_signal:
                hr_confidence = float(np.clip(hr_confidence, 0.10, 0.38))
                retake_required = True
                if "Low confidence" not in " ".join(retake_reasons):
                    retake_reasons.append("Low confidence estimated result; retake recommended.")
            elif bool(hr_window_consensus.get("unstable", False)):
                hr_confidence = float(np.clip(hr_confidence, 0.10, 0.30))
                warning = "High variability detected - result may not be reliable."
            elif periodicity < 0.30 or valid_windows < 4:
                hr_confidence = float(np.clip(hr_confidence, 0.10, 0.40))
                warning = "Low confidence result: estimated from weak signal."
            elif valid_windows < _MIN_CONSENSUS_WINDOWS:
                warning = (
                    "Window count is limited for weak signal. "
                    "Use a longer recording for higher stability."
                )

        reliability = _derive_reliability(
            confidence=hr_confidence,
            periodicity=periodicity,
            valid_windows=valid_windows,
            has_signal=has_min_signal,
        )
        if estimated_from_weak_signal and has_min_signal:
            reliability = "low"
        if reliability == "low" and warning is None and heart_rate is not None:
            warning = "Low confidence result: estimated from weak signal."
        if heart_rate is not None:
            if warning and "pulse" in warning.lower() and "no" in warning.lower():
                warning = "Low signal strength - estimate may vary."
            elif warning is None and (stability == "Low" or signal_strength < 0.35):
                warning = "Low signal strength - estimate may vary."
            retake_reasons = [
                r for r in retake_reasons if "could not detect" not in str(r).lower()
            ]

        blink_results = aggregate_blink_results(
            self._ear_values, self._blink_detector.blink_count, duration_sec
        )
        eye_movement = aggregate_eye_movement(self._gaze_offsets, duration_sec)
        facial_motion = aggregate_facial_motion(self._region_positions)
        eye_color = aggregate_eye_color(self._eye_color_samples)
        skin_analysis = aggregate_skin_analysis(self._skin_samples)

        return _emit_result({
            "module_name": "face_module",
            "scan_duration_sec": round(duration_sec, 2),
            "heart_rate": heart_rate,
            "stable_heart_rate": median_hr if median_hr is not None else heart_rate,
            "heart_rate_range": [hr_min, hr_max] if hr_min is not None and hr_max is not None else None,
            "stability": stability,
            "heart_rate_unit": "bpm",
            "heart_rate_confidence": hr_confidence,
            "blink_rate": blink_results.get("blink_rate_per_min"),
            "blink_rate_unit": "blinks/min",
            "eye_stability": round(eye_movement.get("eye_stability"), 3)
                if eye_movement.get("eye_stability") is not None else None,
            "facial_tension_index": round(facial_motion.get("facial_tension_index"), 3)
                if facial_motion.get("facial_tension_index") is not None else None,
            "skin_signal_stability": round(skin_analysis.get("skin_signal_stability"), 3)
                if skin_analysis.get("skin_signal_stability") is not None else None,
            "scan_quality": scan_quality,
            "retake_required": retake_required,
            "retake_reasons": retake_reasons,
            "confidence": hr_confidence,
            "hr_confidence": hr_confidence,
            "reliability": reliability,
            "hr_result_tier": hr_result_tier,
            "result_available": result_available,
            "retake_recommended": retake_required,
            "estimated_from_weak_signal": estimated_from_weak_signal,
            "signal_strength": round(signal_strength, 3),
            "periodicity_score": round(periodicity, 3),
            "valid_windows": valid_windows,
            "method_used": method_used,
            "warning": warning,
            "hr_rejection_reasons": rejection_codes,
            "confidence_breakdown": confidence_breakdown,
            "risk": risk,
            "message": message,
            "debug": {
                "frames_processed": self._frame_count,
                "valid_frames": self._valid_frame_count,
                "dropped_frames": max(self._frame_count - self._valid_frame_count, 0),
                "fps": round(self._fps, 1),
                "duration_sec": round(duration_sec, 2),
                "timing": timing_diag,
                "effective_fps": timing_diag.get("effective_fps"),
                "cadence_jitter_cv": timing_diag.get("cadence_jitter_cv"),
                "suppression_reason": suppression_reason,
                "usable_frame_count": self._valid_frame_count,
                "overexposed_reject_count": self._overexposed_reject_count,
                "low_brightness_reject_count": self._low_brightness_reject_count,
                "blur_reject_count": self._blur_reject_count,
                "motion_reject_count": self._motion_reject_count,
                "roi_instability_reject_count": self._roi_instability_reject_count,
                "signal_strength": round(signal_strength, 3),
                "periodicity_score": round(periodicity, 3),
                "valid_windows": valid_windows,
                "method_used": method_used,
                "hr_values": hr_window_consensus.get("hr_values", []),
                "median_hr": hr_window_consensus.get("median_hr"),
                "std_dev": hr_window_consensus.get("std_dev", 0.0),
                "variance": hr_window_consensus.get("variance", 0.0),
                "valid_window_count": hr_window_consensus.get("valid_window_count", 0),
                "selected_windows": hr_window_consensus.get("selected_windows", []),
                "rejected_windows": hr_window_consensus.get("rejected_windows", []),
                "peak_quality_score": hr_window_consensus.get("peak_quality_score", 0.0),
                "dominant_cluster_ratio": hr_window_consensus.get("dominant_cluster_ratio", 0.0),
                "dominant_cluster_std": hr_window_consensus.get("dominant_cluster_std", 0.0),
                "cluster_tightness": hr_window_consensus.get("cluster_tightness", 0.0),
                "heart_rate_range": [hr_min, hr_max] if hr_min is not None and hr_max is not None else None,
                "stability": stability,
                "final_method_used": method_used,
                "roi_adaptive_weights": _roi_adaptive_weights,
                "method_adaptive_weights": _method_adaptive_weights,
                "candidate_rejections": _candidate_rejections,
                "resampled_candidate_count": _resampled_candidate_count,
                "resampled_candidate_ratio": round(_resampled_candidate_ratio, 4),
                "candidate_effective_fps": _candidate_effective_fps,
                "recovery_triggered": _stream_recovery_triggered,
                "recovery_success": _stream_recovery_succeeded,
                "recovery": {k: v for k, v in _stream_recovery.items() if k != "attempt_log"}
                    if _stream_recovery else None,
                "hr_rejection_codes": rejection_codes,
                "final_acceptance_reason": (
                    f"{hr_result_tier}: periodicity={periodicity:.3f}, valid_windows={valid_windows}, signal_strength={signal_strength:.3f}"
                    if has_min_signal else f"result_unavailable: reasons={','.join(rejection_codes)}"
                ),
                "hr_result_tier": hr_result_tier,
                "result_available": result_available,
                "estimated_from_weak_signal": estimated_from_weak_signal,
                "rppg_model": {
                    "model_used": comparison.get("model_name", "none"),
                    "source": comparison.get("source", "signal_pipeline"),
                    "selection_reason": comparison.get("selection_reason", ""),
                },
                # Competition engine fields
                "selected_roi": _selected_roi,
                "selected_method": _selected_method,
                "competition_signal_strength": round(_comp_signal_strength, 3),
                "all_candidate_scores": hr_result.get("all_candidate_scores", []),
                "roi_scores": hr_result.get("roi_scores", {}),
                "method_scores": hr_result.get("method_scores", {}),
                "roi_skin_coverage": {k: round(v, 3) for k, v in _roi_skin_cov.items()},
            },
            "hr_timeseries": hr_timeseries,
            "blink_analysis": blink_results,
            "eye_movement": eye_movement,
            "facial_motion": facial_motion,
            "eye_color": eye_color,
            "skin_color": skin_analysis,
            "metric_name": "heart_rate",
            "value": heart_rate,
            "unit": "bpm",
        }, source="stream")

    # ------------------------------------------------------------------
    def finalise(self) -> Dict[str, Any]:
        """Force finalisation (call if the client disconnects early)."""
        if self._finalised_result is not None:
            return self._finalised_result
        self.is_done = True
        result = self._compute_final_result()
        self._finalised_result = result
        return result

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass


def analyze_face_stream(
    frame_iterator,           # iterable of BGR numpy frames
    fps: float = 30.0,
) -> "Generator[Dict[str, Any], None, None]":
    """Generator-based streaming pipeline.

    Yields ``"interim"`` event dicts periodically and a single
    ``"final"`` event when the scan completes.  Suitable for use with
    any iterator of BGR numpy frames (camera loop, WebSocket handler,
    test harness).

    Parameters
    ----------
    frame_iterator :
        Any iterable that yields BGR ``np.ndarray`` frames.
    fps : float
        Camera / video frames-per-second (used for time calculations).

    Yields
    ------
    dict
        ``{"event": "interim" | "final", "data": {...}}``

    Example
    -------
    ::

        proc = FaceStreamProcessor(fps=30)
        for event in analyze_face_stream(camera_frames(), fps=30):
            send_to_client(event)
    """
    proc = FaceStreamProcessor(fps=fps)
    try:
        for frame in frame_iterator:
            if proc.is_done:
                break
            event = proc.push_frame(frame)
            if event is not None:
                yield event
                if event["event"] == "final":
                    return
        # Iterator exhausted before target duration → force finalise
        if not proc.is_done:
            yield proc._build_final_event()
    finally:
        proc.close()
