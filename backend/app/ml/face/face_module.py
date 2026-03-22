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
    HEAD_ANCHORS,
    LEFT_CHEEK,
    LEFT_IRIS,
    REGION_LANDMARKS,
    RIGHT_CHEEK,
    apply_clahe,
    brightness_quality as _brightness_quality,
    denoise_frame,
    extract_anchor_positions,
    extract_face_roi_haar,
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
    build_skin_mask,
)
from backend.app.utils.signal_processing import (
    compute_hr_timeseries,
    compute_periodicity,
    compute_signal_strength,
    estimate_bpm,
    median_filter_bpm,
    robust_hr_consensus,
    select_best_windows,
)
from backend.app.ml.face.rppg_models import (
    compare_with_signal_pipeline,
    infer_rppg_models,
)
from backend.app.ml.face.rppg_recovery import recover_heart_rate

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_COLOR_SAMPLE_INTERVAL = 10   # sample iris / skin colour every N frames
_MOTION_REJECT_THRESHOLD = 15.0  # px – reject frame if head motion > this
_HR_WINDOW_SEC = 5.0
_HR_STRIDE_SEC = 0.5
_MAX_MODEL_FRAMES = 300  # max frames to keep for model inference
_MIN_CONSENSUS_WINDOWS = 2  # lowered from 4 — web/mobile cameras often produce only 2-4 quality windows

# Cross-run smoothing cache (process-local).
_LAST_HR_EMA: Dict[str, Optional[float]] = {"file": None, "stream": None}
_CROSS_RUN_ALPHA = 0.7


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


def _apply_cross_run_smoothing(current_hr: float, channel: str) -> Tuple[float, Optional[float]]:
    """EMA smoothing against previous scan result for run-to-run consistency."""
    prev = _LAST_HR_EMA.get(channel)
    if prev is None:
        _LAST_HR_EMA[channel] = current_hr
        return round(current_hr, 1), None
    smoothed = _CROSS_RUN_ALPHA * current_hr + (1.0 - _CROSS_RUN_ALPHA) * prev
    _LAST_HR_EMA[channel] = smoothed
    return round(float(smoothed), 1), round(abs(current_hr - prev), 2)


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
        return _build_error_result(
            "OpenCV is not installed. Cannot process video.",
            {"dependency_missing": "opencv-python"},
        )
    if not video_path or not os.path.isfile(str(video_path)):
        return _build_error_result(
            "Video file not found or path is invalid.",
            {"video_path": video_path},
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _build_error_result(
            "Could not open video file.",
            {"video_path": video_path},
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

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

    frames_read = 0
    face_detected_count = 0
    valid_frame_count = 0
    overexposed_reject_count = 0

    blink_detector = BlinkDetector()
    prev_anchors: Optional[np.ndarray] = None
    landmark_smoother = LandmarkSmoother(alpha=0.6)

    # ══════════════════════════════════════════════════════════════
    # Frame loop
    # ══════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        h, w = frame.shape[:2]

        # --- Frame-level brightness ---
        bright = frame_brightness(frame)
        bq = _brightness_quality(bright)
        brightness_scores.append(bq)

        # --- Frame-level overexposure ---
        oe = frame_overexposure_ratio(frame)
        overexposure_ratios.append(oe)

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
            timestamp_ms = int((frames_read / fps) * 1000)
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
                fq = _brightness_quality(bright) * (1.0 / (1.0 + motion_mag * 0.5))
                frame_qualities.append(fq)

                # Skip frame if motion too large (don't corrupt pulse signal)
                if motion_mag > _MOTION_REJECT_THRESHOLD:
                    continue

                # Skip frame if heavily overexposed (blown-out ROIs)
                if oe > 0.50:
                    overexposed_reject_count += 1
                    continue

                valid_frame_count += 1

                # Collect frames for pretrained model inference
                if len(model_frames) < _MAX_MODEL_FRAMES:
                    model_frames.append(frame.copy())

                # 1) Multi-ROI green-channel extraction
                roi_vals = extract_frame_roi_signals(frame, lm_list, h, w)
                for name in ROI_NAMES:
                    v = roi_vals.get(name)
                    if v is not None:
                        roi_traces[name].append(v)

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
        "frames_with_face": face_detected_count,
        "fps": round(fps, 1),
        "duration_sec": round(duration_sec, 2),
    }

    # --- Guard: too few frames ---
    if frames_read < FACE_MIN_FRAMES:
        return _build_error_result(
            f"Video too short ({frames_read} frames). Need at least {FACE_MIN_FRAMES}.",
            debug_info,
        )
    if face_detected_count < FACE_MIN_FRAMES:
        return _build_error_result(
            "Face not detected in enough frames for reliable estimation.",
            debug_info,
        )
    # Guard: too few *usable* frames (passes motion + overexposure filters)
    _MIN_USABLE_FRAMES = max(FACE_MIN_FRAMES // 2, 15)
    if valid_frame_count < _MIN_USABLE_FRAMES:
        return _build_error_result(
            f"Too few usable frames ({valid_frame_count}) after filtering motion "
            f"and overexposure. Need at least {_MIN_USABLE_FRAMES}. "
            "Please rescan in stable lighting and keep your head still.",
            debug_info,
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
        roi_traces, roi_weights, fps, RPPG_LOW_HZ, RPPG_HIGH_HZ,
        roi_rgb_traces=roi_rgb_traces,
        roi_overexposure=roi_overexposure,
        roi_skin_coverage=roi_skin_cov,
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
    debug_info["overexposed_reject_count"] = overexposed_reject_count

    # --- Pretrained rPPG model inference (optional layer) ---
    model_result = infer_rppg_models(model_frames, fps)
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
        scan_duration_sec=duration_sec,
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

    # ── Recovery mode ─────────────────────────────────────────────────────
    # Before declaring the scan unreliable, try multiple extraction strategies
    # on the same frame data.  Only triggered when signal quality is weak.
    _RECOVERY_TRIGGER_PERIOD = 0.18  # below this periodicity → attempt recovery
    _recovery_result: Optional[Dict[str, Any]] = None
    _recovery_triggered = (
        bpm is None or bpm <= 0
        or hr_quality < 0.05
        or periodicity < _RECOVERY_TRIGGER_PERIOD
    )
    if _recovery_triggered and valid_frame_count >= 20:
        _recovery_result = recover_heart_rate(
            roi_traces, roi_rgb_traces, fps, RPPG_LOW_HZ, RPPG_HIGH_HZ
        )
        if _recovery_result["recovery_success"]:
            # Use the best result found across all recovery stages.
            # This is still derived from real signal data — not fabricated.
            bpm = _recovery_result["bpm"]
            hr_quality = _recovery_result["quality"]
            periodicity = _recovery_result["periodicity"]
        # Always log recovery details (success or failure)
        debug_info["recovery"] = {
            k: v for k, v in _recovery_result.items() if k != "attempt_log"
        }
        debug_info["recovery_attempt_log"] = _recovery_result["attempt_log"]
    debug_info["recovery_triggered"] = _recovery_triggered
    _recovery_succeeded = (
        _recovery_result is not None and _recovery_result["recovery_success"]
    )

    # --- HR timeseries (spatially-averaged ROI signal for better weak-signal stability) ---
    averaged_trace = _build_spatially_averaged_trace(roi_traces)
    best_roi = max(roi_traces, key=lambda k: len(roi_traces[k]))
    source_trace = averaged_trace if averaged_trace else roi_traces[best_roi]
    hr_timeseries = compute_hr_timeseries(
        source_trace,
        fps,
        _HR_WINDOW_SEC,
        _HR_STRIDE_SEC,
        RPPG_LOW_HZ,
        RPPG_HIGH_HZ,
        min_window_quality=0.10,
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
    elif (
        hr_window_consensus.get("median_hr") is not None
        and float(hr_window_consensus.get("std_dev", 999)) < 8.0
        and int(hr_window_consensus.get("valid_window_count", 0)) >= 3
        and 45.0 <= float(hr_window_consensus["median_hr"]) <= 180.0
    ):
        # Soft fallback: cluster gate just barely failed but timeseries median is
        # stable enough to use.  More reliable than the raw competition winner.
        bpm = hr_window_consensus["median_hr"]

    hr_filtered_values = hr_window_consensus.get("filtered_hr_values", [])
    hr_min = round(min(hr_filtered_values), 1) if hr_filtered_values else None
    hr_max = round(max(hr_filtered_values), 1) if hr_filtered_values else None
    median_hr = hr_window_consensus.get("median_hr")
    std_dev = float(hr_window_consensus.get("std_dev", 0.0))
    stability = _stability_label(std_dev)

    # --- Determine final HR result ---
    _MIN_SIGNAL_PERIODICITY = 0.10  # lowered — accept weaker periodic signals
    _MIN_WEAK_WINDOWS = 1  # lowered — even 1 valid window is enough for a low-confidence estimate
    windows_from_recovery = int((_recovery_result or {}).get("windows_accepted", 0))
    valid_windows = max(int(hr_window_consensus.get("valid_window_count", 0)), windows_from_recovery)
    method_used = (
        (_recovery_result or {}).get("extraction_method_used", "multi_roi_primary")
        if _recovery_succeeded else "multi_roi_primary"
    )
    if hr_window_consensus.get("has_consensus"):
        method_used = f"window_consensus_{hr_window_consensus.get('method', 'median')}"
    has_min_signal = (
        bpm is not None
        and bpm > 0
        and (
            periodicity >= _MIN_SIGNAL_PERIODICITY
            or valid_windows >= _MIN_WEAK_WINDOWS
        )
    )

    suppression_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    warning: Optional[str] = None
    # Prefer competition engine signal_strength; fall back to old formula
    if competition_signal_strength > 0.0:
        signal_strength = float(np.clip(competition_signal_strength, 0.0, 1.0))
    else:
        signal_strength = float(np.clip(0.6 * max(hr_quality, 0.0) + 0.4 * max(periodicity, 0.0), 0.0, 1.0))

    # --- Signal validation gate (soft degradation model) ---
    # Instead of a binary kill-switch, weak-but-present signals are kept
    # with degraded confidence so the score engine can fuse them at low
    # weight.  Only truly catastrophic signals (no BPM / extreme noise)
    # are hard-rejected.
    peak_support_count = int(hr_window_consensus.get("dominant_cluster_size", 0))
    _soft_penalty = 0.0  # accumulated confidence penalty for soft issues
    if has_min_signal:
        _gate_std_dev = float(hr_window_consensus.get("std_dev", 0.0))
        _gate_valid_windows = int(hr_window_consensus.get("valid_window_count", valid_windows))

        # --- Hard fail: truly unusable (extreme noise or no signal) ---
        if _gate_std_dev > 15.0:
            rejection_reason = (
                f"HR variability extremely high (std_dev={_gate_std_dev:.1f} bpm > 15.0 bpm). "
                "Please hold very still with steady lighting."
            )
            has_min_signal = False
        elif signal_strength < 0.08:
            rejection_reason = (
                f"rPPG signal not detected (signal_strength={signal_strength:.2f} < 0.08). "
                "Ensure good lighting and keep your face centred in frame."
            )
            has_min_signal = False
        else:
            # --- Soft penalties: degrade confidence but keep the HR ---
            if _gate_std_dev > 8.0:
                _soft_penalty += 0.20
                warning = "High HR variability — result may be less accurate."
            if _gate_valid_windows < 3:
                _soft_penalty += 0.15
            if peak_support_count < 2:
                _soft_penalty += 0.10
            if signal_strength < 0.30:
                _soft_penalty += 0.15

    if not has_min_signal:
        risk = "unreliable"
        if rejection_reason:
            message = rejection_reason
        else:
            message = (
                "Unable to estimate heart rate: no usable pulse signal was detected "
                "in this scan. Please retry with steady posture and better lighting."
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
    else:
        bpm = float(np.clip(bpm, 35.0, 220.0))
        risk, message = _classify_hr(bpm)
        heart_rate = round(bpm, 1)
        snr_factor = float(hr_window_consensus.get("snr_proxy", hr_quality))
        peak_quality_factor = float(hr_window_consensus.get("peak_quality_score", 0.0))
        cluster_tightness = float(hr_window_consensus.get("cluster_tightness", 0.0))
        run_delta = None
        # Always apply cross-run EMA for cross-scan stability.
        # Previously, EMA was skipped when consensus failed, letting the raw
        # competition-engine BPM fluctuate between harmonics (e.g. 85/73/49).
        heart_rate, run_delta = _apply_cross_run_smoothing(float(heart_rate), channel="file")
        bpm = heart_rate
        valid_window_ratio = float(np.clip(valid_windows / 8.0, 0.0, 1.0))
        hr_confidence = (
            0.25 * signal_strength
            + 0.25 * peak_quality_factor
            + 0.20 * snr_factor
            + 0.15 * cluster_tightness
            + 0.15 * valid_window_ratio
        )
        # Cross-run consistency adjustment
        if run_delta is not None:
            if run_delta <= 3.0:
                hr_confidence = min(1.0, hr_confidence + 0.05)
            elif run_delta >= 8.0:
                hr_confidence = max(0.0, hr_confidence - 0.05)
        # Apply soft penalties from validation gate
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
        hr_confidence = float(np.clip(hr_confidence, 0.0, 1.0))

        # Weak-but-real signals are returned with low confidence, never suppressed.
        if bool(hr_window_consensus.get("unstable", False)):
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

        if _recovery_succeeded:
            hr_confidence = round(min(hr_confidence, 0.75), 3)
            method = _recovery_result.get("extraction_method_used", "multi-strategy")
            roi = _recovery_result.get("roi_used", "best")
            message += f" (Recovered via {method} on {roi} ROI.)"

    reliability = _derive_reliability(
        confidence=hr_confidence,
        periodicity=periodicity,
        valid_windows=valid_windows,
        has_signal=has_min_signal,
    )
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
        "signal_strength": round(signal_strength, 3),
        "periodicity_score": round(periodicity, 3),
        "valid_windows": valid_windows,
        "method_used": method_used,
        "warning": warning,
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
        self._blink_detector = BlinkDetector()
        self._prev_anchors: Optional[np.ndarray] = None
        self._overexposed_reject_count = 0
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
    def push_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
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
                self._frame_qualities.append(fq)

                if motion_mag <= _MOTION_REJECT_THRESHOLD:
                    if oe > 0.50:
                        self._overexposed_reject_count += 1
                    else:
                        self._valid_frame_count += 1

                        if len(self._model_frames) < _STREAM_MAX_BUFFER:
                            self._model_frames.append(frame.copy())

                        roi_vals = extract_frame_roi_signals(frame, lm_list, h, w)
                        for name in ROI_NAMES:
                            v = roi_vals.get(name)
                            if v is not None:
                                self._roi_traces[name].append(v)

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
                            _skin_mask = build_skin_mask(frame)
                            _roi_funcs = {
                                "forehead": extract_forehead_roi,
                                "left_cheek": extract_cheek_roi,
                                "right_cheek": extract_cheek_roi,
                            }
                            for _rname in ROI_NAMES:
                                _rfn = _roi_funcs.get(_rname)
                                if _rfn is None:
                                    continue
                                try:
                                    if _rname == "right_cheek":
                                        _roi_patch = _rfn(frame, lm_list, h, w, side="right")
                                    elif _rname == "left_cheek":
                                        _roi_patch = _rfn(frame, lm_list, h, w, side="left")
                                    else:
                                        _roi_patch = _rfn(frame, lm_list, h, w)
                                    if _roi_patch is not None and _roi_patch.size > 0:
                                        _rmask = _skin_mask[
                                            :_roi_patch.shape[0], :_roi_patch.shape[1]
                                        ] if _skin_mask is not None else None
                                        _qm = compute_roi_quality_metrics(_roi_patch, _rmask)
                                        self._roi_skin_quality[_rname].append(_qm["quality_score"])
                                except Exception:
                                    pass

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

        if self._frame_count < FACE_MIN_FRAMES or self._valid_frame_count < FACE_MIN_FRAMES:
            return _build_error_result(
                "Not enough valid frames captured for a reliable estimate.",
                {"frames": self._frame_count, "valid": self._valid_frame_count},
            )
        _MIN_USABLE_FRAMES = max(FACE_MIN_FRAMES // 2, 15)
        if self._valid_frame_count < _MIN_USABLE_FRAMES:
            return _build_error_result(
                f"Too few usable frames ({self._valid_frame_count}) after filtering "
                f"motion and overexposure. Need at least {_MIN_USABLE_FRAMES}. "
                "Please rescan in stable lighting and keep your head still.",
                {"frames": self._frame_count, "valid": self._valid_frame_count},
            )

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
        )
        bpm = hr_result["bpm"]
        hr_quality = hr_result["quality"]
        roi_agreement = hr_result["roi_agreement"]
        periodicity = hr_result["periodicity"]
        _comp_signal_strength = float(hr_result.get("signal_strength", 0.0))
        _selected_roi = hr_result.get("selected_roi")
        _selected_method = hr_result.get("selected_method")

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

        # ── Recovery mode (streaming path) ──────────────────────────────
        _RECOVERY_TRIGGER_PERIOD = 0.18
        _stream_recovery: Optional[Dict[str, Any]] = None
        _stream_recovery_triggered = (
            bpm is None or bpm <= 0
            or hr_quality < 0.05
            or periodicity < _RECOVERY_TRIGGER_PERIOD
        )
        if _stream_recovery_triggered and self._valid_frame_count >= 20:
            _stream_recovery = recover_heart_rate(
                self._roi_traces, self._roi_rgb_traces, self._fps,
                RPPG_LOW_HZ, RPPG_HIGH_HZ,
            )
            if _stream_recovery["recovery_success"]:
                bpm = _stream_recovery["bpm"]
                hr_quality = _stream_recovery["quality"]
                periodicity = _stream_recovery["periodicity"]
        _stream_recovery_succeeded = (
            _stream_recovery is not None and _stream_recovery["recovery_success"]
        )

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

        _MIN_SIGNAL_PERIODICITY = 0.20
        _MIN_WEAK_WINDOWS = 2
        windows_from_recovery = int((_stream_recovery or {}).get("windows_accepted", 0))
        valid_windows = max(int(hr_window_consensus.get("valid_window_count", 0)), windows_from_recovery)
        method_used = (
            (_stream_recovery or {}).get("extraction_method_used", "multi_roi_primary")
            if _stream_recovery_succeeded else "multi_roi_primary"
        )
        if hr_window_consensus.get("has_consensus"):
            method_used = f"window_consensus_{hr_window_consensus.get('method', 'median')}"
        has_min_signal = (
            bpm is not None
            and bpm > 0
            and (
                periodicity >= _MIN_SIGNAL_PERIODICITY
                or valid_windows >= _MIN_WEAK_WINDOWS
            )
        )
        signal_strength = (
            float(np.clip(_comp_signal_strength, 0.0, 1.0))
            if _comp_signal_strength > 0.0
            else float(np.clip(0.6 * max(hr_quality, 0.0) + 0.4 * max(periodicity, 0.0), 0.0, 1.0))
        )
        warning: Optional[str] = None

        if not has_min_signal:
            risk = "unreliable"
            message = (
                "Unable to estimate heart rate: no usable pulse signal was detected "
                "in this scan. Please retry with steady posture and better lighting."
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
            if hr_window_consensus.get("has_consensus"):
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
            dominant_ratio = float(hr_window_consensus.get("dominant_cluster_ratio", 0.0))
            if std_dev < 3.0:
                hr_confidence = min(1.0, hr_confidence + 0.06)
            if dominant_ratio >= 0.70:
                hr_confidence = min(1.0, hr_confidence + 0.05)
            hr_confidence = float(np.clip(hr_confidence, 0.0, 1.0))
            if bool(hr_window_consensus.get("unstable", False)):
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
            if _stream_recovery_succeeded:
                hr_confidence = round(min(hr_confidence, 0.75), 3)
                method = _stream_recovery.get("extraction_method_used", "multi-strategy")
                message += f" (Recovered via {method}.)"

        reliability = _derive_reliability(
            confidence=hr_confidence,
            periodicity=periodicity,
            valid_windows=valid_windows,
            has_signal=has_min_signal,
        )
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

        return {
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
            "signal_strength": round(signal_strength, 3),
            "periodicity_score": round(periodicity, 3),
            "valid_windows": valid_windows,
            "method_used": method_used,
            "warning": warning,
            "confidence_breakdown": confidence_breakdown,
            "risk": risk,
            "message": message,
            "debug": {
                "frames_processed": self._frame_count,
                "valid_frames": self._valid_frame_count,
                "fps": round(self._fps, 1),
                "duration_sec": round(duration_sec, 2),
                "suppression_reason": suppression_reason,
                "usable_frame_count": self._valid_frame_count,
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
                "recovery_triggered": _stream_recovery_triggered,
                "recovery_success": _stream_recovery_succeeded,
                "recovery": {k: v for k, v in _stream_recovery.items() if k != "attempt_log"}
                    if _stream_recovery else None,
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
        }

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
