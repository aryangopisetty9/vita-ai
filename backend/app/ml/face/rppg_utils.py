"""
Vita AI – rPPG Utilities
==========================
Multi-ROI rPPG signal extraction and heart-rate estimation helpers.

This module bridges vision_utils (ROI extraction) and
signal_processing (filtering / BPM estimation) to provide a clean
interface for the face module.

Features
--------
* Multi-ROI extraction (forehead + left cheek + right cheek).
* POS (Plane-Orthogonal-to-Skin) colour projection for robust pulse
  extraction that handles skin-tone variation better than green-only.
* Per-ROI quality scoring by amplitude, stability, and periodicity.
* Intelligent ROI dropping — corrupted or weak ROIs are excluded
  from fusion rather than dragging down the estimate.

Extension points
----------------
* Replace ``extract_rppg_signals`` with a deep-rPPG encoder
  (Open-rPPG) that takes raw
  frames and outputs a pulse waveform directly.
* Add SpO₂ estimation by comparing red and infrared channel ratios
  (requires NIR camera or dual-LED setup).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.ml.face.vision_utils import (
    LEFT_CHEEK,
    RIGHT_CHEEK,
    extract_cheek_roi,
    extract_forehead_roi,
    mean_green_from_roi_tiled,
    mean_rgb_from_roi_tiled,
    build_skin_mask,
    overexposure_ratio,
    compute_roi_quality_metrics,
)
from backend.app.utils.signal_processing import (
    chrom_project,
    compute_hr_timeseries,
    compute_periodicity,
    compute_signal_strength,
    estimate_bpm,
    fuse_roi_signals,
    lgi_project,
    spectral_hr_estimate,
    moving_average_smooth,
    pos_project,
    signal_amplitude,
    signal_color_variance,
    signal_illumination_stability,
    signal_stability,
    detrend_signal,
    temporal_normalization,
    bandpass_fft,
    robust_hr_consensus,
)
from backend.app.ml.face.signal_quality_model import (
    evaluate_signal_quality,
    extract_quality_features,
)


# ROI names used throughout the pipeline
ROI_NAMES = ("forehead", "left_cheek", "right_cheek")

# Minimum per-ROI quality to include in fusion (below = dropped)
_MIN_ROI_QUALITY = 0.08

# Maximum allowed overexposure ratio on an ROI to accept it
_MAX_ROI_OVEREXPOSURE = 0.40
_CANDIDATE_MIN_SIGNAL_STRENGTH = 0.03
_CANDIDATE_MIN_PERIODICITY = 0.03
_CANDIDATE_MIN_VALID_WINDOWS = 1
_CANDIDATE_MAX_WINDOW_STD_NO_CONSENSUS = 22.0
_MIN_EFFECTIVE_FPS = 12.0
_TARGET_RESAMPLE_FPS = 15.0


def _apply_skin_mask(roi_crop: np.ndarray, mask_crop: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Combine an optional polygon mask with a skin-colour mask.

    Returns the combined mask or None if no skin pixels survive.
    """
    skin = build_skin_mask(roi_crop)
    # Reject very dark shadows and clipped highlights inside ROI.
    luma = (
        0.114 * roi_crop[:, :, 0].astype(np.float64)
        + 0.587 * roi_crop[:, :, 1].astype(np.float64)
        + 0.299 * roi_crop[:, :, 2].astype(np.float64)
    )
    tone_ok = ((luma >= 22.0) & (luma <= 245.0)).astype(np.uint8) * 255
    skin = skin & tone_ok
    if mask_crop is not None:
        combined = skin & mask_crop
    else:
        combined = skin
    if int(np.sum(combined > 0)) < 10:
        return None
    return combined


def extract_frame_roi_signals(
    frame_bgr: np.ndarray,
    lm_list,
    h: int,
    w: int,
) -> Dict[str, Optional[float]]:
    """Extract green-channel mean from each facial ROI for one frame.

    A skin-colour mask is applied to exclude non-skin pixels (hair,
    background leakage, specular highlights) from the mean.

    Returns {roi_name: green_mean_or_None}.
    """
    signals: Dict[str, Optional[float]] = {}

    # Forehead
    fh_roi = extract_forehead_roi(frame_bgr, lm_list, h, w)
    if fh_roi is not None and fh_roi.size > 0:
        skin_m = _apply_skin_mask(fh_roi)
        signals["forehead"] = mean_green_from_roi_tiled(fh_roi, skin_m)
    else:
        signals["forehead"] = None

    # Left cheek
    lc = extract_cheek_roi(frame_bgr, lm_list, LEFT_CHEEK, h, w)
    if lc is not None:
        roi_crop, mask_crop = lc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["left_cheek"] = mean_green_from_roi_tiled(roi_crop, combined)
    else:
        signals["left_cheek"] = None

    # Right cheek
    rc = extract_cheek_roi(frame_bgr, lm_list, RIGHT_CHEEK, h, w)
    if rc is not None:
        roi_crop, mask_crop = rc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["right_cheek"] = mean_green_from_roi_tiled(roi_crop, combined)
    else:
        signals["right_cheek"] = None

    return signals


def extract_frame_roi_rgb(
    frame_bgr: np.ndarray,
    lm_list,
    h: int,
    w: int,
) -> Dict[str, Optional[np.ndarray]]:
    """Extract mean [R, G, B] from each facial ROI for one frame.

    A skin-colour mask is applied so only skin pixels contribute to the
    mean (consistent with ``extract_frame_roi_signals``).

    Used by the POS colour projection pipeline.
    Returns {roi_name: np.array([R, G, B]) or None}.
    """
    signals: Dict[str, Optional[np.ndarray]] = {}

    # Forehead
    fh_roi = extract_forehead_roi(frame_bgr, lm_list, h, w)
    if fh_roi is not None and fh_roi.size > 0:
        skin_m = _apply_skin_mask(fh_roi)
        signals["forehead"] = mean_rgb_from_roi_tiled(fh_roi, skin_m)
    else:
        signals["forehead"] = None

    # Left cheek
    lc = extract_cheek_roi(frame_bgr, lm_list, LEFT_CHEEK, h, w)
    if lc is not None:
        roi_crop, mask_crop = lc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["left_cheek"] = mean_rgb_from_roi_tiled(roi_crop, combined)
    else:
        signals["left_cheek"] = None

    # Right cheek
    rc = extract_cheek_roi(frame_bgr, lm_list, RIGHT_CHEEK, h, w)
    if rc is not None:
        roi_crop, mask_crop = rc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["right_cheek"] = mean_rgb_from_roi_tiled(roi_crop, combined)
    else:
        signals["right_cheek"] = None

    return signals


def extract_frame_overexposure(
    frame_bgr: np.ndarray,
    lm_list,
    h: int,
    w: int,
) -> Dict[str, float]:
    """Return per-ROI overexposure ratio for one frame."""
    result: Dict[str, float] = {}

    fh_roi = extract_forehead_roi(frame_bgr, lm_list, h, w)
    result["forehead"] = overexposure_ratio(fh_roi) if fh_roi is not None else 0.0

    lc = extract_cheek_roi(frame_bgr, lm_list, LEFT_CHEEK, h, w)
    if lc is not None:
        roi_crop, mask_crop = lc
        result["left_cheek"] = overexposure_ratio(roi_crop, mask_crop)
    else:
        result["left_cheek"] = 0.0

    rc = extract_cheek_roi(frame_bgr, lm_list, RIGHT_CHEEK, h, w)
    if rc is not None:
        roi_crop, mask_crop = rc
        result["right_cheek"] = overexposure_ratio(roi_crop, mask_crop)
    else:
        result["right_cheek"] = 0.0

    return result


def _compute_roi_signal_quality(
    signal: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    overexposure_mean: float = 0.0,
    skin_coverage: float = 1.0,
) -> Tuple[float, float, float]:
    """Composite per-ROI quality score combining multiple factors.

    Returns
    -------
    (quality, color_variance, illumination_stability)
    where quality is 0-1 (higher = more trustworthy).
    """
    if len(signal) < 8:
        return 0.0, 0.0, 0.0

    amp = signal_amplitude(signal)
    stab = signal_stability(signal, fps)
    period = compute_periodicity(signal, fps, low_hz, high_hz)
    color_var = signal_color_variance(signal)
    illum_stab = signal_illumination_stability(signal, fps)

    # Amplitude score: very low amplitude signals are likely noise
    amp_score = float(np.clip(amp / 5.0, 0.0, 1.0))

    # Overexposure penalty
    oe_penalty = float(np.clip(1.0 - overexposure_mean / _MAX_ROI_OVEREXPOSURE, 0.0, 1.0))

    # Skin coverage bonus
    skin_score = float(np.clip(skin_coverage, 0.0, 1.0))

    # Weighted combination — illumination stability gates unreliable ROIs
    quality = (
        0.20 * amp_score
        + 0.15 * stab
        + 0.20 * period
        + 0.15 * oe_penalty
        + 0.10 * skin_score
        + 0.10 * color_var
        + 0.10 * illum_stab
    )
    return float(np.clip(quality, 0.0, 1.0)), color_var, illum_stab


def _resample_uniform_signal(
    values: np.ndarray,
    timestamps: Optional[np.ndarray],
    fallback_fps: float,
) -> Tuple[np.ndarray, float, bool]:
    """Resample non-uniformly sampled values onto a uniform time grid."""
    if timestamps is None or len(values) < 8:
        return values, fallback_fps, False

    n = min(len(values), len(timestamps))
    if n < 8:
        return values, fallback_fps, False

    v = values[:n].astype(np.float64)
    t = timestamps[:n].astype(np.float64)
    t = t - t[0]

    # Keep strictly increasing timestamps only.
    keep = np.concatenate(([True], np.diff(t) > 1e-4))
    t = t[keep]
    v = v[keep]
    if len(v) < 8:
        return values, fallback_fps, False

    dt = np.median(np.diff(t)) if len(t) >= 2 else 0.0
    if dt <= 0:
        return values, fallback_fps, False

    est_fps = float(np.clip(1.0 / dt, 8.0, 120.0))
    # Trigger resampling only if cadence drift is non-trivial.
    if abs(est_fps - fallback_fps) / max(fallback_fps, 1e-6) < 0.05:
        return v, fallback_fps, False

    step = 1.0 / est_fps
    uniform_t = np.arange(0.0, t[-1] + 1e-12, step, dtype=np.float64)
    if len(uniform_t) < 8:
        return v, fallback_fps, False
    v_u = np.interp(uniform_t, t, v)
    return v_u.astype(np.float64), est_fps, True


def _resample_uniform_rgb(
    rgb_arr: np.ndarray,
    timestamps: Optional[np.ndarray],
    fallback_fps: float,
) -> Tuple[np.ndarray, float, bool]:
    """Resample RGB traces onto a uniform time grid."""
    if timestamps is None or len(rgb_arr) < 8:
        return rgb_arr, fallback_fps, False

    n = min(len(rgb_arr), len(timestamps))
    if n < 8:
        return rgb_arr, fallback_fps, False

    rgb = rgb_arr[:n].astype(np.float64)
    t = timestamps[:n].astype(np.float64)
    t = t - t[0]

    keep = np.concatenate(([True], np.diff(t) > 1e-4))
    t = t[keep]
    rgb = rgb[keep]
    if len(rgb) < 8:
        return rgb_arr, fallback_fps, False

    dt = np.median(np.diff(t)) if len(t) >= 2 else 0.0
    if dt <= 0:
        return rgb_arr, fallback_fps, False

    est_fps = float(np.clip(1.0 / dt, 8.0, 120.0))
    if abs(est_fps - fallback_fps) / max(fallback_fps, 1e-6) < 0.05:
        return rgb, fallback_fps, False

    step = 1.0 / est_fps
    uniform_t = np.arange(0.0, t[-1] + 1e-12, step, dtype=np.float64)
    if len(uniform_t) < 8:
        return rgb, fallback_fps, False

    out = np.zeros((len(uniform_t), 3), dtype=np.float64)
    for ch in range(3):
        out[:, ch] = np.interp(uniform_t, t, rgb[:, ch])
    return out, est_fps, True


def _upsample_to_target_fps_signal(
    values: np.ndarray,
    current_fps: float,
    target_fps: float,
) -> Tuple[np.ndarray, float, bool]:
    """Linearly upsample 1-D signal when effective FPS is too low."""
    if len(values) < 8 or current_fps <= 0 or current_fps >= target_fps:
        return values, current_fps, False
    duration = float((len(values) - 1) / max(current_fps, 1e-6))
    if duration <= 0:
        return values, current_fps, False
    src_t = np.arange(len(values), dtype=np.float64) / max(current_fps, 1e-6)
    dst_t = np.arange(0.0, duration + 1e-12, 1.0 / target_fps, dtype=np.float64)
    if len(dst_t) < 8:
        return values, current_fps, False
    out = np.interp(dst_t, src_t, values.astype(np.float64))
    return out.astype(np.float64), target_fps, True


def _upsample_to_target_fps_rgb(
    rgb_arr: Optional[np.ndarray],
    current_fps: float,
    target_fps: float,
) -> Tuple[Optional[np.ndarray], float, bool]:
    """Linearly upsample RGB traces when effective FPS is too low."""
    if rgb_arr is None or len(rgb_arr) < 8 or current_fps <= 0 or current_fps >= target_fps:
        return rgb_arr, current_fps, False
    duration = float((len(rgb_arr) - 1) / max(current_fps, 1e-6))
    if duration <= 0:
        return rgb_arr, current_fps, False
    src_t = np.arange(len(rgb_arr), dtype=np.float64) / max(current_fps, 1e-6)
    dst_t = np.arange(0.0, duration + 1e-12, 1.0 / target_fps, dtype=np.float64)
    if len(dst_t) < 8:
        return rgb_arr, current_fps, False
    out = np.zeros((len(dst_t), 3), dtype=np.float64)
    rgb = rgb_arr.astype(np.float64)
    for ch in range(3):
        out[:, ch] = np.interp(dst_t, src_t, rgb[:, ch])
    return out, target_fps, True


def _build_candidate_signal(
    method: str,
    green_signal: np.ndarray,
    rgb_arr: Optional[np.ndarray],
    fps: float,
    low_hz: float,
    high_hz: float,
    timestamps: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], float, bool]:
    """Build a pulse signal for a given extraction method.

    Methods: 'green', 'pos', 'chrom', 'pca'.

    Each method applies only the preprocessing it requires; functions
    that already normalise internally (POS, CHROM, PCA/LGI) receive
    only the remaining steps to avoid double-normalization.
    """
    green_sig, eff_fps, green_rs = _resample_uniform_signal(green_signal, timestamps, fps)
    rgb_eff, rgb_fps, rgb_rs = _resample_uniform_rgb(rgb_arr, timestamps, fps) if rgb_arr is not None else (None, fps, False)

    # Low-FPS protection: upsample to a safe analysis cadence for frequency estimation.
    low_fps_resampled = False
    if eff_fps < _MIN_EFFECTIVE_FPS:
        green_sig, eff_fps, up_g = _upsample_to_target_fps_signal(green_sig, eff_fps, _TARGET_RESAMPLE_FPS)
        low_fps_resampled = low_fps_resampled or up_g
    if rgb_eff is not None and rgb_fps < _MIN_EFFECTIVE_FPS:
        rgb_eff, rgb_fps, up_rgb = _upsample_to_target_fps_rgb(rgb_eff, rgb_fps, _TARGET_RESAMPLE_FPS)
        low_fps_resampled = low_fps_resampled or up_rgb

    if method == "green":
        sig = green_sig.copy()
        # Green needs full preprocessing: MA → detrend → temporal normalize → bandpass
        ma_window = max(3, int(eff_fps * 0.12))
        sig = moving_average_smooth(sig.astype(np.float64), window=ma_window)
        sig = detrend_signal(sig)
        sig = temporal_normalization(sig, window=max(int(eff_fps * 2), 8))
        sig = bandpass_fft(sig, eff_fps, low_hz, high_hz)
    elif method == "pos" and rgb_eff is not None and len(rgb_eff) >= 8:
        eff_fps = rgb_fps
        sig = pos_project(rgb_eff, eff_fps)
        # POS applies per-window temporal normalization internally (overlap-add);
        # only detrend + bandpass here to avoid double normalization.
        if len(sig) < 8:
            return None, eff_fps, (green_rs or rgb_rs or low_fps_resampled)
        sig = detrend_signal(sig.astype(np.float64))
        sig = bandpass_fft(sig, eff_fps, low_hz, high_hz)
    elif method == "chrom" and rgb_eff is not None and len(rgb_eff) >= 8:
        eff_fps = rgb_fps
        sig = chrom_project(rgb_eff, eff_fps)
        # CHROM applies temporal normalization + detrend internally;
        # only bandpass here to avoid triple normalization.
        if len(sig) < 8:
            return None, eff_fps, (green_rs or rgb_rs or low_fps_resampled)
        sig = bandpass_fft(sig.astype(np.float64), eff_fps, low_hz, high_hz)
    elif method == "pca" and rgb_eff is not None and len(rgb_eff) >= 8:
        eff_fps = rgb_fps
        sig = lgi_project(rgb_eff, eff_fps, low_hz=low_hz, high_hz=high_hz)
        # PCA/LGI applies full preprocessing internally (detrend + temporal norm + bandpass).
        if len(sig) < 8:
            return None, eff_fps, (green_rs or rgb_rs or low_fps_resampled)
    elif method == "intensity" and rgb_eff is not None and len(rgb_eff) >= 8:
        eff_fps = rgb_fps
        intensity = np.mean(rgb_eff, axis=1)
        sig = detrend_signal(intensity.astype(np.float64))
        sig = temporal_normalization(sig, window=max(int(eff_fps * 2), 8))
        sig = bandpass_fft(sig, eff_fps, low_hz, high_hz)
    else:
        return None, eff_fps, (green_rs or rgb_rs or low_fps_resampled)
    if len(sig) < 8:
        return None, eff_fps, (green_rs or rgb_rs or low_fps_resampled)
    sig = _suppress_outlier_jumps(sig)
    return sig, eff_fps, (green_rs or rgb_rs or low_fps_resampled)


def _suppress_outlier_jumps(signal: np.ndarray) -> np.ndarray:
    """Suppress abrupt sample jumps that are unlikely to be physiological."""
    if len(signal) < 8:
        return signal
    s = signal.astype(np.float64).copy()
    d = np.diff(s)
    if len(d) < 4:
        return s
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad < 1e-9:
        return s
    thresh = 6.0 * mad
    bad = np.where(np.abs(d - med) > thresh)[0] + 1
    for idx in bad.tolist():
        if 1 <= idx < len(s) - 1:
            s[idx] = 0.5 * (s[idx - 1] + s[idx + 1])
    return s


_EXTRACTION_METHODS = ("green", "pos", "chrom", "pca", "intensity")


def estimate_heart_rate_multi_roi(
    roi_traces: Dict[str, List[float]],
    roi_weights: Dict[str, float],
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    roi_rgb_traces: Optional[Dict[str, List[np.ndarray]]] = None,
    roi_overexposure: Optional[Dict[str, List[float]]] = None,
    roi_skin_coverage: Optional[Dict[str, float]] = None,
    roi_timestamps: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Full ROI × method competition engine.

    Evaluates up to 9 candidates (3 ROIs × 3 methods: green/POS/CHROM),
    scores each by signal_strength, selects the best single candidate,
    and also computes a weighted multi-ROI fusion as a fallback.

    Returns a rich result dict with the selected candidate info, all
    candidate scores, and legacy-compatible fields.
    """
    # ── Phase 1: collect surviving ROIs ──────────────────────────
    roi_signals: Dict[str, np.ndarray] = {}
    per_roi_signal_quality: Dict[str, float] = {}
    per_roi_color_var: Dict[str, float] = {}
    per_roi_illum_stab: Dict[str, float] = {}
    per_roi_overexposure: Dict[str, float] = {}
    rois_dropped: List[str] = []

    for name, trace in roi_traces.items():
        if len(trace) < 8:
            continue
        arr = np.array(trace, dtype=np.float64)
        oe_mean = 0.0
        if roi_overexposure and name in roi_overexposure:
            oe_vals = roi_overexposure[name]
            if oe_vals:
                oe_mean = float(np.mean(oe_vals))
        per_roi_overexposure[name] = round(oe_mean, 4)
        skin_cov = 1.0
        if roi_skin_coverage and name in roi_skin_coverage:
            skin_cov = roi_skin_coverage[name]
        sq, color_var, illum_stab = _compute_roi_signal_quality(
            arr, fps, low_hz, high_hz, oe_mean, skin_cov
        )
        per_roi_signal_quality[name] = round(sq, 4)
        per_roi_color_var[name] = round(color_var, 4)
        per_roi_illum_stab[name] = round(illum_stab, 4)
        # Drop ROIs with unstable illumination (strong lamp flicker / motion)
        if illum_stab < 0.20:
            rois_dropped.append(name)
            continue
        if sq < _MIN_ROI_QUALITY:
            rois_dropped.append(name)
            continue
        if oe_mean > _MAX_ROI_OVEREXPOSURE:
            rois_dropped.append(name)
            continue
        roi_signals[name] = arr

    if not roi_signals:
        return {
            "bpm": None, "quality": 0.0,
            "per_roi_quality": per_roi_signal_quality,
            "per_roi_bpm": {}, "roi_agreement": 0.0,
            "periodicity": 0.0, "pos_used": False,
            "rois_dropped": rois_dropped,
            "selected_roi": None, "selected_method": None,
            "signal_strength": 0.0,
            "all_candidate_scores": [],
            "roi_scores": per_roi_signal_quality,
            "roi_color_variance": per_roi_color_var,
            "roi_illumination_stability": per_roi_illum_stab,
            "method_scores": {},
            "harmonic_checked": False,
            "harmonic_corrected": False,
            "original_hr": None,
            "corrected_hr": None,
            "peak_prominence": 0.0,
            "peak_support_count": 0,
        }

    # ── Phase 2: build candidate signals (ROI × method) ─────────
    candidates: List[Dict[str, Any]] = []
    candidate_rejections: List[Dict[str, Any]] = []

    # Adaptive ROI weights from measurable quality evidence.
    adaptive_roi_weights: Dict[str, float] = {}
    for roi_name in roi_signals:
        base = float(roi_weights.get(roi_name, 1.0))
        roi_q = float(per_roi_signal_quality.get(roi_name, 0.0))
        illum = float(per_roi_illum_stab.get(roi_name, 0.0))
        oe = float(per_roi_overexposure.get(roi_name, 0.0))
        oe_pen = float(np.clip(1.0 - oe / _MAX_ROI_OVEREXPOSURE, 0.0, 1.0))
        adaptive_roi_weights[roi_name] = float(np.clip(base * (0.45 + 0.55 * roi_q) * (0.65 + 0.35 * illum) * (0.70 + 0.30 * oe_pen), 0.05, 3.0))

    # Normalise ROI weights for transparent debug and stable scaling.
    roi_w_sum = float(sum(adaptive_roi_weights.values()))
    if roi_w_sum > 1e-9:
        for k in list(adaptive_roi_weights.keys()):
            adaptive_roi_weights[k] = round(adaptive_roi_weights[k] / roi_w_sum, 4)

    for roi_name in roi_signals:
        green_sig = roi_signals[roi_name]
        trace_ts = None
        if roi_timestamps and roi_name in roi_timestamps and len(roi_timestamps[roi_name]) >= 8:
            trace_ts = np.array(roi_timestamps[roi_name], dtype=np.float64)
        rgb_arr = None
        if roi_rgb_traces and roi_name in roi_rgb_traces and len(roi_rgb_traces[roi_name]) >= 8:
            rgb_arr = np.array(roi_rgb_traces[roi_name], dtype=np.float64)
            if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3:
                rgb_arr = None

        for method in _EXTRACTION_METHODS:
            pulse, eff_fps, was_resampled = _build_candidate_signal(
                method,
                green_sig,
                rgb_arr,
                fps,
                low_hz,
                high_hz,
                timestamps=trace_ts,
            )
            if pulse is None:
                continue
            hr_est = spectral_hr_estimate(pulse, eff_fps, low_hz, high_hz)
            bpm_est = hr_est["bpm"]
            spec_q = hr_est["quality"]
            if bpm_est <= 0:
                candidate_rejections.append({
                    "roi": roi_name,
                    "method": method,
                    "reason": "invalid_bpm",
                })
                continue
            ss = compute_signal_strength(pulse, eff_fps, low_hz, high_hz)
            roi_q = per_roi_signal_quality.get(roi_name, 0.0)

            # Multi-window agreement per candidate.
            ts = compute_hr_timeseries(
                pulse.tolist(),
                eff_fps,
                window_sec=5.0,
                stride_sec=0.5,
                low_hz=low_hz,
                high_hz=high_hz,
                min_window_quality=0.06,
            )
            consensus = robust_hr_consensus(ts, periodicity=ss["periodicity"], min_windows=2)
            valid_windows = int(consensus.get("valid_window_count", 0))
            window_consistency = float(consensus.get("consensus_confidence", 0.0))
            window_std = float(consensus.get("std_dev", 999.0))
            has_consensus = bool(consensus.get("has_consensus", False))

            # Pre-estimation gate: reject weak candidates early.
            if (
                ss["signal_strength"] < _CANDIDATE_MIN_SIGNAL_STRENGTH
                or ss["periodicity"] < _CANDIDATE_MIN_PERIODICITY
            ):
                candidate_rejections.append({
                    "roi": roi_name,
                    "method": method,
                    "reason": "weak_signal_pre_gate",
                    "signal_strength": ss["signal_strength"],
                    "periodicity": ss["periodicity"],
                })
                continue
            if valid_windows < _CANDIDATE_MIN_VALID_WINDOWS:
                candidate_rejections.append({
                    "roi": roi_name,
                    "method": method,
                    "reason": "insufficient_windows",
                    "valid_windows": valid_windows,
                })
                continue
            if (not has_consensus) and window_std > _CANDIDATE_MAX_WINDOW_STD_NO_CONSENSUS:
                candidate_rejections.append({
                    "roi": roi_name,
                    "method": method,
                    "reason": "inconsistent_windows",
                    "std_dev": window_std,
                })
                continue

            # Part 3: Peak dominance validation — reduce confidence for weak or
            # isolated peaks so they can't win the competition unfairly.
            dominance_penalty = 1.0
            if hr_est["peak_prominence_raw"] < 0.20:
                dominance_penalty *= 0.70  # weak spectral peak
            if ss["peak_support_count"] <= 1:
                dominance_penalty *= 0.60  # isolated — no cross-window agreement
            spec_q = spec_q * dominance_penalty
            # Fair merit-based score: signal_strength is the primary
            # determinant (65%); ROI quality and illumination stability
            # add secondary weight; tiny ROI preference (max 5 points)
            # prevents stacking multiplicative biases.
            illum_stab = per_roi_illum_stab.get(roi_name, 1.0)
            roi_w = float(adaptive_roi_weights.get(roi_name, 0.33))

            # ML/rule quality gate uses measurable features only.
            features = extract_quality_features(
                pulse,
                eff_fps,
                low_hz,
                high_hz,
                motion_contamination=float(np.clip(1.0 - illum_stab, 0.0, 1.0)),
                roi_stability=illum_stab,
                brightness_quality=float(np.clip(roi_q, 0.0, 1.0)),
                overexposure_ratio=float(per_roi_overexposure.get(roi_name, 0.0)),
                window_consistency=window_consistency,
                valid_window_ratio=float(np.clip(valid_windows / 8.0, 0.0, 1.0)),
            )
            quality_eval = evaluate_signal_quality(features)
            quality_prob = float(quality_eval["good_signal_probability"])

            evidence_score = (
                0.65 * ss["signal_strength"]
                + 0.25 * roi_q
                + 0.05 * illum_stab
                + 0.05 * window_consistency
            )
            candidates.append({
                "roi": roi_name,
                "method": method,
                "bpm": round(bpm_est, 1),
                "spec_quality": round(spec_q, 4),
                "signal_strength": ss["signal_strength"],
                "snr": ss["snr"],
                "periodicity": ss["periodicity"],
                "peak_prominence": ss["peak_prominence"],
                "harmonic_consistency": ss["harmonic_consistency"],
                "inter_window_consistency": ss["inter_window_consistency"],
                # cross-window stability fields from compute_signal_strength
                "peak_stability": ss["peak_stability"],
                "modal_bpm": ss["modal_bpm"],
                "peak_support_count": ss["peak_support_count"],
                # harmonic correction / validation debug fields
                "harmonic_checked": hr_est["harmonic_checked"],
                "harmonic_corrected": hr_est["harmonic_corrected"],
                "original_bpm": hr_est["original_bpm"],
                "corrected_bpm": hr_est["corrected_bpm"],
                "peak_prominence_raw": hr_est["peak_prominence_raw"],
                "peak_snr": hr_est["peak_snr"],
                "physiologically_valid": hr_est["physiologically_valid"],
                "roi_quality": round(roi_q, 4),
                "window_consistency": round(window_consistency, 4),
                "window_std_dev": round(window_std, 3),
                "valid_windows": valid_windows,
                "has_window_consensus": has_consensus,
                "effective_fps": round(float(eff_fps), 3),
                "resampled": bool(was_resampled),
                "quality_probability": round(quality_prob, 4),
                "quality_source": quality_eval.get("source", "rule_fallback"),
                "quality_label": quality_eval.get("label", "bad_signal"),
                "quality_features": {k: round(float(v), 4) for k, v in features.items()},
                "evidence_score": round(float(np.clip(evidence_score, 0.0, 1.0)), 4),
                "roi_weight": round(roi_w, 4),
                "_pulse": pulse,  # internal, not serialised
            })

    # Adaptive method-family weighting from measurable candidate evidence.
    method_strength: Dict[str, float] = {m: 0.0 for m in _EXTRACTION_METHODS}
    for c in candidates:
        m = c["method"]
        q = float(c.get("quality_probability", 0.0))
        method_strength[m] = max(method_strength[m], float(c.get("evidence_score", 0.0)) * (0.55 + 0.45 * q))

    method_total = float(sum(method_strength.values()))
    if method_total > 1e-9:
        method_weights = {m: round(method_strength[m] / method_total, 4) for m in method_strength}
    else:
        method_weights = {m: 0.0 for m in method_strength}

    # Final candidate score combines evidence + ROI + method + quality probability.
    for c in candidates:
        roi_w = float(c.get("roi_weight", 0.33))
        method_w = float(method_weights.get(c["method"], 0.0))
        quality_prob = float(c.get("quality_probability", 0.0))
        base = float(c.get("evidence_score", 0.0))
        final_score = base * (0.65 + 0.35 * roi_w) * (0.70 + 0.30 * method_w) * (0.45 + 0.55 * quality_prob)
        c["candidate_score"] = round(float(np.clip(final_score, 0.0, 1.0)), 4)
        c["method_weight"] = round(method_w, 4)

    # ── Phase 3: select best candidate ───────────────────────────
    per_roi_bpm: Dict[str, float] = {}
    if candidates:
        candidates.sort(key=lambda c: c["candidate_score"], reverse=True)
        best = candidates[0]
        selected_roi = best["roi"]
        selected_method = best["method"]
        best_bpm = best["bpm"]
        best_quality = best["candidate_score"]
        best_signal_strength = best["signal_strength"]
        best_periodicity = best["periodicity"]
        pos_used = best["method"] in ("pos", "chrom", "pca")

        # --- Additional harmonic correction via cross-window modal BPM ---
        # If compute_signal_strength found that windows consistently vote for
        # ~2× the spectral BPM, use the modal BPM as the harmonic-corrected value.
        modal = best.get("modal_bpm", 0.0)
        support = best.get("peak_support_count", 0)
        _har_corrected = best.get("harmonic_corrected", False)
        if (
            not _har_corrected          # not already corrected in spectral analysis
            and modal > 0
            and best_bpm > 0
            and abs(modal - 2.0 * best_bpm) <= 10.0  # modal ≈ 2× spectral BPM
            and support >= 2                           # enough windows support it
            and 45.0 <= modal <= 180.0
        ):
            best_bpm = modal
            best["harmonic_corrected"] = True
            best["corrected_bpm"] = round(modal, 1)
            best["original_bpm"] = best.get("original_bpm", best_bpm)

        # Fill per_roi_bpm from best candidate per ROI
        roi_best: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            rn = c["roi"]
            if rn not in roi_best or c["candidate_score"] > roi_best[rn]["candidate_score"]:
                roi_best[rn] = c
        for rn, cb in roi_best.items():
            per_roi_bpm[rn] = cb["bpm"]
    else:
        # Fallback to legacy green-only fusion
        for name, sig in roi_signals.items():
            bpm_g, _ = estimate_bpm(sig, fps, low_hz, high_hz)
            per_roi_bpm[name] = bpm_g
        adjusted_weights = {}
        for name in roi_signals:
            base_w = roi_weights.get(name, 1.0)
            sq = per_roi_signal_quality.get(name, 0.0)
            adjusted_weights[name] = base_w * (0.3 + 0.7 * sq)
        fused_bpm, fused_quality, _ = fuse_roi_signals(
            roi_signals, adjusted_weights, fps, low_hz, high_hz
        )
        return {
            "bpm": fused_bpm if fused_bpm > 0 else None,
            "quality": round(fused_quality, 4),
            "per_roi_quality": per_roi_signal_quality,
            "per_roi_bpm": {k: round(v, 1) for k, v in per_roi_bpm.items()},
            "roi_agreement": 0.5, "periodicity": 0.0,
            "pos_used": False, "rois_dropped": rois_dropped,
            "selected_roi": None, "selected_method": "green_fusion_fallback",
            "signal_strength": 0.0, "all_candidate_scores": [],
            "roi_scores": per_roi_signal_quality,
            "roi_color_variance": per_roi_color_var,
            "roi_illumination_stability": per_roi_illum_stab,
            "roi_adaptive_weights": adaptive_roi_weights,
            "method_scores": {},
            "method_adaptive_weights": {m: 0.0 for m in _EXTRACTION_METHODS},
            "candidate_rejections": candidate_rejections,
            # Part 5 debug fields (empty for fallback path)
            "harmonic_checked": False,
            "harmonic_corrected": False,
            "original_hr": fused_bpm if fused_bpm > 0 else None,
            "corrected_hr": None,
            "peak_prominence": 0.0,
            "peak_support_count": 0,
        }

    # ── Phase 4: weighted fusion of top candidates (fallback BPM) ──
    # Use top candidates that agree with the best (within ±8 bpm)
    agreeing = [c for c in candidates if abs(c["bpm"] - best_bpm) <= 8.0]
    if len(agreeing) >= 2:
        w_sum = sum(c["candidate_score"] for c in agreeing)
        if w_sum > 0:
            fused_bpm = sum(c["bpm"] * c["candidate_score"] for c in agreeing) / w_sum
        else:
            fused_bpm = best_bpm
    else:
        fused_bpm = best_bpm

    # ── Phase 5: ROI agreement ───────────────────────────────────
    valid_bpms = [b for b in per_roi_bpm.values() if b > 0]
    if len(valid_bpms) >= 2:
        std_bpm = float(np.std(valid_bpms))
        roi_agreement = float(np.clip(1.0 - std_bpm / 20.0, 0, 1))
    else:
        roi_agreement = 0.5

    # ── Phase 6: method-level scores ─────────────────────────────
    method_scores: Dict[str, float] = {}
    for m in _EXTRACTION_METHODS:
        method_cands = [c for c in candidates if c["method"] == m]
        if method_cands:
            method_scores[m] = round(max(c["candidate_score"] for c in method_cands), 4)
        else:
            method_scores[m] = 0.0

    # Sanitise candidate list for JSON serialisation (remove numpy arrays)
    serialisable = []
    for c in candidates:
        sc = {k: v for k, v in c.items() if k != "_pulse"}
        serialisable.append(sc)

    eff_fps_values = [float(c.get("effective_fps", fps)) for c in serialisable]
    resampled_count = int(sum(1 for c in serialisable if bool(c.get("resampled", False))))
    effective_fps_summary = {
        "min": round(float(np.min(eff_fps_values)), 3) if eff_fps_values else round(float(fps), 3),
        "median": round(float(np.median(eff_fps_values)), 3) if eff_fps_values else round(float(fps), 3),
        "max": round(float(np.max(eff_fps_values)), 3) if eff_fps_values else round(float(fps), 3),
    }

    return {
        "bpm": round(fused_bpm, 1) if fused_bpm > 0 else None,
        "quality": round(best_quality, 4),
        "per_roi_quality": per_roi_signal_quality,
        "per_roi_bpm": {k: round(v, 1) for k, v in per_roi_bpm.items()},
        "roi_agreement": round(roi_agreement, 3),
        "periodicity": round(best_periodicity, 3),
        "pos_used": pos_used,
        "rois_dropped": rois_dropped,
        # Competition fields
        "selected_roi": selected_roi,
        "selected_method": selected_method,
        "signal_strength": round(best_signal_strength, 4),
        "all_candidate_scores": serialisable,
        "roi_scores": per_roi_signal_quality,
        "roi_color_variance": per_roi_color_var,
        "roi_illumination_stability": per_roi_illum_stab,
        "roi_adaptive_weights": adaptive_roi_weights,
        "method_scores": method_scores,
        "method_adaptive_weights": method_weights,
        "candidate_rejections": candidate_rejections,
        "resampled_candidate_count": resampled_count,
        "resampled_candidate_ratio": round(float(resampled_count / max(len(serialisable), 1)), 4),
        "effective_fps_summary": effective_fps_summary,
        # Part 5: harmonic correction + peak debug fields from best candidate
        "harmonic_checked": best.get("harmonic_checked", False),
        "harmonic_corrected": best.get("harmonic_corrected", False),
        "original_hr": best.get("original_bpm", best_bpm),
        "corrected_hr": best.get("corrected_bpm"),
        "peak_prominence": best.get("peak_prominence", 0.0),
        "peak_support_count": best.get("peak_support_count", 0),
    }
