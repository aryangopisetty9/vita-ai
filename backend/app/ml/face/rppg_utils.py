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
    mean_green_from_roi,
    mean_rgb_from_roi,
    build_skin_mask,
    overexposure_ratio,
    compute_roi_quality_metrics,
)
from backend.app.utils.signal_processing import (
    chrom_project,
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
)


# ROI names used throughout the pipeline
ROI_NAMES = ("forehead", "left_cheek", "right_cheek")

# Minimum per-ROI quality to include in fusion (below = dropped)
_MIN_ROI_QUALITY = 0.08

# Maximum allowed overexposure ratio on an ROI to accept it
_MAX_ROI_OVEREXPOSURE = 0.40


def _apply_skin_mask(roi_crop: np.ndarray, mask_crop: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Combine an optional polygon mask with a skin-colour mask.

    Returns the combined mask or None if no skin pixels survive.
    """
    skin = build_skin_mask(roi_crop)
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
        signals["forehead"] = mean_green_from_roi(fh_roi, skin_m)
    else:
        signals["forehead"] = None

    # Left cheek
    lc = extract_cheek_roi(frame_bgr, lm_list, LEFT_CHEEK, h, w)
    if lc is not None:
        roi_crop, mask_crop = lc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["left_cheek"] = mean_green_from_roi(roi_crop, combined)
    else:
        signals["left_cheek"] = None

    # Right cheek
    rc = extract_cheek_roi(frame_bgr, lm_list, RIGHT_CHEEK, h, w)
    if rc is not None:
        roi_crop, mask_crop = rc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["right_cheek"] = mean_green_from_roi(roi_crop, combined)
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
        signals["forehead"] = mean_rgb_from_roi(fh_roi, skin_m)
    else:
        signals["forehead"] = None

    # Left cheek
    lc = extract_cheek_roi(frame_bgr, lm_list, LEFT_CHEEK, h, w)
    if lc is not None:
        roi_crop, mask_crop = lc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["left_cheek"] = mean_rgb_from_roi(roi_crop, combined)
    else:
        signals["left_cheek"] = None

    # Right cheek
    rc = extract_cheek_roi(frame_bgr, lm_list, RIGHT_CHEEK, h, w)
    if rc is not None:
        roi_crop, mask_crop = rc
        combined = _apply_skin_mask(roi_crop, mask_crop)
        signals["right_cheek"] = mean_rgb_from_roi(roi_crop, combined)
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


def _build_candidate_signal(
    method: str,
    green_signal: np.ndarray,
    rgb_arr: Optional[np.ndarray],
    fps: float,
) -> Optional[np.ndarray]:
    """Build a pulse signal for a given extraction method.

    Methods: 'green', 'pos', 'chrom', 'pca'.

    Each method applies only the preprocessing it requires; functions
    that already normalise internally (POS, CHROM, PCA/LGI) receive
    only the remaining steps to avoid double-normalization.
    """
    if method == "green":
        sig = green_signal.copy()
        # Green needs full preprocessing: MA → detrend → temporal normalize → bandpass
        ma_window = max(3, int(fps * 0.12))
        sig = moving_average_smooth(sig.astype(np.float64), window=ma_window)
        sig = detrend_signal(sig)
        sig = temporal_normalization(sig, window=max(int(fps * 2), 8))
        sig = bandpass_fft(sig, fps, 0.7, 3.5)
    elif method == "pos" and rgb_arr is not None and len(rgb_arr) >= 8:
        sig = pos_project(rgb_arr, fps)
        # POS applies per-window temporal normalization internally (overlap-add);
        # only detrend + bandpass here to avoid double normalization.
        if len(sig) < 8:
            return None
        sig = detrend_signal(sig.astype(np.float64))
        sig = bandpass_fft(sig, fps, 0.7, 3.5)
    elif method == "chrom" and rgb_arr is not None and len(rgb_arr) >= 8:
        sig = chrom_project(rgb_arr, fps)
        # CHROM applies temporal normalization + detrend internally;
        # only bandpass here to avoid triple normalization.
        if len(sig) < 8:
            return None
        sig = bandpass_fft(sig.astype(np.float64), fps, 0.7, 3.5)
    elif method == "pca" and rgb_arr is not None and len(rgb_arr) >= 8:
        sig = lgi_project(rgb_arr, fps)
        # PCA/LGI applies full preprocessing internally (detrend + temporal norm + bandpass).
        if len(sig) < 8:
            return None
    else:
        return None
    if len(sig) < 8:
        return None
    return sig


_EXTRACTION_METHODS = ("green", "pos", "chrom", "pca")


def estimate_heart_rate_multi_roi(
    roi_traces: Dict[str, List[float]],
    roi_weights: Dict[str, float],
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    roi_rgb_traces: Optional[Dict[str, List[np.ndarray]]] = None,
    roi_overexposure: Optional[Dict[str, List[float]]] = None,
    roi_skin_coverage: Optional[Dict[str, float]] = None,
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
    for roi_name in roi_signals:
        green_sig = roi_signals[roi_name]
        rgb_arr = None
        if roi_rgb_traces and roi_name in roi_rgb_traces and len(roi_rgb_traces[roi_name]) >= 8:
            rgb_arr = np.array(roi_rgb_traces[roi_name], dtype=np.float64)
            if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3:
                rgb_arr = None

        for method in _EXTRACTION_METHODS:
            pulse = _build_candidate_signal(method, green_sig, rgb_arr, fps)
            if pulse is None:
                continue
            hr_est = spectral_hr_estimate(pulse, fps, low_hz, high_hz)
            bpm_est = hr_est["bpm"]
            spec_q = hr_est["quality"]
            if bpm_est <= 0:
                continue
            ss = compute_signal_strength(pulse, fps, low_hz, high_hz)
            roi_q = per_roi_signal_quality.get(roi_name, 0.0)
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
            base_w = roi_weights.get(roi_name, 1.0)  # 1.5 for forehead, 1.0 for cheeks
            max_w = max(roi_weights.values()) if roi_weights else 1.5
            roi_pref = float(np.clip((base_w - 1.0) / max(max_w - 1.0, 1e-6) * 0.05, 0.0, 0.05))
            candidate_score = (
                0.65 * ss["signal_strength"]
                + 0.25 * roi_q
                + 0.05 * illum_stab
                + roi_pref
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
                "candidate_score": round(float(np.clip(candidate_score, 0.0, 1.0)), 4),
                "_pulse": pulse,  # internal, not serialised
            })

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
            "method_scores": {},
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
        "method_scores": method_scores,
        # Part 5: harmonic correction + peak debug fields from best candidate
        "harmonic_checked": best.get("harmonic_checked", False),
        "harmonic_corrected": best.get("harmonic_corrected", False),
        "original_hr": best.get("original_bpm", best_bpm),
        "corrected_hr": best.get("corrected_bpm"),
        "peak_prominence": best.get("peak_prominence", 0.0),
        "peak_support_count": best.get("peak_support_count", 0),
    }
