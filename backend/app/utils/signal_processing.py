"""
Vita AI – Signal Processing Utilities
======================================
Reusable, modular signal-processing helpers for rPPG pulse estimation
and related temporal analysis.

All functions are pure-NumPy / SciPy-free so the dependency footprint
stays minimal.

Extension points
----------------
* Swap ``bandpass_fft`` for a Butterworth IIR filter (``scipy.signal``)
  for real-time streaming where FFT-on-full-window is impractical.
* Plug in a learned rPPG model (Open-rPPG) by
  replacing ``estimate_bpm_from_signal`` with a model-inference call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Band-pass filtering
# ═══════════════════════════════════════════════════════════════════════════

def bandpass_fft(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> np.ndarray:
    """Zero-phase FFT band-pass filter.

    Parameters
    ----------
    signal : 1-D float array
    fps : sampling rate (frames per second)
    low_hz, high_hz : pass-band edges in Hz

    Returns
    -------
    Filtered signal (same length).
    """
    n = len(signal)
    if n < 4:
        return signal.copy()
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_vals = np.fft.rfft(signal)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    fft_vals[~mask] = 0.0
    return np.fft.irfft(fft_vals, n=n)


# ═══════════════════════════════════════════════════════════════════════════
# Detrending
# ═══════════════════════════════════════════════════════════════════════════

def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend from a 1-D signal (least-squares fit)."""
    n = len(signal)
    if n < 2:
        return signal.copy()
    x = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(x, signal, 1)
    trend = np.polyval(coeffs, x)
    return signal - trend


def temporal_normalization(signal: np.ndarray, window: int = 30) -> np.ndarray:
    """Sliding-window z-score normalisation to reduce illumination drift.

    Each sample is normalised by ``(x - mu) / sigma`` computed from a
    local window centred on that sample.
    """
    n = len(signal)
    if n < 4:
        return signal.copy()
    half = max(window // 2, 1)
    out = np.zeros_like(signal, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = signal[lo:hi]
        mu = np.mean(seg)
        sigma = np.std(seg)
        if sigma < 1e-8:
            out[i] = 0.0
        else:
            out[i] = (signal[i] - mu) / sigma
    return out


def moving_average_smooth(
    signal: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Apply a causal moving-average to reduce high-frequency noise.

    Uses reflection padding at the edges to prevent boundary artifacts.
    A small window (3-7 samples) is recommended to leave the physiological
    pulse band intact while removing frame-to-frame jitter.

    Parameters
    ----------
    signal : 1-D float array
    window : number of samples to average (odd recommended)

    Returns
    -------
    Smoothed signal of the same length.
    """
    n = len(signal)
    if n < 4 or window < 2:
        return signal.copy()
    half = window // 2
    # Reflection-pad both ends to avoid boundary shrinkage
    padded = np.pad(signal.astype(np.float64), half, mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    # Trim to original length (convolve with 'valid' on padded input)
    return smoothed[:n]


def chrom_project(rgb_traces: np.ndarray, fps: float) -> np.ndarray:
    """CHROM rPPG color projection (de Haan and Jeanne style variant).

    Converts RGB traces into a pulse signal that is often more stable than
    green-only under illumination drift.
    """
    n = rgb_traces.shape[0]
    if n < 8:
        return np.zeros(n, dtype=np.float64)

    rgb = rgb_traces.astype(np.float64).copy()
    # Temporal normalization per channel
    rgb = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)

    # Chrominance components
    x = 3.0 * rgb[:, 0] - 2.0 * rgb[:, 1]
    y = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]

    std_y = float(np.std(y))
    alpha = float(np.std(x) / std_y) if std_y > 1e-8 else 1.0
    s = x - alpha * y

    s = detrend_signal(s)
    s = temporal_normalization(s, window=max(int(fps * 1.6), 8))
    return s


def lgi_project(rgb_traces: np.ndarray, fps: float) -> np.ndarray:
    """LGI / PCA-based rPPG colour projection (4th extraction method).

    Projects channel-normalized RGB traces onto their second principal
    component.  The first component captures residual illumination (DC
    shift); the second typically isolates the cardiac pulse signal.
    This is conceptually similar to Local Group Invariance (Pilz et al.
    2018) but operates on whole-ROI temporal averages rather than spatial
    pixel groups, making it computationally cheap as a 4th candidate
    alongside green/POS/CHROM.

    Parameters
    ----------
    rgb_traces : (N, 3) float64 array — per-frame [R, G, B] means.
    fps : frames per second.

    Returns
    -------
    1-D pulse signal of length N (detrended, temporally normalized,
    and bandpass-filtered — preprocessing applied internally so callers
    should NOT double-normalize).
    """
    n = rgb_traces.shape[0]
    if n < 8:
        return np.zeros(n, dtype=np.float64)

    rgb = rgb_traces.astype(np.float64)

    # Channel-wise temporal normalization to remove illumination bias
    means = np.mean(rgb, axis=0, keepdims=True)
    means[means < 1e-6] = 1.0
    rgb_norm = rgb / means

    # Centre the normalized traces
    centered = rgb_norm - np.mean(rgb_norm, axis=0, keepdims=True)

    # PCA via 3×3 covariance matrix SVD — cheap and numerically stable
    try:
        cov = (centered.T @ centered) / max(n - 1, 1)
        _, _, Vt = np.linalg.svd(cov)
        # Project onto 2nd principal component (index 1)
        pulse = centered @ Vt[1]
    except np.linalg.LinAlgError:
        # SVD failed — fall back to G−R difference (often sufficient)
        pulse = centered[:, 1] - centered[:, 0]

    pulse = detrend_signal(pulse)
    pulse = temporal_normalization(pulse, window=max(int(fps * 2), 8))
    pulse = bandpass_fft(pulse, fps, 0.7, 3.5)
    return pulse


# ═══════════════════════════════════════════════════════════════════════════
# BPM estimation
# ═══════════════════════════════════════════════════════════════════════════

def spectral_hr_estimate(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> Dict[str, Any]:
    """Full spectral HR estimation with harmonic correction and physiological validation.

    Extends the basic pipeline with:

    * **Harmonic correction** — when the dominant spectral peak maps to a
      BPM < 50 (a common sub-harmonic mis-selection), checks whether the
      2× harmonic peak has *both* higher spectral prominence *and* higher
      SNR.  If so, the BPM is replaced with the harmonic value.
    * **Physiological validation** — BPM < 45 is discarded and both
      ``bpm`` and ``quality`` are returned as ``0.0``.

    Parameters
    ----------
    signal : array-like green/pulse trace (1-D float64).
    fps    : camera frame rate in Hz.
    low_hz, high_hz : bandpass bounds for the valid HR frequency band.

    Returns
    -------
    Dict with keys:
      ``bpm``, ``quality``, ``original_bpm``,
      ``harmonic_checked``, ``harmonic_corrected``, ``corrected_bpm``,
      ``peak_prominence_raw``, ``peak_snr``, ``physiologically_valid``.
    """
    _empty: Dict[str, Any] = dict(
        bpm=0.0, quality=0.0, original_bpm=0.0,
        harmonic_checked=False, harmonic_corrected=False, corrected_bpm=None,
        peak_prominence_raw=0.0, peak_snr=0.0, physiologically_valid=False,
    )
    if len(signal) < 8:
        return _empty

    # --- Preprocessing (identical to estimate_bpm) ---
    sig = detrend_signal(signal.astype(np.float64))
    sig = temporal_normalization(sig, window=max(int(fps * 2), 8))
    ma_window = max(3, int(fps * 0.12))
    sig = moving_average_smooth(sig, window=ma_window)
    filtered = bandpass_fft(sig, fps, low_hz, high_hz)

    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    magnitude = np.abs(np.fft.rfft(filtered))
    valid = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(valid):
        return _empty

    valid_freqs = freqs[valid]
    valid_mag = magnitude[valid]

    mean_mag = float(np.mean(valid_mag))
    if mean_mag < 1e-12:
        return _empty

    # --- Dominant peak ---
    peak_idx = int(np.argmax(valid_mag))
    dominant_freq = float(valid_freqs[peak_idx])
    original_bpm = dominant_freq * 60.0   # BPM *before* any correction
    peak_val = float(valid_mag[peak_idx])
    peak_snr = peak_val / mean_mag

    def _local_prominence(idx: int) -> float:
        """(peak − base) / peak — purely spectral, 0–1."""
        pv = float(valid_mag[idx])
        if pv < 1e-12:
            return 0.0
        lmin = pv
        for i in range(idx - 1, -1, -1):
            if valid_mag[i] <= lmin:
                lmin = float(valid_mag[i])
            else:
                break
        rmin = pv
        for i in range(idx + 1, len(valid_mag)):
            if valid_mag[i] <= rmin:
                rmin = float(valid_mag[i])
            else:
                break
        return float(np.clip((pv - max(lmin, rmin)) / pv, 0.0, 1.0))

    raw_prominence = _local_prominence(peak_idx)

    # --- Part 1: Harmonic correction ---
    # Triggered when the dominant BPM is suspiciously low (< 50 bpm), which
    # is characteristic of the FFT selecting the sub-harmonic instead of the
    # true pulse frequency.  We prefer the 2× harmonic only when it has
    # *both* higher prominence AND higher SNR than the current dominant peak.
    harmonic_checked = False
    harmonic_corrected = False
    corrected_bpm: Optional[float] = None

    if original_bpm < 50.0:
        harmonic_freq = dominant_freq * 2.0
        harmonic_bpm_cand = harmonic_freq * 60.0
        if low_hz <= harmonic_freq <= high_hz and 45.0 <= harmonic_bpm_cand <= 180.0:
            harmonic_checked = True
            h_idx = int(np.argmin(np.abs(valid_freqs - harmonic_freq)))
            h_mag = float(valid_mag[h_idx])
            h_snr = h_mag / mean_mag
            h_prominence = _local_prominence(h_idx)
            # Require BOTH higher prominence AND higher SNR — avoids false corrections
            if h_snr > peak_snr and h_prominence > raw_prominence and h_mag >= 0.35 * peak_val:
                # Commit to harmonic
                dominant_freq = harmonic_freq
                corrected_bpm = round(harmonic_bpm_cand, 1)
                harmonic_corrected = True
                peak_idx = h_idx
                peak_val = h_mag
                peak_snr = h_snr
                raw_prominence = h_prominence

    # --- Subharmonic correction (detect 2nd harmonic mislabeled as fundamental) ---
    # When the dominant BPM is suspiciously high (> 90 bpm) the FFT may have
    # locked onto the 2nd harmonic of the true cardiac signal.  Check if BPM/2
    # is a valid resting rate with strong spectral presence.  Only correct when
    # the sub-harmonic has clear prominence AND at least 40% of the dominant
    # magnitude — a weak sub-harmonic is just noise, not the fundamental.
    current_bpm_before_sub = dominant_freq * 60.0
    if not harmonic_corrected and current_bpm_before_sub > 90.0:
        sub_freq = dominant_freq / 2.0
        sub_bpm_cand = sub_freq * 60.0
        if low_hz <= sub_freq <= high_hz and 45.0 <= sub_bpm_cand <= 90.0:
            harmonic_checked = True
            s_idx = int(np.argmin(np.abs(valid_freqs - sub_freq)))
            s_mag = float(valid_mag[s_idx])
            s_snr = s_mag / mean_mag
            s_prominence = _local_prominence(s_idx)
            if s_prominence > 0.25 and s_mag >= 0.40 * peak_val and s_snr > 2.5:
                dominant_freq = sub_freq
                corrected_bpm = round(sub_bpm_cand, 1)
                harmonic_corrected = True
                peak_idx = s_idx
                peak_val = s_mag
                peak_snr = s_snr
                raw_prominence = s_prominence

    bpm = dominant_freq * 60.0

    # --- Part 2: Physiological validation (bounds: 45–180 bpm) ---
    if bpm < 45.0 or bpm > 180.0:
        return dict(
            bpm=0.0, quality=0.0,
            original_bpm=round(original_bpm, 1),
            harmonic_checked=harmonic_checked,
            harmonic_corrected=harmonic_corrected,
            corrected_bpm=corrected_bpm,
            peak_prominence_raw=round(float(np.clip(raw_prominence * 1.5, 0.0, 1.0)), 4),
            peak_snr=round(peak_snr, 4),
            physiologically_valid=False,
        )

    quality = float(np.clip(peak_snr / 10.0, 0.0, 1.0))
    if bpm < 50.0:
        quality *= 0.8  # borderline: slight confidence reduction

    return dict(
        bpm=round(bpm, 1),
        quality=round(quality, 4),
        original_bpm=round(original_bpm, 1),
        harmonic_checked=harmonic_checked,
        harmonic_corrected=harmonic_corrected,
        corrected_bpm=corrected_bpm,
        peak_prominence_raw=round(float(np.clip(raw_prominence * 1.5, 0.0, 1.0)), 4),
        peak_snr=round(peak_snr, 4),
        physiologically_valid=True,
    )


def estimate_bpm(
    green_signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> Tuple[float, float]:
    """Estimate heart rate (BPM) from a green-channel temporal trace.

    Thin wrapper around :func:`spectral_hr_estimate` that preserves the
    ``(bpm, quality)`` return signature for backward compatibility.
    Harmonic correction and physiological validation are applied
    transparently — all callers benefit automatically.

    Returns
    -------
    (bpm, quality)  where quality ∈ [0, 1].
    """
    result = spectral_hr_estimate(green_signal, fps, low_hz, high_hz)
    return result["bpm"], result["quality"]


def compute_periodicity(signal: np.ndarray, fps: float, low_hz: float = 0.7, high_hz: float = 3.5) -> float:
    """Return a 0-1 periodicity score (spectral concentration near peak).

    A strongly periodic pulse signal has most energy at one frequency.
    """
    if len(signal) < 8:
        return 0.0
    filtered = bandpass_fft(signal, fps, low_hz, high_hz)
    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mag = np.abs(np.fft.rfft(filtered))
    valid = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(valid):
        return 0.0
    vm = mag[valid]
    total = float(np.sum(vm))
    if total < 1e-12:
        return 0.0
    peak = float(np.max(vm))
    return float(np.clip(peak / total * 3.0, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Multi-ROI fusion
# ═══════════════════════════════════════════════════════════════════════════

def fuse_roi_signals(
    roi_signals: Dict[str, np.ndarray],
    roi_weights: Dict[str, float],
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> Tuple[float, float, Dict[str, float]]:
    """Fuse pulse signals from multiple ROIs into one BPM estimate.

    Parameters
    ----------
    roi_signals : {roi_name: 1-D green-channel array}
    roi_weights : {roi_name: weight ∈ [0, 1]}
        Weights may come from face quality / lighting / motion scores.
    fps : frames per second

    Returns
    -------
    (fused_bpm, fused_quality, per_roi_quality)
    """
    bpms: List[float] = []
    qualities: List[float] = []
    weights: List[float] = []
    per_roi_q: Dict[str, float] = {}

    for name, sig in roi_signals.items():
        if len(sig) < 8:
            per_roi_q[name] = 0.0
            continue
        bpm, q = estimate_bpm(sig, fps, low_hz, high_hz)
        per_roi_q[name] = round(q, 4)
        if bpm <= 0:
            continue
        w = roi_weights.get(name, 1.0) * q  # weight × quality
        bpms.append(bpm)
        qualities.append(q)
        weights.append(w)

    if not bpms:
        return 0.0, 0.0, per_roi_q

    w_arr = np.array(weights)
    w_sum = float(np.sum(w_arr))
    if w_sum < 1e-12:
        return 0.0, 0.0, per_roi_q

    w_norm = w_arr / w_sum
    fused_bpm = float(np.dot(bpms, w_norm))
    fused_quality = float(np.dot(qualities, w_norm))

    return round(fused_bpm, 1), round(fused_quality, 4), per_roi_q


# ═══════════════════════════════════════════════════════════════════════════
# Sliding-window HR timeseries
# ═══════════════════════════════════════════════════════════════════════════

def compute_hr_timeseries(
    green_values: List[float],
    fps: float,
    window_sec: float = 5.0,
    stride_sec: float = 0.5,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    min_window_quality: float = 0.10,
) -> List[Dict[str, Any]]:
    """Sliding-window BPM estimates for heart-rate graphing.

    Each entry: ``{"timestamp_sec": ..., "bpm": ..., "quality": ...,
    "snr": ..., "amplitude": ...}``

    Shorter default stride (0.5 s) produces more windows for stronger
    consensus.  Per-window SNR and amplitude are included so the
    consensus step can weight better windows more.

    Parameters
    ----------
    min_window_quality : float
        Windows with quality below this threshold are suppressed so
        noisy estimates don't pollute the timeseries.
    """
    signal = np.array(green_values, dtype=np.float64)
    n = len(signal)
    window_frames = int(window_sec * fps)
    stride_frames = max(int(stride_sec * fps), 1)

    if n < window_frames:
        hr_est = spectral_hr_estimate(signal, fps, low_hz, high_hz)
        bpm = hr_est["bpm"]
        # Richer quality: blend spectral SNR quality + peak prominence + SNR proxy
        q = float(np.clip(
            0.50 * hr_est["quality"]
            + 0.30 * hr_est["peak_prominence_raw"]
            + 0.20 * min(1.0, hr_est["peak_snr"] / 8.0),
            0.0, 1.0,
        ))
        if bpm > 0 and q >= min_window_quality:
            snr = compute_snr(signal, fps, low_hz, high_hz)
            amp = signal_amplitude(signal)
            return [{"timestamp_sec": round(n / (2 * fps), 2),
                     "bpm": round(bpm, 1), "quality": round(q, 3),
                     "snr": round(snr, 4), "amplitude": round(amp, 4)}]
        return []

    timeseries: List[Dict[str, Any]] = []
    start = 0
    while start + window_frames <= n:
        segment = signal[start:start + window_frames]
        hr_est = spectral_hr_estimate(segment, fps, low_hz, high_hz)
        bpm = hr_est["bpm"]
        # Richer quality: blend spectral SNR quality + peak prominence + SNR proxy
        q = float(np.clip(
            0.50 * hr_est["quality"]
            + 0.30 * hr_est["peak_prominence_raw"]
            + 0.20 * min(1.0, hr_est["peak_snr"] / 8.0),
            0.0, 1.0,
        ))
        mid_time = (start + window_frames / 2) / fps
        if bpm > 0 and q >= min_window_quality:
            snr = compute_snr(segment, fps, low_hz, high_hz)
            amp = signal_amplitude(segment)
            timeseries.append({
                "timestamp_sec": round(mid_time, 2),
                "bpm": round(bpm, 1),
                "quality": round(q, 3),
                "snr": round(snr, 4),
                "amplitude": round(amp, 4),
            })
        start += stride_frames

    return timeseries


# ═══════════════════════════════════════════════════════════════════════════
# Temporal smoothing
# ═══════════════════════════════════════════════════════════════════════════

def median_filter_bpm(
    timeseries: List[Dict[str, Any]],
    kernel: int = 3,
) -> List[Dict[str, Any]]:
    """Apply a sliding median filter to BPM values in an HR timeseries.

    Reduces transient spikes from noise or missed beats while preserving
    real changes in heart rate.

    Parameters
    ----------
    timeseries : list of dict
        Each entry has ``"bpm"``, ``"quality"``, ``"timestamp_sec"``.
    kernel : int
        Window size for the median filter (must be odd ≥ 1).

    Returns
    -------
    Smoothed copy of the timeseries (originals are not mutated).
    """
    n = len(timeseries)
    if n < 3 or kernel < 2:
        return timeseries

    kernel = kernel if kernel % 2 == 1 else kernel + 1
    half = kernel // 2
    bpms = [e["bpm"] for e in timeseries]

    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        med_bpm = float(np.median(bpms[lo:hi]))
        entry = dict(timeseries[i])
        entry["bpm"] = round(med_bpm, 1)
        smoothed.append(entry)
    return smoothed


def robust_hr_consensus(
    timeseries: List[Dict[str, Any]],
    periodicity: float = 0.0,
    min_windows: int = 2,
) -> Dict[str, Any]:
    """Derive a stable HR from multiple windows with outlier removal.

    Steps
    -----
    1. Collect per-window BPM values.
    2. Remove outliers using MAD and then IQR filters.
    3. Compute final HR from median or trimmed mean.
    4. Return stability stats for confidence shaping.
    """
    def _empty(method: str = "none", rejected_windows: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        return {
            "has_consensus": False,
            "heart_rate": None,
            "median_hr": None,
            "hr_values": [],
            "filtered_hr_values": [],
            "removed_outliers": [],
            "selected_windows": [],
            "rejected_windows": rejected_windows or [],
            "std_dev": 0.0,
            "variance": 0.0,
            "valid_window_count": 0,
            "window_count_score": 0.0,
            "snr_proxy": 0.0,
            "peak_quality_score": 0.0,
            "stability_score": 0.0,
            "unstable": True,
            "periodicity_score": round(float(np.clip(periodicity, 0.0, 1.0)), 3),
            "method": method,
        }

    if not timeseries:
        return _empty()

    # Build per-window records first so we can report selected/rejected windows.
    raw_windows: List[Dict[str, Any]] = []
    for idx, e in enumerate(timeseries):
        bpm = e.get("bpm")
        if bpm is None:
            continue
        raw_windows.append(
            {
                "index": idx,
                "timestamp_sec": e.get("timestamp_sec"),
                "bpm": float(bpm),
                "quality": float(e.get("quality", 0.0)),
                "snr": float(e.get("snr", 0.0)),
                "amplitude": float(e.get("amplitude", 0.0)),
                "reason": "",
            }
        )
    if not raw_windows:
        return _empty(method="no_window_bpm")

    rejected_windows: List[Dict[str, Any]] = []

    # Range filtering: hard bounds first.
    in_hard_range: List[Dict[str, Any]] = []
    for w in raw_windows:
        if 40.0 <= w["bpm"] <= 180.0:
            in_hard_range.append(w)
        else:
            bad = dict(w)
            bad["reason"] = "outside_hard_range"
            rejected_windows.append(bad)

    if not in_hard_range:
        return _empty(method="hard_range_reject", rejected_windows=rejected_windows)

    # Preferred resting range: if enough windows are present, prioritize them.
    preferred = [w for w in in_hard_range if 50.0 <= w["bpm"] <= 100.0]
    candidate_windows = preferred if len(preferred) >= min_windows else in_hard_range
    if candidate_windows is preferred:
        for w in in_hard_range:
            if not (50.0 <= w["bpm"] <= 100.0):
                bad = dict(w)
                bad["reason"] = "outside_preferred_range"
                rejected_windows.append(bad)

    # Peak-quality validation proxy from per-window spectrum quality.
    quality_floor = 0.08
    by_quality: List[Dict[str, Any]] = []
    for w in candidate_windows:
        if w["quality"] >= quality_floor:
            by_quality.append(w)
        else:
            bad = dict(w)
            bad["reason"] = "flat_or_ambiguous_peak"
            rejected_windows.append(bad)
    candidate_windows = by_quality

    if len(candidate_windows) < min_windows:
        return _empty(method="insufficient_quality_windows", rejected_windows=rejected_windows)

    arr = np.array([w["bpm"] for w in candidate_windows], dtype=np.float64)
    filtered = arr.copy()

    # 1) MAD outlier rejection.
    if len(filtered) >= 3:
        med = float(np.median(filtered))
        mad = float(np.median(np.abs(filtered - med)))
        if mad > 1e-6:
            mod_z = 0.6745 * (filtered - med) / mad
            filtered = filtered[np.abs(mod_z) <= 3.5]

    # 2) IQR outlier rejection.
    if len(filtered) >= 4:
        q1 = float(np.percentile(filtered, 25))
        q3 = float(np.percentile(filtered, 75))
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        filtered = filtered[(filtered >= lo) & (filtered <= hi)]

    if len(filtered) == 0:
        return _empty(method="all_outliers", rejected_windows=rejected_windows)

    # Track which original candidate BPMs survived outlier rejection (before merging).
    pre_merge_set = set(float(v) for v in filtered.tolist())
    selected_windows = [w for w in candidate_windows if float(w["bpm"]) in pre_merge_set]
    for w in candidate_windows:
        if float(w["bpm"]) not in pre_merge_set:
            bad = dict(w)
            bad["reason"] = "outlier_rejected"
            rejected_windows.append(bad)

    removed_vals = [round(float(v), 1) for v in arr.tolist() if float(v) not in pre_merge_set]

    # Part 5: merge BPM values that are within ±5 bpm of each other so that
    # closely-grouped peaks are consolidated before cluster selection.
    # IMPORTANT: each merged mean is replicated `len(group)` times to preserve
    # the original window count — this ensures the cluster-size gate counts
    # actual windows rather than collapsed group representatives.
    if len(filtered) >= 2:
        sorted_vals = np.sort(filtered).tolist()
        merged_vals: List[float] = []
        used: set = set()
        for i, vi in enumerate(sorted_vals):
            if i in used:
                continue
            group = [vi]
            for j, vj in enumerate(sorted_vals):
                if j != i and j not in used and abs(vj - vi) <= 5.0:
                    group.append(vj)
                    used.add(j)
            used.add(i)
            # Replicate the merged mean once per contributing original value
            merged_vals.extend([float(np.mean(group))] * len(group))
        filtered = np.array(merged_vals, dtype=np.float64)

    filtered_vals = [round(float(v), 1) for v in filtered.tolist()]

    std_dev = float(np.std(filtered)) if len(filtered) > 1 else 0.0
    variance = float(np.var(filtered)) if len(filtered) > 1 else 0.0
    valid_window_count = int(len(filtered))
    window_count_score = float(np.clip(valid_window_count / 8.0, 0.0, 1.0))
    stability_score = float(np.clip(1.0 - std_dev / 10.0, 0.0, 1.0))

    selected_qualities = [float(w["quality"]) for w in selected_windows]
    snr_proxy = float(np.clip(np.mean(selected_qualities) if selected_qualities else 0.0, 0.0, 1.0))

    median_hr = float(np.median(filtered))
    near_median_ratio = float(np.mean(np.abs(filtered - median_hr) <= 4.0)) if len(filtered) > 0 else 0.0
    peak_quality_score = float(np.clip(0.6 * snr_proxy + 0.4 * near_median_ratio, 0.0, 1.0))

    # Consistency threshold requested by user: std_dev > 10 bpm means unstable.
    unstable = bool(std_dev > 10.0)

    if valid_window_count < min_windows:
        return {
            "has_consensus": False,
            "heart_rate": None,
            "median_hr": round(median_hr, 1),
            "hr_values": [round(float(v), 1) for v in arr.tolist()],
            "filtered_hr_values": filtered_vals,
            "removed_outliers": removed_vals,
            "selected_windows": selected_windows,
            "rejected_windows": rejected_windows,
            "std_dev": round(std_dev, 3),
            "variance": round(variance, 3),
            "valid_window_count": valid_window_count,
            "window_count_score": round(window_count_score, 3),
            "snr_proxy": round(snr_proxy, 3),
            "peak_quality_score": round(peak_quality_score, 3),
            "stability_score": round(stability_score, 3),
            "unstable": unstable,
            "periodicity_score": round(float(np.clip(periodicity, 0.0, 1.0)), 3),
            "method": "insufficient_windows",
        }

    # Dominant-cluster selection: choose the densest BPM band (5 bpm radius ≈ ±5 bpm spread).
    sorted_filtered = np.sort(filtered)
    clusters: List[np.ndarray] = []
    current_cluster = [float(sorted_filtered[0])]
    for v in sorted_filtered[1:]:
        if abs(float(v) - current_cluster[-1]) <= 5.0:
            current_cluster.append(float(v))
        else:
            clusters.append(np.array(current_cluster, dtype=np.float64))
            current_cluster = [float(v)]
    clusters.append(np.array(current_cluster, dtype=np.float64))

    dominant = max(
        clusters,
        key=lambda c: (len(c), -float(np.std(c)) if len(c) > 1 else 0.0),
    )
    dominant_ratio = float(len(dominant) / max(len(filtered), 1))
    dominant_std = float(np.std(dominant)) if len(dominant) > 1 else 0.0
    cluster_tightness = float(np.clip(1.0 - dominant_std / 8.0, 0.0, 1.0))

    # Hard consensus gate: require cluster_size ≥ 2 and spread ≤ 12 bpm.
    # Lowered from (3, 10) to (2, 12) so that shorter/noisier scans can still
    # reach consensus when 2+ windows tightly agree, reducing harmonic-jump
    # instability across repeated scans.
    cluster_spread = float(np.max(dominant) - np.min(dominant)) if len(dominant) > 1 else 0.0
    if len(dominant) < 2 or cluster_spread > 12.0:
        return {
            "has_consensus": False,
            "heart_rate": None,
            "median_hr": round(median_hr, 1),
            "hr_values": [round(float(v), 1) for v in arr.tolist()],
            "filtered_hr_values": filtered_vals,
            "removed_outliers": removed_vals,
            "selected_windows": selected_windows,
            "rejected_windows": rejected_windows,
            "std_dev": round(std_dev, 3),
            "variance": round(variance, 3),
            "valid_window_count": valid_window_count,
            "window_count_score": round(window_count_score, 3),
            "snr_proxy": round(snr_proxy, 3),
            "peak_quality_score": round(peak_quality_score, 3),
            "dominant_cluster_size": int(len(dominant)),
            "dominant_cluster_ratio": round(dominant_ratio, 3),
            "dominant_cluster_std": round(dominant_std, 3),
            "cluster_tightness": round(cluster_tightness, 3),
            "stability_score": round(stability_score, 3),
            "unstable": True,
            "periodicity_score": round(float(np.clip(periodicity, 0.0, 1.0)), 3),
            "method": "cluster_too_small_or_wide",
        }

    # SNR + amplitude weighted average within dominant cluster for a
    # more stable final HR (Part 3 — higher weight for better windows).
    # After peak-merging, dominant values may differ from original window BPMs,
    # so match using proximity (within 5 bpm) rather than exact equality.
    dom_min = float(np.min(dominant)) - 5.0
    dom_max = float(np.max(dominant)) + 5.0
    dominant_windows = [w for w in selected_windows if dom_min <= float(w["bpm"]) <= dom_max]
    if dominant_windows:
        w_vals = []
        w_bpms = []
        for w in dominant_windows:
            snr_w = float(w.get("snr", 0.0))
            amp_w = float(w.get("amplitude", 0.0))
            q_w = float(w.get("quality", 0.0))
            # Weight = quality × (1 + snr_boost) × (1 + amp_boost)
            snr_boost = float(np.clip(snr_w * 2.0, 0.0, 1.0))
            amp_boost = float(np.clip(amp_w / 3.0, 0.0, 1.0))
            weight = q_w * (1.0 + snr_boost) * (1.0 + amp_boost)
            w_vals.append(max(weight, 1e-6))
            w_bpms.append(float(w["bpm"]))
        w_arr_d = np.array(w_vals)
        b_arr_d = np.array(w_bpms)
        final_hr = float(np.dot(b_arr_d, w_arr_d) / np.sum(w_arr_d))
    else:
        final_hr = float(np.median(dominant))

    # --- Part 4: Window consensus check ---
    # Compare final_hr against the median of all valid (non-outlier) windows.
    # A large deviation means the dominant cluster is a minority view; penalize
    # stability proportionally so downstream confidence reflects this.
    median_deviation_bpm = abs(final_hr - median_hr)
    # consensus_confidence: 1.0 at 0 deviation, 0.0 at ≥ 20 bpm deviation
    consensus_confidence = float(np.clip(1.0 - median_deviation_bpm / 20.0, 0.0, 1.0))
    # Prefer cluster-based HR is already the default path above; here we
    # just modulate stability_score as the confidence signal.
    stability_score = stability_score * (0.5 + 0.5 * consensus_confidence)

    return {
        "has_consensus": True,
        "heart_rate": round(final_hr, 1),
        "median_hr": round(median_hr, 1),
        "hr_values": [round(float(v), 1) for v in arr.tolist()],
        "filtered_hr_values": filtered_vals,
        "removed_outliers": removed_vals,
        "selected_windows": selected_windows,
        "rejected_windows": rejected_windows,
        "std_dev": round(std_dev, 3),
        "variance": round(variance, 3),
        "valid_window_count": valid_window_count,
        "window_count_score": round(window_count_score, 3),
        "snr_proxy": round(snr_proxy, 3),
        "peak_quality_score": round(peak_quality_score, 3),
        "dominant_cluster_size": int(len(dominant)),
        "dominant_cluster_ratio": round(dominant_ratio, 3),
        "dominant_cluster_std": round(dominant_std, 3),
        "cluster_tightness": round(cluster_tightness, 3),
        "stability_score": round(stability_score, 3),
        "unstable": unstable,
        "median_deviation_bpm": round(median_deviation_bpm, 1),
        "consensus_confidence": round(consensus_confidence, 3),
        "periodicity_score": round(float(np.clip(periodicity, 0.0, 1.0)), 3),
        "method": "dominant_cluster_median",
    }


# ═══════════════════════════════════════════════════════════════════════════
# POS (Plane-Orthogonal-to-Skin) colour projection
# ═══════════════════════════════════════════════════════════════════════════

def pos_project(rgb_traces: np.ndarray, fps: float) -> np.ndarray:
    """POS (Plane Orthogonal to Skin) rPPG colour projection.

    De Haan & Jeanne 2013 – projects multi-channel (R, G, B) temporal
    traces into a single pulse signal that is more robust to illumination
    changes and skin-tone variation than green-channel-only extraction.

    Parameters
    ----------
    rgb_traces : (N, 3) float64 array – per-frame [R, G, B] means.
    fps : frames per second (used only for window sizing).

    Returns
    -------
    1-D pulse signal of length N.
    """
    n = rgb_traces.shape[0]
    if n < 4:
        return np.zeros(n)

    pulse = np.zeros(n, dtype=np.float64)
    # Sliding window: ~1.6 seconds
    wlen = max(int(fps * 1.6), 4)

    for start in range(0, n - wlen + 1):
        window = rgb_traces[start: start + wlen]
        # Temporal normalisation: divide each channel by its mean
        means = np.mean(window, axis=0)
        means[means < 1e-6] = 1.0  # avoid division by zero
        cn = window / means  # (wlen, 3) normalised

        # POS projection
        s1 = cn[:, 1] - cn[:, 2]           # G - B
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]  # G + B - 2R
        std1 = np.std(s1)
        std2 = np.std(s2)
        alpha = std1 / std2 if std2 > 1e-8 else 0.0
        h = s1 + alpha * s2

        # Overlap-add
        pulse[start: start + wlen] += h - np.mean(h)

    return pulse


def signal_amplitude(signal: np.ndarray) -> float:
    """Return the amplitude (peak-to-peak) of a 1-D signal after detrending."""
    if len(signal) < 4:
        return 0.0
    s = detrend_signal(signal.astype(np.float64))
    return float(np.max(s) - np.min(s))


def signal_color_variance(signal: np.ndarray) -> float:
    """Return a 0-1 score for how large the pulsatile color variance is.

    ROIs with higher color variance carry stronger physiological signal.
    Measured as the standard deviation of the detrended trace, clipped
    to a reasonable scale.

    A score near 1.0 = strong color modulation; near 0.0 = static / flat.
    """
    if len(signal) < 4:
        return 0.0
    s = detrend_signal(signal.astype(np.float64))
    std = float(np.std(s))
    # Typical useful range: 0–5 ADU; normalise accordingly
    return float(np.clip(std / 3.0, 0.0, 1.0))


def signal_illumination_stability(signal: np.ndarray, fps: float, window_sec: float = 2.0) -> float:
    """0-1 score for how stable the illumination-driven baseline is.

    Computes per-window mean values and measures their variation **relative to
    the signal's overall amplitude** rather than their own absolute mean.  This
    avoids division-by-zero issues when the signal is zero-centred (typical of
    detrended physiological traces).  Uniform illumination → small baseline
    drift → high score.  Flickering or large drift → low score.
    """
    n = len(signal)
    win = max(int(fps * window_sec), 8)
    if n < win:
        return 0.5
    means = []
    start = 0
    while start + win <= n:
        seg = signal[start:start + win]
        means.append(float(np.mean(seg)))
        start += win // 2
    if len(means) < 2:
        return 0.5
    arr = np.array(means)
    # Normalise by signal amplitude so the metric is scale-invariant and
    # never suffers from a near-zero denominator on AC-only traces.
    sig_range = float(np.max(signal) - np.min(signal))
    if sig_range < 1e-8:
        return 0.5
    drift = float(np.std(arr) / sig_range)
    # drift < 0.02 → perfect stability; drift > 0.12 → highly unstable
    return float(np.clip(1.0 - drift / 0.12, 0.0, 1.0))


def signal_stability(signal: np.ndarray, fps: float, window_sec: float = 3.0) -> float:
    """0-1 score for how stable a signal's amplitude is across windows.

    A stable pulse signal has consistent amplitude in every window.
    """
    n = len(signal)
    win = max(int(fps * window_sec), 8)
    if n < win:
        return 0.5
    amps = []
    for start in range(0, n - win + 1, win // 2):
        seg = signal[start: start + win]
        amps.append(float(np.max(seg) - np.min(seg)))
    if len(amps) < 2:
        return 0.5
    amps_arr = np.array(amps)
    mean_amp = np.mean(amps_arr)
    if mean_amp < 1e-8:
        return 0.0
    cv = float(np.std(amps_arr) / mean_amp)  # coefficient of variation
    # Low cv = high stability: cv=0 → 1.0, cv=1 → 0.0
    return float(np.clip(1.0 - cv, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Signal Strength Scoring Engine
# ═══════════════════════════════════════════════════════════════════════════

def compute_snr(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> float:
    """Spectral SNR: peak power / mean power in the HR band.

    Returns a 0-1 score where higher = cleaner pulse peak.
    """
    n = len(signal)
    if n < 8:
        return 0.0
    filtered = bandpass_fft(signal.astype(np.float64), fps, low_hz, high_hz)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mag = np.abs(np.fft.rfft(filtered))
    valid = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(valid):
        return 0.0
    vm = mag[valid]
    mean_m = float(np.mean(vm))
    if mean_m < 1e-12:
        return 0.0
    peak = float(np.max(vm))
    snr = peak / mean_m
    return float(np.clip(snr / 12.0, 0.0, 1.0))


def compute_peak_prominence(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> float:
    """FFT peak prominence: how much the dominant peak stands above its neighbours.

    Higher prominence → more distinct heart-rate frequency → stronger signal.
    Returns 0-1 score.
    """
    n = len(signal)
    if n < 16:
        return 0.0
    filtered = bandpass_fft(signal.astype(np.float64), fps, low_hz, high_hz)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mag = np.abs(np.fft.rfft(filtered))
    valid = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(valid):
        return 0.0
    vm = mag[valid]
    if len(vm) < 3:
        return 0.0
    peak_idx = int(np.argmax(vm))
    peak_val = float(vm[peak_idx])

    # Find the nearest local minima on each side
    left_min = peak_val
    for i in range(peak_idx - 1, -1, -1):
        if vm[i] <= left_min:
            left_min = float(vm[i])
        else:
            break
    right_min = peak_val
    for i in range(peak_idx + 1, len(vm)):
        if vm[i] <= right_min:
            right_min = float(vm[i])
        else:
            break
    base = max(left_min, right_min)
    prominence = peak_val - base
    if peak_val < 1e-12:
        return 0.0
    # Normalise: prominence / peak magnitude, then scale
    rel_prominence = prominence / peak_val
    return float(np.clip(rel_prominence * 1.5, 0.0, 1.0))


def compute_harmonic_consistency(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> float:
    """Check if the 2nd harmonic of the dominant frequency is present.

    A real pulse signal often has a 2nd harmonic; noise rarely does.
    Returns 0-1 score.
    """
    n = len(signal)
    if n < 16:
        return 0.0
    sig = detrend_signal(signal.astype(np.float64))
    sig = temporal_normalization(sig, window=max(int(fps * 2), 8))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mag = np.abs(np.fft.rfft(sig))
    valid = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(valid):
        return 0.0
    vm = mag[valid]
    vf = freqs[valid]
    peak_idx = int(np.argmax(vm))
    f0 = float(vf[peak_idx])
    f0_mag = float(vm[peak_idx])
    if f0 < 0.5 or f0_mag < 1e-12:
        return 0.0

    # Check for 2nd harmonic at ~2*f0
    f2 = 2.0 * f0
    harm_band = (freqs >= f2 - 0.15) & (freqs <= f2 + 0.15)
    if not np.any(harm_band):
        return 0.2  # can't assess, assume weak presence
    harm_mag = float(np.max(mag[harm_band]))
    mean_noise = float(np.mean(mag[(freqs >= low_hz) & (freqs <= min(high_hz, 5.0))]))
    if mean_noise < 1e-12:
        return 0.2
    # Harmonic should be above noise floor but below fundamental
    harm_ratio = harm_mag / mean_noise
    return float(np.clip(harm_ratio / 5.0, 0.0, 1.0))


def compute_inter_window_consistency(
    signal: np.ndarray,
    fps: float,
    window_sec: float = 5.0,
    stride_sec: float = 1.0,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> float:
    """How consistent is the dominant frequency across sliding windows.

    Low variance in per-window BPM → high consistency → real pulse.
    Returns 0-1 score.
    """
    n = len(signal)
    wlen = max(int(fps * window_sec), 16)
    stride = max(int(fps * stride_sec), 1)
    if n < wlen:
        return 0.0
    bpms = []
    for start in range(0, n - wlen + 1, stride):
        seg = signal[start:start + wlen]
        bpm, q = estimate_bpm(seg, fps, low_hz, high_hz)
        if bpm > 0 and q > 0.05:
            bpms.append(bpm)
    if len(bpms) < 2:
        return 0.0
    arr = np.array(bpms)
    med = float(np.median(arr))
    if med < 30:
        return 0.0
    cv = float(np.std(arr) / med) if med > 0 else 1.0
    return float(np.clip(1.0 - cv * 5.0, 0.0, 1.0))


def compute_cross_window_peak_stability(
    signal: np.ndarray,
    fps: float,
    window_sec: float = 4.0,
    stride_sec: float = 0.5,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    tolerance_bpm: float = 4.0,
) -> Tuple[float, float, int]:
    """Measure how consistently the dominant peak frequency appears across windows.

    A real physiological pulse has the same frequency in most windows.
    An isolated or noise-driven peak appears in only one or two windows.

    Parameters
    ----------
    tolerance_bpm : float
        Two windows are considered to agree if their dominant BPM is
        within this many BPM of each other.

    Returns
    -------
    (stability_score, modal_bpm, support_count)
    stability_score : 0-1 — fraction of windows supporting the modal BPM
    modal_bpm : the most common BPM frequency across windows
    support_count : number of windows voting for modal_bpm
    """
    n = len(signal)
    wlen = max(int(fps * window_sec), 16)
    stride = max(int(fps * stride_sec), 1)
    if n < wlen:
        return 0.0, 0.0, 0

    window_bpms: List[float] = []
    for start in range(0, n - wlen + 1, stride):
        seg = signal[start:start + wlen]
        bpm, q = estimate_bpm(seg, fps, low_hz, high_hz)
        if bpm > 0 and q > 0.04:
            window_bpms.append(bpm)

    if len(window_bpms) < 2:
        return 0.0, 0.0, 0

    # Find modal BPM: the value with the most neighbours within tolerance
    arr = np.array(window_bpms)
    best_modal = 0.0
    best_count = 0
    for candidate in arr:
        count = int(np.sum(np.abs(arr - candidate) <= tolerance_bpm))
        if count > best_count:
            best_count = count
            best_modal = float(candidate)

    total_windows = len(window_bpms)
    stability = float(best_count / total_windows)
    # Penalize if support_count is very small (isolated peak)
    if best_count <= 1:
        stability = 0.0
    return float(np.clip(stability, 0.0, 1.0)), round(best_modal, 1), best_count


def compute_signal_strength(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> Dict[str, float]:
    """Composite signal-strength score combining multiple quality metrics.

    Includes cross-window peak stability so isolated peaks are penalised
    and only peaks supported by multiple windows are boosted.

    Returns a dict with individual metrics and a final ``signal_strength`` score:
    ``signal_strength = 0.25*snr + 0.18*periodicity + 0.18*peak_prominence
                        + 0.12*harmonic + 0.12*consistency + 0.15*peak_stability``
    """
    snr = compute_snr(signal, fps, low_hz, high_hz)
    periodicity = compute_periodicity(signal, fps, low_hz, high_hz)
    prominence = compute_peak_prominence(signal, fps, low_hz, high_hz)
    harmonic = compute_harmonic_consistency(signal, fps, low_hz, high_hz)
    consistency = compute_inter_window_consistency(signal, fps, 5.0, 1.0, low_hz, high_hz)
    peak_stability, modal_bpm, support_count = compute_cross_window_peak_stability(
        signal, fps, 4.0, 0.5, low_hz, high_hz
    )
    # Penalise if isolated (support_count <= 1) — no cross-window agreement
    isolation_penalty = 1.0 if support_count > 1 else 0.5

    strength = (
        0.25 * snr
        + 0.18 * periodicity
        + 0.18 * prominence
        + 0.12 * harmonic
        + 0.12 * consistency
        + 0.15 * peak_stability
    ) * isolation_penalty
    return {
        "signal_strength": round(float(np.clip(strength, 0.0, 1.0)), 4),
        "snr": round(snr, 4),
        "periodicity": round(periodicity, 4),
        "peak_prominence": round(prominence, 4),
        "harmonic_consistency": round(harmonic, 4),
        "inter_window_consistency": round(consistency, 4),
        "peak_stability": round(peak_stability, 4),
        "modal_bpm": modal_bpm,
        "peak_support_count": support_count,
    }


def score_window(
    segment: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    motion_score: float = 0.0,
    brightness_stability: float = 1.0,
    overexposure_ratio: float = 0.0,
) -> Dict[str, Any]:
    """Score a single time-window for suitability as an HR estimation source.

    Called by ``select_best_windows`` below.  Returns both the BPM estimate
    and a composite window quality score.
    """
    bpm, spec_quality = estimate_bpm(segment, fps, low_hz, high_hz)
    snr = compute_snr(segment, fps, low_hz, high_hz)
    periodicity = compute_periodicity(segment, fps, low_hz, high_hz)
    prominence = compute_peak_prominence(segment, fps, low_hz, high_hz)

    # Motion penalty: 0 motion → 1.0, high motion → lower
    motion_penalty = float(np.clip(1.0 / (1.0 + motion_score * 0.3), 0.0, 1.0))
    oe_penalty = float(np.clip(1.0 - overexposure_ratio / 0.35, 0.0, 1.0))

    window_strength = (
        0.25 * snr
        + 0.25 * periodicity
        + 0.15 * prominence
        + 0.10 * spec_quality
        + 0.10 * motion_penalty
        + 0.10 * brightness_stability
        + 0.05 * oe_penalty
    )
    return {
        "bpm": round(bpm, 1),
        "spec_quality": round(spec_quality, 4),
        "snr": round(snr, 4),
        "periodicity": round(periodicity, 4),
        "peak_prominence": round(prominence, 4),
        "motion_penalty": round(motion_penalty, 4),
        "brightness_stability": round(brightness_stability, 4),
        "overexposure_penalty": round(oe_penalty, 4),
        "window_strength": round(float(np.clip(window_strength, 0.0, 1.0)), 4),
    }


def select_best_windows(
    timeseries: List[Dict[str, Any]],
    signal: np.ndarray,
    fps: float,
    window_sec: float = 5.0,
    stride_sec: float = 0.5,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    motion_scores: Optional[List[float]] = None,
    brightness_scores: Optional[List[float]] = None,
    overexposure_ratios: Optional[List[float]] = None,
    top_fraction: float = 0.65,
    min_window_strength: float = 0.12,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rank sliding windows and keep only the strongest ones.

    Returns (selected_windows, rejected_windows) where each entry
    includes the window score breakdown.
    """
    n = len(signal)
    wlen = max(int(fps * window_sec), 16)
    stride = max(int(fps * stride_sec), 1)
    if n < wlen:
        return [], []

    all_windows: List[Dict[str, Any]] = []
    idx = 0
    start = 0
    while start + wlen <= n:
        seg = signal[start:start + wlen]
        # Average motion over this window's frames
        ms = 0.0
        if motion_scores:
            w_motion = motion_scores[start:start + wlen]
            ms = float(np.mean(w_motion)) if w_motion else 0.0
        bs = 1.0
        if brightness_scores:
            w_bright = brightness_scores[start:start + wlen]
            if w_bright and len(w_bright) >= 2:
                b_std = float(np.std(w_bright))
                bs = float(np.clip(1.0 - b_std / 0.3, 0.0, 1.0))
        oe = 0.0
        if overexposure_ratios:
            w_oe = overexposure_ratios[start:start + wlen]
            oe = float(np.mean(w_oe)) if w_oe else 0.0

        ws = score_window(seg, fps, low_hz, high_hz, ms, bs, oe)
        ws["index"] = idx
        ws["start_frame"] = start
        ws["end_frame"] = start + wlen
        ws["timestamp_sec"] = round((start + wlen / 2) / fps, 2)
        all_windows.append(ws)
        start += stride
        idx += 1

    if not all_windows:
        return [], []

    # Sort by window_strength descending
    all_windows.sort(key=lambda w: w["window_strength"], reverse=True)

    # Keep top fraction, but also enforce minimum strength
    n_keep = max(int(len(all_windows) * top_fraction), 1)
    selected = []
    rejected = []
    for w in all_windows:
        if len(selected) < n_keep and w["window_strength"] >= min_window_strength:
            w["status"] = "selected"
            selected.append(w)
        else:
            w["status"] = "rejected"
            w["reason"] = "below_strength_cutoff" if w["window_strength"] < min_window_strength else "below_top_fraction"
            rejected.append(w)

    return selected, rejected
