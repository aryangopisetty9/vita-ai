"""
Lightweight live face-signal analysis.

This module estimates heart rate from a compact temporal signal sampled on-device
from camera frames, so the backend does not need full frame uploads.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend.app.utils.signal_processing import (
    bandpass_fft,
    detrend_signal,
    temporal_normalization,
)


def _interp_peak_lag(ac: np.ndarray, idx: int) -> float:
    """Quadratic interpolation around autocorr peak for sub-sample lag."""
    if idx <= 0 or idx >= len(ac) - 1:
        return float(idx)
    y0 = float(ac[idx - 1])
    y1 = float(ac[idx])
    y2 = float(ac[idx + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(idx)
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(idx + delta)


def _autocorr_bpm(signal: np.ndarray, sampling_hz: float) -> Dict[str, Any]:
    """Estimate BPM from autocorrelation peak within physiological lag range."""
    if len(signal) < 8 or sampling_hz <= 0:
        return {
            "bpm": None,
            "peak": 0.0,
            "lag": None,
        }

    sig = signal.astype(np.float64)
    sig = sig - np.mean(sig)
    denom = float(np.dot(sig, sig))
    if denom <= 1e-9:
        return {
            "bpm": None,
            "peak": 0.0,
            "lag": None,
        }

    full = np.correlate(sig, sig, mode="full")
    ac = full[len(sig) - 1 :] / denom

    min_bpm = 45.0
    max_bpm = 170.0
    min_lag = int(max(1, np.floor((sampling_hz * 60.0) / max_bpm)))
    max_lag = int(min(len(ac) - 1, np.ceil((sampling_hz * 60.0) / min_bpm)))
    if max_lag <= min_lag:
        return {
            "bpm": None,
            "peak": 0.0,
            "lag": None,
        }

    search = ac[min_lag : max_lag + 1]
    if len(search) == 0:
        return {
            "bpm": None,
            "peak": 0.0,
            "lag": None,
        }

    peak_rel = int(np.argmax(search))
    peak_idx = int(min_lag + peak_rel)
    peak_val = float(ac[peak_idx])
    lag = _interp_peak_lag(ac, peak_idx)
    if lag <= 0:
        return {
            "bpm": None,
            "peak": peak_val,
            "lag": None,
        }

    bpm = float(np.clip(60.0 * sampling_hz / lag, 35.0, 200.0))
    return {
        "bpm": bpm,
        "peak": peak_val,
        "lag": lag,
    }


def analyze_live_face_signal(
    signal: List[float],
    duration_sec: float,
    sampling_hz: float,
    *,
    brightness_mean: float | None = None,
    frames_seen: int | None = None,
    frames_processed: int | None = None,
    frames_skipped: int | None = None,
) -> Dict[str, Any]:
    """Fast, timeout-safe live analysis from on-device sampled face signal."""
    sig = np.asarray(signal, dtype=np.float64)
    n = int(len(sig))

    hard_fail_reasons: List[str] = []
    if duration_sec < 10.0:
        hard_fail_reasons.append("scan_too_short")
    if n < 16:
        hard_fail_reasons.append("insufficient_signal_samples")
    if sampling_hz <= 0.0:
        hard_fail_reasons.append("invalid_sampling_rate")

    if hard_fail_reasons:
        return {
            "module_name": "face_module",
            "scan_duration_sec": float(duration_sec),
            "heart_rate": None,
            "heart_rate_unit": "bpm",
            "heart_rate_confidence": 0.0,
            "scan_quality": 0.0,
            "retake_required": True,
            "retake_reasons": hard_fail_reasons,
            "message": "Unable to estimate from live signal. Please keep face centered and retry.",
            "risk": "unreliable",
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
            "method_used": "live_signal",
            "warning": "No usable live signal was detected.",
            "debug": {
                "frames_seen": frames_seen,
                "frames_processed": frames_processed,
                "frames_skipped": frames_skipped,
                "brightness_mean": brightness_mean,
                "sample_count": n,
                "sampling_hz": sampling_hz,
                "hard_fail_reasons": hard_fail_reasons,
            },
            "metric_name": "heart_rate",
            "value": None,
            "unit": "bpm",
        }

    # Detrend + normalization for robust autocorrelation.
    proc = detrend_signal(sig)
    proc = temporal_normalization(proc, window=max(int(sampling_hz * 4), 5))
    nyquist = sampling_hz * 0.5
    low_hz = 0.7
    high_hz = min(3.2, max(0.9, nyquist - 0.05))
    proc = bandpass_fft(proc, sampling_hz, low_hz=low_hz, high_hz=high_hz)

    ac_est = _autocorr_bpm(proc, sampling_hz)
    bpm = ac_est["bpm"]
    ac_peak = float(ac_est["peak"])

    amp = float(np.std(proc))
    signal_strength = float(np.clip(amp / 1.5, 0.0, 1.0))
    periodicity = float(np.clip((ac_peak + 1.0) * 0.5, 0.0, 1.0))

    low_light = brightness_mean is not None and float(brightness_mean) < 45.0
    weak_evidence = bool(signal_strength < 0.18 or periodicity < 0.22 or low_light)

    if bpm is None:
        return {
            "module_name": "face_module",
            "scan_duration_sec": float(duration_sec),
            "heart_rate": None,
            "heart_rate_unit": "bpm",
            "heart_rate_confidence": 0.0,
            "scan_quality": 0.0,
            "retake_required": True,
            "retake_reasons": ["no_viable_candidate"],
            "message": "No viable pulse candidate found from live signal. Please retry with steadier lighting.",
            "risk": "unreliable",
            "confidence": 0.0,
            "hr_confidence": 0.0,
            "reliability": "unreliable",
            "hr_result_tier": "result_unavailable",
            "result_available": False,
            "retake_recommended": True,
            "estimated_from_weak_signal": False,
            "signal_strength": round(signal_strength, 3),
            "periodicity_score": round(periodicity, 3),
            "valid_windows": max(1, int(n / max(sampling_hz * 5.0, 1.0))),
            "method_used": "live_signal_autocorr",
            "warning": "Pulse signal too weak from live stream.",
            "debug": {
                "frames_seen": frames_seen,
                "frames_processed": frames_processed,
                "frames_skipped": frames_skipped,
                "brightness_mean": brightness_mean,
                "sample_count": n,
                "sampling_hz": sampling_hz,
                "autocorr_peak": ac_peak,
            },
            "metric_name": "heart_rate",
            "value": None,
            "unit": "bpm",
        }

    conf = float(np.clip(0.45 * periodicity + 0.35 * signal_strength + 0.20 * min(n / 60.0, 1.0), 0.05, 0.92))
    if weak_evidence:
        conf = float(np.clip(conf, 0.10, 0.38))

    reliability = "high" if conf >= 0.60 else "medium" if conf >= 0.35 else "low"
    warning = None
    retake_reasons: List[str] = []
    if weak_evidence:
        warning = "Estimated from weak live signal. Retake recommended."
        retake_reasons.append("weak_live_signal")
    if low_light:
        retake_reasons.append("low_light")

    valid_windows = max(1, int(n / max(sampling_hz * 5.0, 1.0)))

    return {
        "module_name": "face_module",
        "scan_duration_sec": float(duration_sec),
        "heart_rate": round(float(bpm), 1),
        "heart_rate_unit": "bpm",
        "heart_rate_confidence": round(conf, 3),
        "scan_quality": round(float(np.clip(0.4 * signal_strength + 0.6 * periodicity, 0.0, 1.0)), 3),
        "retake_required": bool(weak_evidence),
        "retake_reasons": retake_reasons,
        "message": "Live signal analysis completed.",
        "risk": "low",
        "confidence": round(conf, 3),
        "hr_confidence": round(conf, 3),
        "reliability": reliability,
        "hr_result_tier": "result_available",
        "result_available": True,
        "retake_recommended": bool(weak_evidence),
        "estimated_from_weak_signal": bool(weak_evidence),
        "signal_strength": round(signal_strength, 3),
        "periodicity_score": round(periodicity, 3),
        "valid_windows": valid_windows,
        "method_used": "live_signal_autocorr",
        "warning": warning,
        "debug": {
            "frames_seen": frames_seen,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "brightness_mean": brightness_mean,
            "sample_count": n,
            "sampling_hz": sampling_hz,
            "autocorr_peak": ac_peak,
            "autocorr_lag": ac_est["lag"],
            "band_hz": [low_hz, high_hz],
        },
        "metric_name": "heart_rate",
        "value": round(float(bpm), 1),
        "unit": "bpm",
    }
