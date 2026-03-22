"""
Vita AI – Audio / Breathing Module
====================================
Analyses a short breathing audio recording and estimates respiratory
metrics (breathing rate, respiratory risk).

Pipeline
--------
1. Load audio with Librosa.
2. Compute energy envelope and smooth it.
3. Find peaks in the envelope → approximate breathing cycles.
4. Derive breaths-per-minute and auxiliary features (ZCR, spectral
   centroid, MFCC summary).
5. Classify risk via rule-based heuristics.

Extension points
----------------
* Plug in a pretrained respiratory / cough classifier (e.g. YAMNet)
  by replacing ``_classify_respiratory_risk`` while
  keeping the same return schema.
* For real-time audio streaming, feed chunks into the feature
  extraction pipeline and maintain a rolling buffer.
* Store results in a database by passing the returned dict to a
  persistence layer.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False
    logger.warning("Librosa not installed – audio module will return fallback results.")

from backend.app.core.config import (
    AUDIO_MIN_DURATION,
    AUDIO_SAMPLE_RATE,
    BREATHING_NORMAL_HIGH,
    BREATHING_NORMAL_LOW,
    ENERGY_SMOOTH_WINDOW,
)
from backend.app.ml.audio.audio_models import (
    compare_with_librosa_pipeline,
    infer_audio_models,
)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_error_result(message: str, debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a well-formed error result so callers never crash."""
    return {
        "module_name": "audio_module",
        "metric_name": "breathing_rate",
        "value": None,
        "unit": "breaths/min",
        "confidence": 0.0,
        "breathing_rate": None,
        "reliability": "unreliable",
        "signal_strength": 0.0,
        "warning": message,
        "risk": "error",
        "message": message,
        "retake_required": True,
        "retake_reasons": [message],
        "debug": debug or {},
    }


def _smooth(signal: np.ndarray, window_len: int) -> np.ndarray:
    """Simple moving-average smoothing."""
    if window_len < 2:
        return signal
    kernel = np.ones(window_len) / window_len
    return np.convolve(signal, kernel, mode="same")


# ── Audio preprocessing helpers ──────────────────────────────────────────

def _normalize_amplitude(y: np.ndarray) -> np.ndarray:
    """Peak-normalise audio to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y
    return y / peak


def _trim_silence(y: np.ndarray, sr: int, top_db: float = 25.0) -> np.ndarray:
    """Remove leading/trailing silence using librosa.effects.trim."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def _spectral_noise_reduce(y: np.ndarray, sr: int,
                           noise_frames: int = 5) -> np.ndarray:
    """Simple spectral-subtraction noise reduction.

    Estimates the noise floor from the first *noise_frames* STFT frames
    and subtracts it from the magnitude spectrum.  This is a lightweight
    alternative to noisereduce / RNNoise.
    """
    n_fft = 2048
    hop = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(stft), np.angle(stft)

    # Estimate noise floor from the quietest portion
    nf = min(noise_frames, mag.shape[1])
    if nf < 1:
        return y
    noise_profile = np.mean(mag[:, :nf], axis=1, keepdims=True)

    # Subtract noise (floored at zero to avoid negative magnitudes)
    cleaned_mag = np.maximum(mag - noise_profile * 1.2, 0.0)
    cleaned_stft = cleaned_mag * np.exp(1j * phase)
    return librosa.istft(cleaned_stft, hop_length=hop, length=len(y))


def _bandpass_breathing(y: np.ndarray, sr: int,
                        low_hz: float = 0.1, high_hz: float = 2.0) -> np.ndarray:
    """Band-pass the envelope signal to the breathing frequency range.

    Normal breathing is 12-20 breaths/min = 0.2-0.33 Hz.  We use a
    wider 0.1-2.0 Hz band to handle tachypnoea up to 120 breaths/min.
    High-rate breathing (48-70/min = 0.8-1.17 Hz) must not be cut off.
    """
    n = len(y)
    if n < 8:
        return y
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    fft_vals = np.fft.rfft(y)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    fft_vals[~mask] = 0.0
    return np.fft.irfft(fft_vals, n=n)


def _speech_ratio(y: np.ndarray, sr: int) -> float:
    """Estimate the fraction of frames that contain speech-like content.

    Uses zero-crossing rate as a proxy, but rapid breathing through the
    mouth raises ZCR on its own.  We also check spectral centroid: speech
    tends to have much higher centroid (1000+ Hz) than breathing (<600 Hz).
    The combined gate reduces false positives for fast breathing.
    Returns 0-1 where >0.5 suggests mostly speech.
    """
    frame_len = 2048
    hop = 512
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    # Rapid breathing turbulence raises ZCR to ~0.08-0.12 but centroid
    # stays below ~700 Hz.  Real speech has ZCR AND centroid both high.
    speech_frames = np.sum((zcr > 0.12) & (centroid > 700.0))
    return float(speech_frames / max(len(zcr), 1))


def _spectral_flatness_check(y: np.ndarray, sr: int) -> float:
    """Spectral flatness (Wiener entropy) of the raw audio signal.

    Measures how tone-like vs noise-like the signal is:
      Pure tone (fan, AC hum)     : flatness < 0.01
      Tonal + harmonics (HVAC)    : flatness 0.01 – 0.03
      Breathing turbulence        : flatness 0.05 – 0.40
      White / broadband noise     : flatness → 1.0

    Used to reject narrow-band mechanical noise before breathing analysis.
    """
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    return float(np.mean(flatness))


def _extract_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Extract audio features using Librosa.

    Returns a dict of features for downstream analysis.

    # ── EXTENSION POINT ──────────────────────────────────────────
    # Add more features here (e.g. chroma, tonnetz) or feed raw
    # features into a trained classifier instead of rule-based logic.
    # ─────────────────────────────────────────────────────────────
    """
    duration = float(len(y) / sr)

    # Root-mean-square energy
    rms = librosa.feature.rms(y=y)[0]

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # MFCCs (first 13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1).tolist()

    return {
        "duration_sec": round(duration, 2),
        "rms": rms,
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "mfcc_means": mfcc_means,
    }


# ── Multi-method breathing estimation helpers ───────────────────────────

def _fft_breathing_rate(
    envelope: np.ndarray, env_sr: float,
    low_hz: float = 0.07, high_hz: float = 2.0,
) -> float:
    """Dominant breathing frequency from FFT of the energy envelope.

    Searches the [low_hz, high_hz] frequency band (default 0.07–2.0 Hz =
    4–120 breaths/min) for the strongest spectral component.
    """
    n = len(envelope)
    if n < 8:
        return 0.0
    env_centered = envelope - float(np.mean(envelope))
    fft_mag = np.abs(np.fft.rfft(env_centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / env_sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    valid_mags = fft_mag[mask]
    max_mag = float(np.max(valid_mags))
    if max_mag < 1e-10:
        return 0.0
    # Spectral peak prominence: the dominant bin must stand clearly above the
    # spectral noise floor inside the breathing band.  For ambient noise (even
    # AGC-amplified by the browser's WebRTC stack), the spectrum is relatively
    # flat and peak/median is typically 2–3×.  A real breathing signal
    # concentrates energy into a narrow spike, giving a ratio well above 4×.
    # Without this gate, random low-frequency environmental noise is reported
    # as a valid breathing frequency (e.g. 70 bpm from a computer fan).
    # Spectral PNR must be high enough that the peak clearly stands above
    # broadband noise.  AGC-amplified ambient noise typically has PNR 2–4×.
    # Real breathing concentrates energy into a narrow band, yielding 8–15×.
    _SPECTRAL_PNR = 6.0
    noise_floor_mag = float(np.median(valid_mags))
    if noise_floor_mag <= 0 or max_mag / noise_floor_mag < _SPECTRAL_PNR:
        return 0.0
    peak_freq = float(freqs[mask][np.argmax(valid_mags)])
    rate = peak_freq * 60.0
    return round(rate, 1) if 4.0 <= rate <= 120.0 else 0.0


def _autocorr_breathing_rate(
    envelope: np.ndarray, env_sr: float,
    min_bpm: float = 4.0, max_bpm: float = 120.0,
) -> float:
    """Autocorrelation-based breathing rate from the energy envelope.

    Finds the dominant periodic lag in the range corresponding to
    [min_bpm, max_bpm] breaths/min.
    """
    n = len(envelope)
    if n < 10:
        return 0.0
    env_centered = envelope - float(np.mean(envelope))
    norm = float(np.dot(env_centered, env_centered))
    if norm < 1e-12:
        return 0.0
    padded = 2 * n
    fft_env = np.fft.rfft(env_centered, n=padded)
    acorr = np.fft.irfft(fft_env * np.conj(fft_env))[:n].real
    acorr /= (norm + 1e-12)
    min_lag = max(1, int((60.0 / max_bpm) * env_sr))
    max_lag = min(n - 1, int((60.0 / min_bpm) * env_sr))
    if min_lag >= max_lag:
        return 0.0
    best_offset = int(np.argmax(acorr[min_lag : max_lag + 1]))
    best_lag = best_offset + min_lag
    # Autocorrelation coherence: the normalized correlation at the detected
    # period must be meaningfully positive.  For pure or AGC-amplified noise,
    # the autocorrelation is near zero at all lags > 0 (std ≈ 1/√N, where N is
    # the number of envelope samples).  Real periodic breathing produces a clear
    # positive peak close to 1.  A 0.12 threshold is ~3σ above zero for a
    # 500-sample envelope (10 s recording at 50 Hz envelope sample rate).
    # Coherence must be well above the noise floor (~1/√N ≈ 0.04 for a
    # 500-sample envelope).  AGC-amplified room noise can reach 0.10–0.18
    # at individual lags.  Real periodic breathing is typically 0.35–0.80.
    _ACORR_MIN_COHERENCE = 0.25
    if acorr[best_lag] < _ACORR_MIN_COHERENCE:
        return 0.0
    period_sec = best_lag / env_sr
    return round(60.0 / period_sec, 1) if period_sec > 0 else 0.0


def _peak_breathing_rate_adaptive(
    envelope: np.ndarray,
    env_sr: float,
    min_dist_sec: float = 0.5,
) -> tuple[float, list[int], list[float]]:
    """Peak-based breathing rate with adaptive threshold and dynamic spacing.

    High-rate breathing (48-70+/min) produces many small closely-spaced
    peaks.  To handle this correctly:

    * Threshold uses the 45th percentile (lower than 55th) to avoid
      rejecting real peaks in a dense fast signal.
    * A two-pass approach: first detect with min_dist_sec, then if the
      implied rate exceeds 40/min, reduce min_dist to 0.4 s and re-detect.
    * Raw peak count is returned for debug.

    Returns (bpm, peak_indices, interval_seconds).
    """
    n = len(envelope)
    if n < 5:
        return 0.0, [], []

    # Peak prominence: a detected peak must stand out above the local
    # valley (minimum in a window around each peak) by at least this fraction
    # of the envelope's dynamic range.  Noise has tiny uniform bumps that fail
    # this check.  We use a half-window of 0.5 s on each side (or the full
    # min_dist) to find the baseline around each candidate peak.
    _env_range = float(np.max(envelope) - np.min(envelope)) if len(envelope) > 1 else 0.0
    _MIN_PROMINENCE = max(0.10 * _env_range, 1e-6)
    _prom_hw = max(int(0.5 * env_sr), 3)  # half-window in samples

    def _detect(min_d_sec: float, pct: float) -> tuple[list[int], list[float]]:
        min_d = max(int(min_d_sec * env_sr), 1)
        thr = float(np.percentile(np.abs(envelope), pct))
        pks: list[int] = []
        for i in range(1, n - 1):
            if envelope[i] > envelope[i - 1] and envelope[i] > envelope[i + 1]:
                if envelope[i] > thr:
                    # Check peak prominence: peak must rise above the local
                    # valley (min value in a window around it)
                    left_bound = max(0, i - _prom_hw)
                    right_bound = min(n, i + _prom_hw + 1)
                    local_min = float(np.min(envelope[left_bound:right_bound]))
                    prom = envelope[i] - local_min
                    if prom < _MIN_PROMINENCE:
                        continue
                    if not pks or (i - pks[-1]) >= min_d:
                        pks.append(i)
        if len(pks) < 2:
            return pks, []
        ivs = [float((pks[j] - pks[j - 1]) / env_sr) for j in range(1, len(pks))]
        return pks, ivs

    # First pass: standard parameters
    peaks, intervals = _detect(min_dist_sec, 45.0)

    if len(peaks) >= 2:
        avg_iv = float(np.mean(intervals))
        implied_bpm = 60.0 / avg_iv if avg_iv > 0 else 0.0
        # High-rate mode: if implied BPM > 40, try tighter spacing
        if implied_bpm > 40.0:
            peaks2, intervals2 = _detect(0.35, 40.0)
            if len(peaks2) >= len(peaks):
                peaks, intervals = peaks2, intervals2

    if len(peaks) < 2:
        return 0.0, peaks, []
    avg_interval = float(np.mean(intervals))
    bpm = 60.0 / avg_interval if avg_interval > 0 else 0.0
    if not 4.0 <= bpm <= 120.0:
        return 0.0, peaks, intervals
    return round(bpm, 1), peaks, intervals


def _detect_subharmonic(
    rate: float, other_rates: list[float], tol: float = 0.18,
) -> tuple[float, bool]:
    """Correct subharmonic detection: returns (rate×2, True) when it matches better.

    Peak detection can lock onto every second breath cycle, yielding half the
    true rate.  If doubling the candidate agrees with more of the other method
    estimates than the original rate, the original was a subharmonic and is
    corrected upward.
    """
    if rate <= 0:
        return rate, False
    harmonic = rate * 2.0
    if harmonic > 120.0:
        return rate, False

    def _close(a: float, b: float) -> bool:
        return abs(a - b) / max(a, b, 1e-6) < tol

    harmonic_votes = sum(1 for r in other_rates if r > 0 and _close(r, harmonic))
    rate_votes     = sum(1 for r in other_rates if r > 0 and _close(r, rate))
    if harmonic_votes > rate_votes:
        return round(harmonic, 1), True
    return rate, False


def _fuse_rate_estimates(
    fft_rate: float, autocorr_rate: float, peak_rate: float,
) -> tuple[float, float, int, str]:
    """Fuse three independent breathing-rate estimates into a consensus value.

    Finds the cluster of estimates that agree within 20% of each other and
    returns their weighted mean.  High-rate bias correction: when all three
    estimates are positive but disagree, prefer the highest-rate agreeing
    pair (FFT + autocorr) over collapsing to a lower peak estimate, because
    peak detection systematically under-counts at high rates.

    Returns (fused_rate, agreement_score 0–1, agreement_count, method_label).
    """
    TOL = 0.22  # slightly wider tolerance — high-rate spread is larger in absolute terms

    def _close(a: float, b: float) -> bool:
        return a > 0 and b > 0 and abs(a - b) / max(a, b) < TOL

    valid = [
        (r, name)
        for r, name in [(fft_rate, "fft"), (autocorr_rate, "autocorr"), (peak_rate, "peak")]
        if 4.0 <= r <= 120.0
    ]
    if not valid:
        return 0.0, 0.0, 0, "none"
    if len(valid) == 1:
        return valid[0][0], 0.45, 1, valid[0][1]

    best_rate, best_votes, best_name, best_group = 0.0, 0, "", []
    for r, name in valid:
        group = [rv for rv, _ in valid if _close(r, rv)]
        if len(group) > best_votes:
            best_votes = len(group)
            best_rate  = float(np.mean(group))
            best_name  = name
            best_group = group

    agreement_score = float(min(1.0, best_votes / max(len(valid), 1)))
    method_label    = f"{best_name}_{best_votes}of{len(valid)}"

    # Disagreement handling: when no two methods agree, prefer the
    # spectral methods (FFT + autocorr) over peak detection, because
    # peak detection systematically under-counts at fast rates.
    if best_votes == 1 and len(valid) >= 2:
        rates_only = [r for r, _ in valid]
        max_spread = max(
            abs(a - b) / max(a, b, 1e-6)
            for i, a in enumerate(rates_only)
            for b in rates_only[i + 1:]
        ) if len(rates_only) >= 2 else 0.0
        if max_spread > 0.25:
            # High-rate bias: if FFT and autocorr are both valid and their
            # mean is ≥ peak_rate * 1.2, trust the spectral pair.
            if fft_rate > 0 and autocorr_rate > 0:
                spectral_mean = (fft_rate + autocorr_rate) / 2.0
                spectral_agree = _close(fft_rate, autocorr_rate)
                if spectral_agree:
                    best_rate    = round(spectral_mean, 1)
                    best_votes   = 2
                    best_name    = "fft+autocorr"
                    agreement_score = 0.70  # two spectral methods agree
                    method_label = "spectral_pair_over_peak"
                else:
                    # All three fully disagree — cap score but do not zero it
                    agreement_score = 0.25
                    method_label    = f"all_disagree_{best_name}"
            else:
                agreement_score = 0.25
                method_label    = f"all_disagree_{best_name}"

    return round(best_rate, 1), agreement_score, best_votes, method_label


def _estimate_breathing_rate(
    y: np.ndarray, sr: int,
) -> tuple[float, int, float, dict]:
    """Estimate breathing rate using multi-method fusion.

    Applies FFT, autocorrelation, and adaptive peak detection on the 20 ms
    RMS energy envelope.  The three independent estimates are reconciled and
    corrected for subharmonic detection (where peak detection locks onto every
    second breath cycle, yielding half the true rate).

    Key improvements over the single-method predecessor
    ---------------------------------------------------
    * Short 150 ms smoothing window preserves fast breathing peaks.
    * Bandpass extended to 2.0 Hz (120 breaths/min).
    * Peak min-distance 0.5 s allows up to 120 breaths/min.
    * 55th-percentile adaptive threshold replaces median + 0.1 * std.
    * Subharmonic correction via cross-method voting.
    * Three-method fusion with agreement scoring.

    Returns
    -------
    (breaths_per_min, cycle_count, quality_score, debug_dict)
    """
    hop    = int(sr * 0.02)           # 20 ms hop → ~50 Hz envelope sample rate
    env_sr = float(sr / hop)
    rms    = librosa.feature.rms(y=y, hop_length=hop)[0]

    if len(rms) < 5:
        return 0.0, 0, 0.0, {"fft_rate": 0.0, "autocorr_rate": 0.0, "peak_rate_raw": 0.0,
                              "raw_peak_count": 0, "merged_peak_count": 0, "speech_score": 0.0,
                              "method_agreement": 0, "high_rate_mode_used": False}

    # Amplitude gate: if the signal is too quiet, reject before further processing.
    envelope_rms_mean = float(np.mean(rms))
    _RMS_GATE = 5e-4  # empirically ~-66 dBFS; below this the mic captured near-silence
    _empty_debug = {
        "fft_rate": 0.0, "autocorr_rate": 0.0, "peak_rate_raw": 0.0,
        "raw_peak_count": 0, "merged_peak_count": 0, "speech_score": 0.0,
        "method_agreement": 0, "high_rate_mode_used": False,
    }
    if envelope_rms_mean < _RMS_GATE:
        return 0.0, 0, 0.0, {
            **_empty_debug,
            "rejection_reason": f"signal_too_quiet (mean_rms={envelope_rms_mean:.2e} < {_RMS_GATE:.2e})",
        }

    # ── Envelope periodicity gate ──────────────────────────────────────
    # Real breathing creates clear periodic peaks and valleys in the RMS
    # envelope.  AGC-amplified ambient noise produces a relatively flat
    # envelope with tiny random fluctuations.  The coefficient of variation
    # (CV = std/mean) of the raw envelope separates the two cleanly:
    #   Noise (flat envelope)       : CV  0.03 – 0.15
    #   Breathing (periodic envelope): CV  0.25 – 0.80+
    _ENV_CV_MIN = 0.18  # below this the envelope is too flat to contain breathing
    _env_std = float(np.std(rms))
    _env_cv = _env_std / (envelope_rms_mean + 1e-12)
    if _env_cv < _ENV_CV_MIN:
        return 0.0, 0, 0.0, {
            **_empty_debug,
            "rejection_reason": (
                f"envelope_too_flat (cv={_env_cv:.4f} < {_ENV_CV_MIN}): "
                "no periodic breathing pattern detected in the energy envelope"
            ),
            "envelope_cv": round(_env_cv, 5),
        }

    # ── Smoothing: short window (80 ms) to preserve fast breathing peaks ──
    # 68/min → period ~0.88 s; 49/min → ~1.22 s.  At env_sr≈50 Hz,
    # 80 ms = 4 samples — tight enough to keep high-rate oscillations.
    short_win      = max(int(0.08 * env_sr), 2)
    envelope_short = _smooth(rms, short_win)

    # Bandpass [0.07, 2.0] Hz = [4, 120] breaths/min for peak detection
    envelope_bp = _bandpass_breathing(envelope_short, env_sr, low_hz=0.07, high_hz=2.0)

    # ── Three independent estimates ──────────────────────────────────────
    fft_rate      = _fft_breathing_rate(envelope_short, env_sr)
    autocorr_rate = _autocorr_breathing_rate(envelope_short, env_sr)
    peak_rate, peaks, intervals = _peak_breathing_rate_adaptive(
        envelope_bp, env_sr, min_dist_sec=0.4,   # 0.4 s → up to 150/min physically
    )
    raw_peak_count = len(peaks)  # before any further merging

    # ── Subharmonic correction ────────────────────────────────────────────
    other_estimates              = [r for r in [fft_rate, autocorr_rate] if r > 0]
    peak_rate_corr, harmonic_corrected = _detect_subharmonic(peak_rate, other_estimates)

    # ── High-rate mode: if spectral methods suggest fast breathing,
    #    run a second tighter peak pass and prefer the result if it
    #    agrees better with the spectral estimates.
    high_rate_mode_used = False
    _spectral_mean = 0.0
    if fft_rate >= 40.0 or autocorr_rate >= 40.0:
        high_rate_mode_used = True
        _spectral_mean = (
            (fft_rate + autocorr_rate) / 2.0 if fft_rate > 0 and autocorr_rate > 0
            else max(fft_rate, autocorr_rate)
        )
        peak_rate_hr, peaks_hr, intervals_hr = _peak_breathing_rate_adaptive(
            envelope_bp, env_sr, min_dist_sec=0.30,
        )
        if len(peaks_hr) >= 2:
            bpm_hr_diff  = abs(peak_rate_hr - _spectral_mean)
            bpm_std_diff = abs(peak_rate_corr - _spectral_mean)
            if bpm_hr_diff < bpm_std_diff:
                peak_rate_corr = peak_rate_hr
                peaks          = peaks_hr
                intervals      = intervals_hr

    merged_peak_count = len(peaks)  # after high-rate re-detection

    # ── Fuse the three estimates ─────────────────────────────────────────
    fused_rate, agreement_score, agreement_count, method_label = _fuse_rate_estimates(
        fft_rate, autocorr_rate, peak_rate_corr,
    )

    # ── Quality: interval consistency + multi-method agreement ───────────
    if len(intervals) >= 2:
        avg_interval     = float(np.mean(intervals))
        cv               = float(np.std(intervals) / (avg_interval + 1e-6))
        # Penalise suspiciously regular intervals — human breathing always
        # has natural jitter (CV typically 0.10-0.50).  Machine-perfect
        # regularity (CV < 0.06) is a hallmark of fan / AC modulation.
        if cv < 0.06:
            interval_quality = 0.12  # severely penalised — likely mechanical
        else:
            cv_normalised    = float(np.clip(cv / 0.3, 0.0, 1.0))
            interval_quality = float(np.clip(1.0 - cv_normalised * 0.7, 0.10, 1.0))
    else:
        interval_quality = 0.25

    cycle_count = len(peaks)
    quality = float(np.clip(
        0.5 * interval_quality
        + 0.3 * agreement_score
        + 0.2 * min(1.0, cycle_count / 5.0),
        0.05, 1.0,
    ))

    debug_dict: dict = {
        "fft_rate":                    fft_rate,
        "autocorr_rate":               autocorr_rate,
        "peak_rate":                   peak_rate,
        "peak_rate_raw":               peak_rate,          # backward compat alias
        "peak_rate_corrected":         peak_rate_corr,
        "fused_rate":                  fused_rate,
        "harmonic_correction_applied": harmonic_corrected,
        "agreement_count":             agreement_count,
        "agreement_score":             round(agreement_score, 3),
        "method_agreement":            agreement_count,    # requested field name
        "method_label":                method_label,
        "detected_peaks":              cycle_count,
        "raw_peak_count":              raw_peak_count,
        "merged_peak_count":           merged_peak_count,
        "high_rate_mode_used":         high_rate_mode_used,
        "speech_score":                0.0,                # filled in at call site
        "interval_cv":                 round(cv, 4) if len(intervals) >= 2 else None,
        "interval_quality":            round(interval_quality, 4),
        "breath_intervals":            [round(x, 3) for x in intervals],
        "smoothing_window_ms":         round(short_win / env_sr * 1000, 1),
    }

    # ── Interval spike filtering: trim top/bottom 15% of intervals before
    #    computing the final mean.  This avoids one missed or double peak
    #    from pulling the reported rate by several BPM.  Skip rolling-
    #    median smoothing that was biasing slow-trend over fast cycles.
    if len(intervals) >= 4:
        sorted_iv = sorted(intervals)
        trim_k    = max(1, len(sorted_iv) // 7)   # ~15% trim
        core_iv   = sorted_iv[trim_k : len(sorted_iv) - trim_k] or sorted_iv
        avg_iv_trim = float(np.mean(core_iv))
        if avg_iv_trim > 0:
            bpm_trim = 60.0 / avg_iv_trim
            if 4.0 <= bpm_trim <= 120.0:
                fused_rate = round(bpm_trim, 1)
        debug_dict["intervals_trimmed"] = [round(x, 3) for x in core_iv]

    # Low-agreement cap: keep quality honest but do NOT zero the rate.
    if agreement_score < 0.20:
        quality = float(np.clip(quality * 0.5, 0.0, 0.35))
        debug_dict["rejection_reason"] = "low_method_agreement"

    if fused_rate <= 0:
        return 0.0, cycle_count, 0.1, debug_dict

    return round(fused_rate, 1), cycle_count, round(quality, 3), debug_dict


def _classify_breathing(rate: float) -> tuple[str, str]:
    """Map breathing rate to a risk label and a plain-language message.

    Risk labels are kept distinct from estimate quality:
      normal    12–20/min  resting adult range
      elevated  20–30/min  light exercise or moderately elevated
      high      30–60/min  vigorous exercise / tachypnea
      very_high >60/min    extreme rate (hyperventilation / very vigorous)
      low_rate  <12/min    bradypnea
      error     no rate
    """
    if rate <= 0:
        return "error", "Could not determine breathing rate."
    if rate < BREATHING_NORMAL_LOW:
        return "low_rate", (
            f"Breathing rate ({rate:.0f}/min) is below the typical resting range "
            f"({BREATHING_NORMAL_LOW}\u2013{BREATHING_NORMAL_HIGH}/min)."
        )
    if rate <= BREATHING_NORMAL_HIGH:
        return "normal", (
            f"Breathing rate ({rate:.0f}/min) is within the normal resting range."
        )
    if rate <= 30:
        return "elevated", (
            f"Breathing rate ({rate:.0f}/min) is above the resting range — "
            "typical of light activity or mild elevation."
        )
    if rate <= 60:
        return "high", (
            f"Breathing rate ({rate:.0f}/min) is elevated — consistent with "
            "vigorous exercise or tachypnea."
        )
    return "very_high", (
        f"Breathing rate ({rate:.0f}/min) is very high — typical of extreme "
        "exertion or hyperventilation."
    )


def _derive_audio_reliability(confidence: float, quality: float, cycles: int, has_signal: bool) -> str:
    """Map breathing evidence to a reliability label.

    Thresholds are calibrated against real-mic recordings:
      high   → estimate is trustworthy for tracking trends
      medium → estimate is plausible; minor retake might improve it
      low    → signal was weak; treat as indicative only
    """
    if not has_signal:
        return "unreliable"
    if confidence >= 0.62 and quality >= 0.45 and cycles >= 4:
        return "high"
    if confidence >= 0.32 and quality >= 0.20 and cycles >= 2:
        return "medium"
    return "low"


def _ensure_loadable(audio_path: str) -> str:
    """Convert webm/ogg to wav if needed so librosa can load it."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in {'.wav', '.mp3', '.flac', '.m4a'}:
        return audio_path  # librosa handles these natively
    # Need ffmpeg for webm/ogg conversion
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return audio_path  # fall back – let librosa try
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    subprocess.run(
        [ffmpeg_exe, '-i', audio_path, '-ar', str(AUDIO_SAMPLE_RATE),
         '-ac', '1', tmp.name, '-y'],
        capture_output=True, check=True,
    )
    return tmp.name


# ---------------------------------------------------------------------------
# Audio recovery helpers
# ---------------------------------------------------------------------------

def _reject_speech_frames(y: np.ndarray, sr: int, zcr_threshold: float = 0.08) -> np.ndarray:
    """Zero out speech-like (high-ZCR) frames, keeping only breathing-like frames."""
    frame_len = 2048
    hop = 512
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop)[0]
    out = np.zeros_like(y)
    for i, zcr_val in enumerate(zcr):
        start = i * hop
        end = min(start + frame_len, len(y))
        if zcr_val <= zcr_threshold:
            out[start:end] += y[start:end]
    peak = np.max(np.abs(out))
    if peak > 1e-8:
        out /= peak
    return out


def _estimate_breathing_rate_relaxed(y: np.ndarray, sr: int) -> tuple[float, int, float]:
    """Breathing-rate estimation with relaxed peak-detection parameters.

    Uses short smoothing and wider bandpass so fast breathing (up to
    120/min) is not cut off.  Quality is capped at 0.60.  Returns
    (0, n, 0) rather than a fabricated value when fewer than 2 peaks
    are detected.
    """
    hop = int(sr * 0.02)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    env_sr = float(sr / hop)
    # Short smoothing (80 ms) to preserve fast cycles
    smooth_samples = max(int(0.08 * env_sr), 2)
    envelope = _smooth(rms, smooth_samples)
    if len(envelope) < 4:
        return 0.0, 0, 0.0
    # Wide bandpass: 0.07-2.0 Hz → 4-120 breaths/min
    envelope = _bandpass_breathing(envelope, env_sr, low_hz=0.07, high_hz=2.0)
    # Tight min_dist so fast breathing is not under-counted
    min_dist = max(int(0.4 * env_sr), 1)
    threshold = np.median(np.abs(envelope)) * 0.7
    peaks: List[int] = []
    for i in range(1, len(envelope) - 1):
        if envelope[i] > envelope[i - 1] and envelope[i] > envelope[i + 1]:
            if abs(envelope[i]) > threshold:
                if not peaks or (i - peaks[-1]) >= min_dist:
                    peaks.append(i)
    cycle_count = len(peaks)
    if cycle_count < 2:
        return 0.0, cycle_count, 0.0
    intervals = np.diff(peaks) * (1.0 / env_sr)
    avg_interval = float(np.mean(intervals))
    if avg_interval <= 0:
        return 0.0, cycle_count, 0.0
    bpm = 60.0 / avg_interval
    if not (4.0 <= bpm <= 120.0):   # was mistakenly capped at 65 — removed
        return 0.0, cycle_count, 0.0
    cv = float(np.std(intervals) / avg_interval) if len(intervals) > 1 else 0.5
    quality = float(np.clip(0.60 - cv * 0.5, 0.10, 0.60))
    return round(bpm, 1), cycle_count, round(quality, 3)


def _estimate_breathing_rate_best_segment(
    y: np.ndarray, sr: int,
    window_sec: float = 5.0,
    stride_sec: float = 1.0,
) -> tuple[float, int, float]:
    """Slide a fixed-length window over the signal; return the highest-quality estimate."""
    window_len = int(window_sec * sr)
    stride = int(stride_sec * sr)
    if len(y) < window_len:
        bpm, cycles, quality, _ = _estimate_breathing_rate(y, sr)
        return bpm, cycles, quality
    best_bpm, best_cycles, best_quality = 0.0, 0, 0.0
    for start in range(0, len(y) - window_len + 1, stride):
        bpm, cycles, quality, _ = _estimate_breathing_rate(y[start:start + window_len], sr)
        if bpm > 0 and quality > best_quality:
            best_bpm, best_cycles, best_quality = bpm, cycles, quality
    return best_bpm, best_cycles, round(best_quality, 3)


def _package_audio_result(
    best: Optional[Dict[str, Any]],
    attempt_log: List[Dict[str, Any]],
    stage_reached: int,
) -> Dict[str, Any]:
    success = best is not None and best["quality"] >= 0.12 and best["rate"] > 0
    return {
        "rate": best["rate"] if success else 0.0,
        "cycles": best["cycles"] if success else 0,
        "quality": best["quality"] if success else 0.0,
        "method": best["method"] if success else "none",
        "recovery_success": success,
        "recovery_attempts": len(attempt_log),
        "stages_reached": stage_reached,
        "attempt_log": attempt_log,
    }


def _recover_breathing_rate(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """4-stage recovery orchestrator for breathing-rate estimation.

    Tries progressively more permissive strategies on the same preprocessed
    audio before giving up.  Never returns a fabricated value — every estimate
    comes from real peak detection on the actual signal.

    Stages
    ------
    1. Aggressive silence trim (top_db=35 vs default 25).
    2. Speech-frame rejection (zero out high-ZCR frames).
    3. Best-segment sliding window (5 s window, 1 s stride).
    4. Relaxed peak-detection (lower min_dist + threshold).

    Returns a dict with: rate, cycles, quality, method, recovery_success,
    recovery_attempts, stages_reached, attempt_log.
    """
    attempt_log: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    def _record(stage: int, method: str, bpm: float, cycles: int, quality: float, note: str = "") -> None:
        nonlocal best
        viable = bpm > 0 and quality >= 0.12 and 4.0 <= bpm <= 120.0
        entry: Dict[str, Any] = {
            "stage": stage,
            "method": method,
            "bpm": round(bpm, 1) if bpm > 0 else None,
            "cycles": cycles,
            "quality": round(quality, 3),
            "viable": viable,
            "note": note,
        }
        attempt_log.append(entry)
        if viable and (best is None or quality > best["quality"]):
            best = {"rate": round(bpm, 1), "cycles": cycles, "quality": round(quality, 3), "method": method}

    # Stage 1: Aggressive silence trim
    try:
        y_trim, _ = librosa.effects.trim(y, top_db=35.0)
        bpm, cycles, quality, _ = _estimate_breathing_rate(y_trim, sr)
        _record(1, "aggressive_trim", bpm, cycles, quality,
                f"trimmed to {len(y_trim)/sr:.1f}s")
    except Exception as exc:  # noqa: BLE001
        attempt_log.append({"stage": 1, "method": "aggressive_trim", "error": str(exc)})

    if best and best["quality"] >= 0.35:
        return _package_audio_result(best, attempt_log, stage_reached=1)

    # Stage 2: Speech-frame rejection
    try:
        y_no_speech = _reject_speech_frames(y, sr)
        bpm, cycles, quality, _ = _estimate_breathing_rate(y_no_speech, sr)
        _record(2, "speech_rejection", bpm, cycles, quality)
    except Exception as exc:  # noqa: BLE001
        attempt_log.append({"stage": 2, "method": "speech_rejection", "error": str(exc)})

    if best and best["quality"] >= 0.30:
        return _package_audio_result(best, attempt_log, stage_reached=2)

    # Stage 3: Best-segment sliding window (5 s, 1 s stride)
    try:
        bpm, cycles, quality = _estimate_breathing_rate_best_segment(y, sr, window_sec=5.0, stride_sec=1.0)
        _record(3, "best_segment_5s", bpm, cycles, quality, "highest-quality 5 s window selected")
    except Exception as exc:  # noqa: BLE001
        attempt_log.append({"stage": 3, "method": "best_segment_5s", "error": str(exc)})

    if best and best["quality"] >= 0.22:
        return _package_audio_result(best, attempt_log, stage_reached=3)

    # Stage 4: Relaxed peak-detection parameters
    try:
        bpm, cycles, quality = _estimate_breathing_rate_relaxed(y, sr)
        _record(4, "relaxed_params", bpm, cycles, quality,
                "min_dist=1.0 s, threshold=median×0.8, requires ≥2 cycles")
    except Exception as exc:  # noqa: BLE001
        attempt_log.append({"stage": 4, "method": "relaxed_params", "error": str(exc)})

    return _package_audio_result(best, attempt_log, stage_reached=4)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """Analyse a breathing audio file and return respiratory metrics.

    Parameters
    ----------
    audio_path : str
        Path to an audio file (wav, mp3, etc.).

    Returns
    -------
    dict
        Standardised module result.

    Integration notes
    -----------------
    * **Flutter / mobile**: Upload audio via ``POST /predict/audio``.
    * **Database**: Persist the returned dict to the user's health record.
    * **YAMNet**: Replace ``_classify_respiratory_risk`` with
      a pretrained classifier for cough / wheeze detection.
    """
    if not _HAS_LIBROSA:
        return _build_error_result(
            "Librosa is not installed. Cannot process audio.",
            {"dependency_missing": "librosa"},
        )

    if not audio_path or not os.path.isfile(audio_path):
        return _build_error_result(
            "Audio file not found or path is invalid.",
            {"audio_path": audio_path},
        )

    # Load audio – convert non-standard formats (webm/ogg) to wav first
    try:
        load_path = _ensure_loadable(audio_path)
        y, sr = librosa.load(load_path, sr=AUDIO_SAMPLE_RATE, mono=True)
        # Clean up temp file if conversion created one
        if load_path != audio_path:
            try:
                os.unlink(load_path)
            except OSError:
                pass
    except Exception as exc:
        return _build_error_result(
            f"Failed to load audio file: {exc}",
            {"audio_path": audio_path},
        )

    duration = len(y) / sr
    if duration < AUDIO_MIN_DURATION:
        return _build_error_result(
            f"Audio too short ({duration:.1f}s). Need at least {AUDIO_MIN_DURATION}s.",
            {"duration_sec": round(duration, 2)},
        )

    # ── Preprocessing pipeline ────────────────────────────────────
    y = _trim_silence(y, sr)

    # ── Pre-normalization silence gate ───────────────────────────
    # CRITICAL: this check MUST run on the raw (non-normalised) signal.
    # After peak-normalization even pure ambient noise is amplified to
    # full scale, making any post-normalization RMS gate useless.
    #
    # Typical signal levels (linear RMS, 0.0 – 1.0 full scale):
    #   Pure silence / off mic : < 0.001  (-60 dBFS)
    #   Ambient room noise     : 0.001 – 0.008  (-60 to -42 dBFS)
    #   Faint distant breathing: 0.008 – 0.018  (-42 to -35 dBFS)
    #   Normal breathing       : 0.018+          (-35 dBFS and up)
    _PRE_NORM_RMS_MIN = 0.012   # ~-38 dBFS — below this is ambient noise, not breathing
    _pre_norm_rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
    if _pre_norm_rms < _PRE_NORM_RMS_MIN:
        return _build_error_result(
            "No breathing detected in the recording. "
            "Please hold the microphone closer and breathe normally for at least 10 seconds.",
            {"pre_norm_rms": round(_pre_norm_rms, 6), "rms_threshold": _PRE_NORM_RMS_MIN},
        )

    # ── Spectral flatness gate: reject narrow-band mechanical noise ────
    # Fan, AC, and HVAC hum concentrate energy at a few discrete
    # frequencies (low spectral flatness).  Real breathing turbulence is
    # broadband.  This gate fires *before* normalization or noise
    # reduction so the original spectral shape is intact.
    _spectral_flat = _spectral_flatness_check(y, sr)
    _SFLATNESS_MIN = 0.02
    if _spectral_flat < _SFLATNESS_MIN:
        return _build_error_result(
            "Environmental noise detected (fan, AC, or mechanical hum). "
            "Please move to a quieter place and record again.",
            {"spectral_flatness": round(_spectral_flat, 6),
             "threshold": _SFLATNESS_MIN,
             "rejection_type": "narrow_band_noise"},
        )

    y = _normalize_amplitude(y)
    y = _spectral_noise_reduce(y, sr)

    # Check for excessive speech content (talking ≠ breathing)
    sr_ratio = _speech_ratio(y, sr)

    # Feature extraction
    features = _extract_features(y, sr)

    # Breathing rate estimation
    breathing_rate, cycle_count, quality, est_debug = _estimate_breathing_rate(y, sr)

    risk, message = _classify_breathing(breathing_rate)

    # ── Confidence: four independent factors, each bringing unique information ──
    #
    # quality         : from _estimate_breathing_rate — already encodes interval CV,
    #                   method agreement, and cycle count. This is the primary signal.
    # agreement_factor: standalone method-agreement score (fft/autocorr/peak cluster).
    # cycle_adequacy  : physical evidence — more detected cycles = more convincing.
    # duration_adequacy: longer recording → more reliable period estimation.
    #
    # Speech penalty: real rapid breathing raises ZCR and sometimes centroid, so
    # the double-gated _speech_ratio already filters most false positives.  We
    # still apply a gentle penalty but floor it at 0.65 to avoid over-penalising
    # turbulent breathing.  When >=6 cycles are detected the signal is clearly
    # breathing regardless of residual turbulence noise.

    agreement_factor   = float(est_debug.get("agreement_score", 0.0))
    cycle_adequacy     = float(np.clip(cycle_count / 8.0, 0.0, 1.0))
    duration_adequacy  = float(np.clip((features["duration_sec"] - 5.0) / 20.0, 0.0, 1.0))
    feature_quality    = 0.5 * duration_adequacy + 0.5 * cycle_adequacy  # kept for debug compat

    # Speech penalty: only penalise content clearly above a breathing-turbulence
    # baseline (~15 % of frames).  Slope reduced vs. old formula.
    _sr_above_baseline = max(0.0, sr_ratio - 0.15)
    speech_penalty = float(np.clip(1.0 - _sr_above_baseline * 1.2, 0.65, 1.0))
    if cycle_count >= 6:
        speech_penalty = max(speech_penalty, 0.85)  # evidence of real breathing overrides

    confidence = (
        0.40 * quality          # dominant: internal quality already encodes CV+agreement+cycles
        + 0.30 * agreement_factor  # method consensus
        + 0.20 * cycle_adequacy    # physical evidence of breathing cycles
        + 0.10 * duration_adequacy # recording completeness
    ) * speech_penalty

    # Signal-strength gate: prevent over-confident reports from genuinely weak signals.
    signal_strength_audio = float(np.clip(0.65 * quality + 0.35 * min(cycle_count / 8.0, 1.0), 0.0, 1.0))
    if signal_strength_audio < 0.35:
        confidence = float(np.clip(confidence, 0.0, 0.38))
    elif signal_strength_audio >= 0.58:
        confidence = float(np.clip(confidence, 0.0, 0.92))
    else:
        confidence = float(np.clip(confidence, 0.0, 0.68))
    confidence = round(float(confidence), 2)

    suppression_reason: Optional[str] = None
    warning: Optional[str] = None

    # ── Recovery: if primary estimation was genuinely poor, try multi-strategy ──
    # Only attempt recovery if the envelope showed SOME periodic structure.
    # If the primary estimation got 0 from all three methods (FFT, autocorr,
    # peaks all returned 0), the signal is almost certainly noise — recovery
    # would only find false positives.
    _all_methods_zero = (
        est_debug.get("fft_rate", 0) == 0
        and est_debug.get("autocorr_rate", 0) == 0
        and est_debug.get("peak_rate_raw", 0) == 0
    )
    _recovery_triggered = (
        (breathing_rate <= 0 or quality < 0.20 or cycle_count < 2)
        and not _all_methods_zero  # don't recover pure noise
    )
    _recovery_result: Optional[Dict[str, Any]] = None
    _recovery_succeeded = False
    if _recovery_triggered:
        _recovery_result = _recover_breathing_rate(y, sr)
        _recovery_succeeded = _recovery_result["recovery_success"]
        if _recovery_succeeded and _recovery_result["rate"] > 0 and _recovery_result["quality"] >= 0.25:
            breathing_rate = _recovery_result["rate"]
            cycle_count = _recovery_result["cycles"]
            quality = _recovery_result["quality"]
            if suppression_reason:
                # Recovery found a real signal — lift suppression
                suppression_reason = None
                risk, message = _classify_breathing(breathing_rate)
                message += f" (Estimated via multi-strategy recovery: {_recovery_result['method']}.)"
            # Recalculate confidence using the same structured formula, capped at 0.80
            _rec_cycle_factor   = float(np.clip(cycle_count / 5.0, 0.0, 1.0))
            _rec_signal_str     = float(np.clip(0.65 * quality + 0.35 * _rec_cycle_factor, 0.0, 1.0))
            _rec_duration       = float(np.clip(features["duration_sec"] / 15.0, 0.0, 1.0))
            _rec_speech_penalty = float(np.clip(1.0 - sr_ratio * 1.5, 0.0, 1.0))
            confidence = (
                0.25 * _rec_signal_str
                + 0.25 * quality          # interval proxy from recovery quality
                + 0.20 * 0.5              # agreement unknown after recovery → assume 0.5
                + 0.15 * 0.33             # single method
                + 0.15 * _rec_duration
            ) * _rec_speech_penalty
            confidence = round(float(np.clip(confidence * 0.80, 0.0, 0.80)), 2)

    # ── Post-hoc mechanical noise check ─────────────────────────────────
    # If spectral flatness is in the grey zone (0.02–0.10) and the detected
    # intervals are suspiciously regular (CV < 0.08), the signal is almost
    # certainly mechanical fan/AC modulation rather than breathing.
    _est_cv = est_debug.get("interval_cv")
    _is_mechanical = False
    if breathing_rate > 0 and _spectral_flat < 0.10:
        if _est_cv is not None and _est_cv < 0.08:
            _is_mechanical = True
        elif _spectral_flat < 0.03:
            _is_mechanical = True  # very narrow-band even without CV evidence

    if _is_mechanical:
        breathing_rate = 0.0
        confidence = 0.0
        quality = 0.0
        risk = "unreliable"
        message = (
            "Environmental noise detected (fan, AC, or mechanical hum). "
            "Please move to a quieter place and record again."
        )
        warning = message
        suppression_reason = (
            f"mechanical_noise (spectral_flatness={_spectral_flat:.4f}, "
            f"interval_cv={_est_cv})"
        )

    # Require BOTH quality and cycle count to exceed safe minima.
    # Raised thresholds to prevent AGC-amplified noise from passing.
    has_signal = breathing_rate > 0 and quality >= 0.30 and cycle_count >= 3
    signal_strength = float(np.clip(0.65 * quality + 0.35 * min(cycle_count / 6.0, 1.0), 0.0, 1.0))

    if not has_signal:
        breathing_rate = 0.0
        confidence = 0.0
        risk = "unreliable"
        message = (
            "Unable to estimate breathing rate: no usable breathing cycles were detected. "
            "Please record at least 10-15 seconds in a quiet environment."
        )
        warning = "No measurable breathing cycles found."
        suppression_reason = (
            f"Insufficient breathing signal (cycles={cycle_count}, quality={quality:.3f})"
        )
    elif quality < 0.18 or cycle_count < 2:
        # Genuinely weak signal — keep the value but cap confidence.
        confidence = float(np.clip(confidence, 0.08, 0.38))
        warning = "Low confidence result: estimated from weak signal."

    reliability = _derive_audio_reliability(
        confidence=confidence,
        quality=quality,
        cycles=cycle_count,
        has_signal=has_signal,
    )

    # Retake logic
    retake_required = False
    retake_reasons: List[str] = []
    if sr_ratio > 0.5:
        retake_reasons.append(
            "Recording contains mostly speech. Please breathe normally "
            "without talking and re-record."
        )
        retake_required = True
    if cycle_count < 3 and suppression_reason is None and not _recovery_succeeded:
        retake_reasons.append(
            "Too few breathing cycles detected. Record for at least 10 seconds "
            "of calm breathing."
        )
        retake_required = True
    if quality < 0.2 and breathing_rate > 0:
        retake_reasons.append(
            "Breathing pattern was irregular — results may be unreliable."
        )
    # Force retake when confidence is too low to show a meaningful result.
    if confidence < 0.45 and breathing_rate > 0:
        retake_required = True
        retake_reasons.append(
            "Confidence too low to report a reliable breathing rate "
            f"({int(confidence * 100)}%). Please record again in a quiet environment."
        )
    # Only suggest retake for genuinely unreliable estimates, not just medium confidence.
    if reliability == "low" and (cycle_count < 3 or quality < 0.15):
        retake_required = True
        retake_reasons.append("Weak breathing signal detected — retake for a more accurate result.")
    if suppression_reason:
        retake_required = True
        retake_reasons.append(suppression_reason)

    # Fill in speech_score in the debug dict returned from _estimate_breathing_rate
    est_debug["speech_score"] = round(sr_ratio, 4)

    debug_info: Dict[str, Any] = {
        "duration_sec":    features["duration_sec"],
        "estimated_cycles": cycle_count,
        "feature_quality": round(feature_quality, 3),
        "cycle_quality":   round(quality, 3),
        "speech_ratio":    round(sr_ratio, 3),
        "speech_score":    round(sr_ratio, 4),  # requested field alias
        "zcr_mean":        round(features["zcr_mean"], 5),
        "spectral_centroid_mean": round(features["spectral_centroid_mean"], 2),
        "suppression_reason": suppression_reason,
        "rejection_reason": est_debug.get("rejection_reason"),
        "spectral_flatness": round(_spectral_flat, 6),
        "is_mechanical_noise": _is_mechanical,
        "signal_strength": round(signal_strength, 3),
        "method_used": (_recovery_result or {}).get("method", "primary_envelope"),
        "recovery_triggered": _recovery_triggered,
        "recovery_success": _recovery_succeeded,
        "recovery": {k: v for k, v in _recovery_result.items() if k != "attempt_log"}
            if _recovery_result else None,
        # Multi-method estimation debug (all requested fields)
        "fft_rate":                    est_debug.get("fft_rate"),
        "autocorr_rate":               est_debug.get("autocorr_rate"),
        "peak_rate":                   est_debug.get("peak_rate"),
        "peak_rate_raw":               est_debug.get("peak_rate_raw"),
        "peak_rate_corrected":         est_debug.get("peak_rate_corrected"),
        "harmonic_correction_applied": est_debug.get("harmonic_correction_applied", False),
        "agreement_score":             est_debug.get("agreement_score", 0.0),
        "method_agreement":            est_debug.get("method_agreement", 0),
        "method_label":                est_debug.get("method_label", ""),
        "detected_peaks":              est_debug.get("detected_peaks", 0),
        "raw_peak_count":              est_debug.get("raw_peak_count", 0),
        "merged_peak_count":           est_debug.get("merged_peak_count", 0),
        "high_rate_mode_used":         est_debug.get("high_rate_mode_used", False),
        "breath_intervals":            est_debug.get("breath_intervals", []),
        "intervals_trimmed":           est_debug.get("intervals_trimmed"),
        "smoothing_window_ms":         est_debug.get("smoothing_window_ms"),
    }

    # --- Pretrained audio model inference (optional layer) ---
    model_result = infer_audio_models(y, sr)
    comparison = compare_with_librosa_pipeline(
        model_result,
        librosa_risk=risk,
        librosa_confidence=confidence,
        librosa_breathing_rate=breathing_rate if breathing_rate > 0 else None,
    )
    debug_info["audio_model"] = {
        "model_used": comparison.get("model_name", "none"),
        "source": comparison.get("source", "librosa_pipeline"),
        "selection_reason": comparison.get("selection_reason", ""),
        "available_models": model_result.get("available_models", []),
        "yamnet_available": "yamnet" in model_result.get("available_models", []),
        "yamnet_loaded": model_result.get("model_loaded", False) and model_result.get("model", "") == "yamnet",
        "yamnet_cached": model_result.get("model_cached", False),
        "inference_source": comparison.get("inference_source", "librosa_pipeline"),
    }
    # If model escalated risk, use model's assessment
    if comparison["source"] not in ("librosa_pipeline",):
        risk = comparison["risk"]
        confidence = comparison["confidence"]
        message = _classify_breathing(breathing_rate)[1]
        if comparison.get("model_labels"):
            message += " Audio model detected: " + ", ".join(comparison["model_labels"]) + "."

    return {
        "module_name": "audio_module",
        "metric_name": "breathing_rate",
        "value": breathing_rate if breathing_rate > 0 else None,
        "breathing_rate": breathing_rate if breathing_rate > 0 else None,
        "unit": "breaths/min",
        "breathing_rate_normal_low": BREATHING_NORMAL_LOW,
        "breathing_rate_normal_high": BREATHING_NORMAL_HIGH,
        "confidence": confidence,
        "reliability": reliability,
        "signal_strength": round(signal_strength, 3),
        "warning": warning,
        "risk": risk,
        "message": message,
        "retake_required": retake_required,
        "retake_reasons": retake_reasons,
        "debug": debug_info,
    }
