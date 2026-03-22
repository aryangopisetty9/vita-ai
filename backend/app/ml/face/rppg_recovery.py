"""
Vita AI – rPPG Result Recovery Engine
=======================================

Multi-stage recovery that maximises the chance of returning a real
heart-rate estimate before declaring a scan unreliable.

Triggered when the initial single-pass estimation fails or produces
very low quality.  All values returned are derived from actual signal
data — no constants, fabricated confidence, or hardcoded defaults.

Recovery stages
---------------
1. **Individual ROI green-channel** – try each ROI independently;
   the global fusion may have averaged away a strong single ROI.
2. **Individual ROI POS projection** – same ROIs with colour
   projection, which is more illumination-robust.
3. **Windowed consensus (green)** – rolling windows scored
   independently; only high-quality windows vote on the final BPM
   via median consensus.
4. **Windowed consensus (POS)** – same rolling window strategy with
   POS projection.
5. **Combined ROI signal** – average all ROI green traces together
   and re-run, useful when no single ROI dominates but the collective
   signal carries a clear pulse.
6. **Combined ROI POS** – as above but with POS projection.

Each stage records a structured attempt log for debug output
(``recovery_attempts``, ``windows_tested``, ``windows_accepted``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.utils.signal_processing import (
    compute_periodicity,
    detrend_signal,
    estimate_bpm,
    pos_project,
)

# ── Viability thresholds (more permissive than the main pipeline) ─────────
_RECOVERY_MIN_PERIODICITY = 0.12  # lower than the main quality gate
_RECOVERY_MIN_BPM = 38.0
_RECOVERY_MAX_BPM = 200.0


# ═══════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════════════

def _is_viable(bpm: Optional[float], periodicity: float) -> bool:
    """Return True if the (bpm, periodicity) pair is worth reporting."""
    return (
        bpm is not None
        and _RECOVERY_MIN_BPM < bpm < _RECOVERY_MAX_BPM
        and periodicity >= _RECOVERY_MIN_PERIODICITY
    )


def _bpm_from_green(
    trace: List[float],
    fps: float,
    low_hz: float,
    high_hz: float,
) -> Tuple[float, float, float]:
    """Returns (bpm, snr_quality, periodicity) from a green-channel trace."""
    if len(trace) < 8:
        return 0.0, 0.0, 0.0
    arr = np.array(trace, dtype=np.float64)
    bpm, q = estimate_bpm(arr, fps, low_hz, high_hz)
    period = compute_periodicity(arr, fps, low_hz, high_hz)
    return float(bpm), float(q), float(period)


def _bpm_from_pos(
    rgb_trace: List[np.ndarray],
    fps: float,
    low_hz: float,
    high_hz: float,
) -> Tuple[float, float, float]:
    """Returns (bpm, snr_quality, periodicity) via POS colour projection."""
    if len(rgb_trace) < 8:
        return 0.0, 0.0, 0.0
    rgb = np.array(rgb_trace, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return 0.0, 0.0, 0.0
    pulse = pos_project(rgb, fps)
    if float(np.std(pulse)) < 1e-8:
        return 0.0, 0.0, 0.0
    bpm, q = estimate_bpm(pulse, fps, low_hz, high_hz)
    period = compute_periodicity(pulse, fps, low_hz, high_hz)
    return float(bpm), float(q), float(period)


# ═══════════════════════════════════════════════════════════════════════════
# Rolling-window consensus helpers
# ═══════════════════════════════════════════════════════════════════════════

def _windowed_bpm_consensus(
    trace: np.ndarray,
    fps: float,
    low_hz: float,
    high_hz: float,
    window_sec: float = 6.0,
    stride_sec: float = 1.0,
    min_window_score: float = 0.10,
) -> Tuple[float, float, float, int, int]:
    """Compute a consensus BPM from high-quality rolling windows.

    Parameters
    ----------
    trace : 1-D float array (green-channel or pulse signal)
    window_sec : length of each analysis window
    stride_sec : step between windows
    min_window_score : combined periodicity+quality threshold to accept

    Returns
    -------
    (consensus_bpm, consensus_quality, consensus_periodicity,
     windows_tested, windows_accepted)
    """
    n = len(trace)
    win = max(int(window_sec * fps), 8)
    stride = max(int(stride_sec * fps), 1)

    good_bpms: List[float] = []
    good_scores: List[float] = []
    windows_tested = 0

    start = 0
    while start + win <= n:
        seg = trace[start: start + win]
        bpm_w, q_w = estimate_bpm(seg, fps, low_hz, high_hz)
        per_w = compute_periodicity(seg, fps, low_hz, high_hz)
        score = 0.4 * q_w + 0.6 * per_w
        if _RECOVERY_MIN_BPM < bpm_w < _RECOVERY_MAX_BPM and score >= min_window_score:
            good_bpms.append(bpm_w)
            good_scores.append(score)
        windows_tested += 1
        start += stride

    if not good_bpms:
        return 0.0, 0.0, 0.0, windows_tested, 0

    # Median consensus is robust to outlier windows
    consensus_bpm = float(np.median(good_bpms))
    consensus_quality = float(np.mean(good_scores))
    # Periodicity re-evaluated on the full signal at the consensus frequency
    consensus_period = compute_periodicity(trace, fps, low_hz, high_hz)

    return (
        round(consensus_bpm, 1),
        round(consensus_quality, 4),
        round(consensus_period, 4),
        windows_tested,
        len(good_bpms),
    )


def _windowed_bpm_consensus_pos(
    rgb_trace: List[np.ndarray],
    fps: float,
    low_hz: float,
    high_hz: float,
    window_sec: float = 6.0,
    stride_sec: float = 1.0,
    min_window_score: float = 0.10,
) -> Tuple[float, float, float, int, int]:
    """Like _windowed_bpm_consensus but operates on an RGB trace via POS."""
    rgb = np.array(rgb_trace, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) < 8:
        return 0.0, 0.0, 0.0, 0, 0

    pulse = pos_project(rgb, fps)
    if float(np.std(pulse)) < 1e-8:
        return 0.0, 0.0, 0.0, 0, 0

    return _windowed_bpm_consensus(
        pulse, fps, low_hz, high_hz, window_sec, stride_sec, min_window_score
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main recovery orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def recover_heart_rate(
    roi_traces: Dict[str, List[float]],
    roi_rgb_traces: Dict[str, List[np.ndarray]],
    fps: float,
    low_hz: float,
    high_hz: float,
) -> Dict[str, Any]:
    """Attempt multi-stage recovery to find a valid HR estimate.

    All result values come from signal computations on the actual
    collected traces — no fabricated defaults.

    Parameters
    ----------
    roi_traces : {roi_name: [green_channel_value_per_frame]}
    roi_rgb_traces : {roi_name: [np.array([R,G,B]) per frame]}
    fps : frames per second
    low_hz, high_hz : heart-rate band-pass limits

    Returns
    -------
    Dict with keys:
        bpm, quality, periodicity               — best viable result (or None/0.0)
        roi_used                                — ROI that produced the best result
        extraction_method_used                  — method name string
        recovery_attempts                       — total attempt count
        recovery_stages_tried                   — list of stage names tried
        windows_tested, windows_accepted        — aggregate window counts
        recovery_success                        — bool
        attempt_log                             — detailed per-attempt list
    """
    attempts: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    total_windows_tested = 0
    total_windows_accepted = 0

    def _record(
        stage: str,
        roi: str,
        method: str,
        bpm: float,
        quality: float,
        periodicity: float,
        w_tested: int = 0,
        w_accepted: int = 0,
    ) -> None:
        nonlocal best, total_windows_tested, total_windows_accepted
        total_windows_tested += w_tested
        total_windows_accepted += w_accepted
        viable = _is_viable(bpm if bpm > 0 else None, periodicity)
        entry: Dict[str, Any] = {
            "stage": stage,
            "roi": roi,
            "method": method,
            "bpm": round(bpm, 1) if bpm else None,
            "quality": round(quality, 4),
            "periodicity": round(periodicity, 3),
            "windows_tested": w_tested,
            "windows_accepted": w_accepted,
            "viable": viable,
        }
        attempts.append(entry)
        # Keep the attempt with the best periodicity (most reliable signal quality)
        if viable and (best is None or periodicity > best["periodicity"]):
            best = entry.copy()

    # ── Stage 1: Individual ROI – green channel ──────────────────────────
    for roi_name, trace in roi_traces.items():
        if len(trace) < 20:
            continue
        bpm, q, period = _bpm_from_green(trace, fps, low_hz, high_hz)
        _record("individual_roi_green", roi_name, "green_channel", bpm, q, period)

    # ── Stage 2: Individual ROI – POS projection ─────────────────────────
    for roi_name, rgb_trace in roi_rgb_traces.items():
        if len(rgb_trace) < 20:
            continue
        bpm, q, period = _bpm_from_pos(rgb_trace, fps, low_hz, high_hz)
        _record("individual_roi_pos", roi_name, "pos_projection", bpm, q, period)

    # Early exit: if we already have a strong result, no need to continue
    if best and best["periodicity"] >= 0.30:
        return _package_result(best, attempts, total_windows_tested,
                                total_windows_accepted, "early_exit_strong_signal")

    # ── Stage 3: Rolling-window consensus – green channel ────────────────
    for roi_name, trace in roi_traces.items():
        if len(trace) < 30:
            continue
        arr = np.array(trace, dtype=np.float64)
        bpm, q, period, w_t, w_a = _windowed_bpm_consensus(
            arr, fps, low_hz, high_hz,
            window_sec=6.0, stride_sec=1.5, min_window_score=0.10,
        )
        _record("windowed_consensus_green", roi_name, "windowed_green",
                bpm, q, period, w_t, w_a)

    # ── Stage 4: Rolling-window consensus – POS projection ───────────────
    for roi_name, rgb_trace in roi_rgb_traces.items():
        if len(rgb_trace) < 30:
            continue
        bpm, q, period, w_t, w_a = _windowed_bpm_consensus_pos(
            rgb_trace, fps, low_hz, high_hz,
            window_sec=6.0, stride_sec=1.5, min_window_score=0.10,
        )
        _record("windowed_consensus_pos", roi_name, "windowed_pos",
                bpm, q, period, w_t, w_a)

    # Early exit: medium-quality result found
    if best and best["periodicity"] >= 0.18:
        return _package_result(best, attempts, total_windows_tested,
                                total_windows_accepted, "early_exit_medium_signal")

    # ── Stage 5: Combined ROI average – green channel ────────────────────
    valid_traces = [
        np.array(t, dtype=np.float64)
        for t in roi_traces.values()
        if len(t) >= 20
    ]
    if len(valid_traces) >= 2:
        min_len = min(len(t) for t in valid_traces)
        combined_green = np.mean(
            np.stack([t[:min_len] for t in valid_traces]), axis=0
        )
        bpm, q, period = _bpm_from_green(list(combined_green), fps, low_hz, high_hz)
        _record("combined_rois_green", "all", "combined_green", bpm, q, period)

        # Also try windowed on combined signal
        if len(combined_green) >= 30:
            bpm, q, period, w_t, w_a = _windowed_bpm_consensus(
                combined_green, fps, low_hz, high_hz,
                window_sec=6.0, stride_sec=2.0, min_window_score=0.08,
            )
            _record("windowed_combined_green", "all", "windowed_combined_green",
                    bpm, q, period, w_t, w_a)

    # ── Stage 6: Combined ROI average – POS projection ───────────────────
    valid_rgb = [
        np.array(t, dtype=np.float64)
        for t in roi_rgb_traces.values()
        if len(t) >= 20
    ]
    if len(valid_rgb) >= 2:
        min_len = min(len(t) for t in valid_rgb)
        combined_rgb = np.mean(
            np.stack([t[:min_len] for t in valid_rgb]), axis=0
        )
        bpm, q, period = _bpm_from_pos(list(combined_rgb), fps, low_hz, high_hz)
        _record("combined_rois_pos", "all", "combined_pos", bpm, q, period)

    return _package_result(
        best, attempts, total_windows_tested, total_windows_accepted,
        "exhausted_all_stages",
    )


def _package_result(
    best: Optional[Dict[str, Any]],
    attempts: List[Dict[str, Any]],
    total_windows_tested: int,
    total_windows_accepted: int,
    exit_reason: str,
) -> Dict[str, Any]:
    stages_tried = list(dict.fromkeys(a["stage"] for a in attempts))  # ordered unique
    if best:
        return {
            "bpm": best["bpm"],
            "quality": best["quality"],
            "periodicity": best["periodicity"],
            "roi_used": best["roi"],
            "extraction_method_used": best["method"],
            "recovery_attempts": len(attempts),
            "recovery_stages_tried": stages_tried,
            "windows_tested": total_windows_tested,
            "windows_accepted": total_windows_accepted,
            "recovery_success": True,
            "exit_reason": exit_reason,
            "attempt_log": attempts,
        }
    return {
        "bpm": None,
        "quality": 0.0,
        "periodicity": 0.0,
        "roi_used": None,
        "extraction_method_used": None,
        "recovery_attempts": len(attempts),
        "recovery_stages_tried": stages_tried,
        "windows_tested": total_windows_tested,
        "windows_accepted": total_windows_accepted,
        "recovery_success": False,
        "exit_reason": exit_reason,
        "attempt_log": attempts,
    }
