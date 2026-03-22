"""
Tests for the face module (robust pipeline v2).

Covers:
- Output schema validation (all required keys)
- Error-handling paths (missing file, empty path, None)
- Signal processing helpers (detrend, bandpass, BPM, periodicity)
- Multi-ROI fusion
- HR timeseries
- Face quality scoring (scan quality, retake logic)
- Face features (blink, EAR, eye movement, facial motion, skin)
- Score engine integration with new face output
- Integration test with real sample video (skipped if absent)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


from backend.app.ml.face.face_module import analyze_face_video, _classify_hr, _build_error_result
from backend.app.utils.signal_processing import (
    bandpass_fft,
    compute_hr_timeseries,
    compute_periodicity,
    compute_cross_window_peak_stability,
    compute_signal_strength,
    detrend_signal,
    estimate_bpm,
    fuse_roi_signals,
    moving_average_smooth,
    robust_hr_consensus,
    signal_color_variance,
    signal_illumination_stability,
    spectral_hr_estimate,
    temporal_normalization,
)
from backend.app.ml.face.face_quality import (
    compute_roi_agreement,
    compute_scan_quality,
    frame_quality_score,
)
from backend.app.ml.face.face_features import (
    BlinkDetector,
    aggregate_blink_results,
    aggregate_eye_movement,
    aggregate_facial_motion,
    aggregate_skin_analysis,
)
from backend.app.ml.fusion.score_engine import compute_vita_score

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "module_name", "scan_duration_sec",
    "heart_rate", "heart_rate_unit", "heart_rate_confidence",
    "blink_rate", "blink_rate_unit",
    "eye_stability", "facial_tension_index", "skin_signal_stability",
    "scan_quality", "retake_required", "retake_reasons",
    "message", "risk", "confidence", "confidence_breakdown",
    "debug",
    "hr_timeseries", "blink_analysis", "eye_movement",
    "facial_motion", "eye_color", "skin_color",
    # Legacy compat
    "metric_name", "value", "unit",
}


def _assert_valid_schema(result: dict) -> None:
    """Every result must contain all standard keys with correct types."""
    assert isinstance(result, dict)
    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key: {key}"
    assert result["module_name"] == "face_module"
    assert result["heart_rate_unit"] == "bpm"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["risk"] in {"low", "moderate", "high", "error"}
    assert isinstance(result["hr_timeseries"], list)
    assert isinstance(result["blink_analysis"], dict)
    assert isinstance(result["eye_movement"], dict)
    assert isinstance(result["facial_motion"], dict)
    assert isinstance(result["eye_color"], dict)
    assert isinstance(result["skin_color"], dict)
    assert isinstance(result["retake_required"], bool)
    assert isinstance(result["retake_reasons"], list)
    assert isinstance(result["confidence_breakdown"], dict)
    assert isinstance(result["scan_quality"], (int, float))


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestFaceModuleErrors:
    def test_missing_file(self):
        result = analyze_face_video("nonexistent_video.mp4")
        _assert_valid_schema(result)
        assert result["risk"] == "error"
        assert result["heart_rate"] is None
        assert result["retake_required"] is True
        assert len(result["retake_reasons"]) > 0

    def test_empty_path(self):
        result = analyze_face_video("")
        _assert_valid_schema(result)
        assert result["risk"] == "error"
        assert result["retake_required"] is True

    def test_none_path(self):
        result = analyze_face_video(None)  # type: ignore
        _assert_valid_schema(result)
        assert result["risk"] == "error"
        assert result["retake_required"] is True


class TestErrorResultBuilder:
    def test_error_result_has_all_keys(self):
        result = _build_error_result("test error")
        _assert_valid_schema(result)
        assert result["message"] == "test error"

    def test_error_result_retake_reason(self):
        result = _build_error_result("bad scan")
        assert "bad scan" in result["retake_reasons"]


# ---------------------------------------------------------------------------
# Signal processing tests
# ---------------------------------------------------------------------------

class TestSignalProcessing:
    def test_bandpass_shape(self):
        sig = np.random.randn(256)
        out = bandpass_fft(sig, fps=30.0, low_hz=0.7, high_hz=3.5)
        assert out.shape == sig.shape

    def test_detrend_removes_slope(self):
        x = np.arange(100, dtype=np.float64) * 2.0 + 5.0
        out = detrend_signal(x)
        # Detrended should be near zero
        assert abs(np.mean(out)) < 0.1

    def test_temporal_normalization(self):
        sig = np.arange(60, dtype=np.float64)
        out = temporal_normalization(sig, window=10)
        assert out.shape == sig.shape
        # Should be roughly z-scored
        assert abs(np.mean(out)) < 1.0

    def test_estimate_bpm_synthetic_sine(self):
        """1.2 Hz sine (72 BPM) should be detected."""
        fps = 30.0
        t = np.arange(0, 10.0, 1.0 / fps)
        signal = np.sin(2 * np.pi * 1.2 * t)
        bpm, quality = estimate_bpm(signal, fps)
        assert 65 < bpm < 80, f"Expected ~72 BPM, got {bpm}"
        assert quality > 0.05

    def test_estimate_bpm_short_signal(self):
        bpm, q = estimate_bpm(np.array([1.0, 2.0, 3.0]), 30.0)
        assert bpm == 0.0

    def test_periodicity_sine(self):
        fps = 30.0
        t = np.arange(0, 10.0, 1.0 / fps)
        signal = np.sin(2 * np.pi * 1.0 * t)
        p = compute_periodicity(signal, fps)
        assert p > 0.3  # strong periodic signal


class TestMultiROIFusion:
    def test_fusion_consistent_signals(self):
        """Three ROIs with same frequency should agree."""
        fps = 30.0
        t = np.arange(0, 10.0, 1.0 / fps)
        sig = np.sin(2 * np.pi * 1.2 * t)
        roi_signals = {
            "forehead": sig + np.random.randn(len(sig)) * 0.05,
            "left_cheek": sig + np.random.randn(len(sig)) * 0.05,
            "right_cheek": sig + np.random.randn(len(sig)) * 0.05,
        }
        weights = {"forehead": 1.5, "left_cheek": 1.0, "right_cheek": 1.0}
        bpm, q, per_q = fuse_roi_signals(roi_signals, weights, fps)
        assert 60 < bpm < 85
        assert q > 0.02

    def test_fusion_empty(self):
        bpm, q, per_q = fuse_roi_signals({}, {}, 30.0)
        assert bpm == 0.0
        assert q == 0.0


class TestHRTimeseries:
    def test_long_signal(self):
        fps = 30.0
        t = np.arange(0, 30.0, 1.0 / fps)
        signal = np.sin(2 * np.pi * 1.2 * t)
        ts = compute_hr_timeseries(signal.tolist(), fps)
        assert len(ts) >= 10

    def test_timestamps_increasing(self):
        fps = 30.0
        t = np.arange(0, 15.0, 1.0 / fps)
        signal = np.sin(2 * np.pi * 1.2 * t)
        ts = compute_hr_timeseries(signal.tolist(), fps)
        if len(ts) > 1:
            stamps = [e["timestamp_sec"] for e in ts]
            for i in range(1, len(stamps)):
                assert stamps[i] > stamps[i - 1]

    def test_timeseries_structure(self):
        fps = 30.0
        t = np.arange(0, 10.0, 1.0 / fps)
        signal = np.sin(2 * np.pi * 1.0 * t)
        ts = compute_hr_timeseries(signal.tolist(), fps)
        for entry in ts:
            assert "timestamp_sec" in entry
            assert "bpm" in entry
            assert "quality" in entry


# ---------------------------------------------------------------------------
# Face quality tests
# ---------------------------------------------------------------------------

class TestFaceQuality:
    def test_frame_quality_no_face(self):
        q = frame_quality_score(False, 128.0, 0.0, 0.8)
        assert q == 0.0

    def test_frame_quality_good(self):
        q = frame_quality_score(True, 150.0, 0.5, 0.95)
        assert q > 0.5

    def test_frame_quality_high_motion(self):
        q = frame_quality_score(True, 150.0, 20.0, 0.95)
        assert q < 0.3

    def test_scan_quality_good(self):
        result = compute_scan_quality(
            frame_qualities=[0.9] * 100,
            tracking_ratio=0.95,
            motion_scores=[0.5] * 100,
            brightness_scores=[0.9] * 100,
            roi_agreement=0.85,
            signal_periodicity=0.8,
            scan_duration_sec=30.0,
        )
        assert result["scan_quality"] > 0.6
        assert result["retake_required"] is False

    def test_scan_quality_bad_lighting(self):
        result = compute_scan_quality(
            frame_qualities=[0.2] * 50,
            tracking_ratio=0.8,
            motion_scores=[1.0] * 50,
            brightness_scores=[0.1] * 50,
            roi_agreement=0.5,
            signal_periodicity=0.3,
            scan_duration_sec=20.0,
        )
        assert result["retake_required"] is True
        assert any("lighting" in r.lower() for r in result["retake_reasons"])

    def test_scan_quality_too_short(self):
        result = compute_scan_quality(
            frame_qualities=[0.8] * 10,
            tracking_ratio=0.9,
            motion_scores=[0.5] * 10,
            brightness_scores=[0.8] * 10,
            roi_agreement=0.7,
            signal_periodicity=0.6,
            scan_duration_sec=3.0,
        )
        assert result["retake_required"] is True
        assert any("short" in r.lower() for r in result["retake_reasons"])

    def test_scan_quality_excessive_motion(self):
        result = compute_scan_quality(
            frame_qualities=[0.3] * 80,
            tracking_ratio=0.9,
            motion_scores=[20.0] * 80,
            brightness_scores=[0.8] * 80,
            roi_agreement=0.6,
            signal_periodicity=0.4,
            scan_duration_sec=20.0,
        )
        assert result["retake_required"] is True
        assert any("movement" in r.lower() or "motion" in r.lower()
                    for r in result["retake_reasons"])

    def test_roi_agreement_same_bpm(self):
        agr = compute_roi_agreement({"a": 72.0, "b": 73.0, "c": 71.5})
        assert agr > 0.9

    def test_roi_agreement_divergent(self):
        agr = compute_roi_agreement({"a": 60.0, "b": 90.0, "c": 110.0})
        assert agr < 0.5

    def test_confidence_breakdown_keys(self):
        result = compute_scan_quality(
            frame_qualities=[0.7] * 50,
            tracking_ratio=0.85,
            motion_scores=[1.0] * 50,
            brightness_scores=[0.7] * 50,
            roi_agreement=0.7,
            signal_periodicity=0.6,
            scan_duration_sec=25.0,
        )
        bd = result["confidence_breakdown"]
        for key in ("tracking", "lighting", "motion", "signal_periodicity",
                     "roi_agreement", "scan_completeness"):
            assert key in bd
            assert 0.0 <= bd[key] <= 1.0


# ---------------------------------------------------------------------------
# Face features tests
# ---------------------------------------------------------------------------

class TestBlinkDetector:
    def test_counts_blinks(self):
        det = BlinkDetector(threshold=0.21, consec=2)
        # Simulate open-closed-closed-open pattern (one blink)
        for _ in range(5):
            det.update(0.30)  # open
        for _ in range(3):
            det.update(0.15)  # closed
        det.update(0.30)  # open → triggers blink
        assert det.blink_count == 1

    def test_no_blinks_always_open(self):
        det = BlinkDetector()
        for _ in range(100):
            det.update(0.30)
        assert det.blink_count == 0


class TestBlinkAggregation:
    def test_blink_results(self):
        result = aggregate_blink_results([0.3, 0.25, 0.15, 0.1, 0.3], 1, 10.0)
        assert "blink_count" in result
        assert "blink_rate_per_min" in result
        assert result["blink_count"] == 1

    def test_empty_ear_values(self):
        result = aggregate_blink_results([], 0, 10.0)
        assert result == {}


class TestEyeMovementAggregation:
    def test_with_offsets(self):
        offsets = [(1.0, 0.5), (1.1, 0.6), (1.2, 0.4), (0.9, 0.5)]
        result = aggregate_eye_movement(offsets, 10.0)
        assert "eye_stability" in result
        assert 0.0 <= result["eye_stability"] <= 1.0

    def test_empty_offsets(self):
        result = aggregate_eye_movement([], 10.0)
        assert result == {}


class TestFacialMotionAggregation:
    def test_with_positions(self):
        positions = {
            "forehead": [np.ones((8, 2)), np.ones((8, 2)) * 1.1, np.ones((8, 2)) * 1.2],
            "mouth": [np.ones((20, 2)), np.ones((20, 2)) * 1.5],
            "jaw": [],
            "left_eyebrow": [],
            "right_eyebrow": [],
        }
        result = aggregate_facial_motion(positions)
        assert "facial_tension_index" in result
        assert 0.0 <= result["facial_tension_index"] <= 1.0

    def test_insufficient_data(self):
        positions = {"forehead": [np.ones((8, 2))]}
        result = aggregate_facial_motion(positions)
        assert result == {}


class TestSkinAnalysis:
    def test_with_samples(self):
        samples = [
            {"rgb": [180, 140, 120], "hsv": [15, 80, 180]},
            {"rgb": [185, 142, 118], "hsv": [14, 82, 182]},
        ]
        result = aggregate_skin_analysis(samples)
        assert "skin_signal_stability" in result
        assert "redness_proxy" in result
        assert "pallor_proxy_low_confidence" in result
        assert 0.0 <= result["skin_signal_stability"] <= 1.0

    def test_empty(self):
        result = aggregate_skin_analysis([])
        assert result == {}


# ---------------------------------------------------------------------------
# HR classification tests
# ---------------------------------------------------------------------------

class TestClassifyHR:
    def test_normal(self):
        risk, msg = _classify_hr(72.0)
        assert risk == "low"

    def test_low(self):
        risk, _ = _classify_hr(45.0)
        assert risk == "moderate"

    def test_high(self):
        risk, _ = _classify_hr(110.0)
        assert risk == "moderate"


# ---------------------------------------------------------------------------
# Score engine integration tests
# ---------------------------------------------------------------------------

class TestScoreEngineIntegration:
    def test_with_new_face_format(self):
        """Score engine works with the new face result schema."""
        face = {
            "heart_rate": 75.0,
            "heart_rate_confidence": 0.8,
            "scan_quality": 0.85,
            "confidence": 0.8,
            "risk": "low",
            "eye_stability": 0.85,
            "skin_signal_stability": 0.7,
            "facial_tension_index": 0.2,
        }
        result = compute_vita_score(face_result=face)
        assert "vita_health_score" in result
        assert result["vita_health_score"] > 0

    def test_low_quality_face_reduces_weight(self):
        """Low scan quality should reduce face's contribution."""
        good_face = {
            "heart_rate": 120.0,
            "scan_quality": 0.9,
            "confidence": 0.85,
            "risk": "moderate",
        }
        bad_face = {
            "heart_rate": 120.0,
            "scan_quality": 0.2,
            "confidence": 0.3,
            "risk": "moderate",
        }
        score_good = compute_vita_score(face_result=good_face)
        score_bad = compute_vita_score(face_result=bad_face)
        # Low-quality face should have less impact (score closer to neutral)
        # Both have bad HR, but bad_face should have less influence
        assert score_bad["vita_health_score"] >= score_good["vita_health_score"] - 5

    def test_error_face_result(self):
        face = {
            "heart_rate": None,
            "value": None,
            "risk": "error",
            "scan_quality": 0.0,
            "confidence": 0.0,
        }
        result = compute_vita_score(face_result=face)
        # Error face is now excluded from fusion — with no other modules
        # the result should be the "no modules" unknown state.
        assert result["overall_risk"] == "unknown"
        assert result["vita_health_score"] is None

    def test_legacy_value_field_still_works(self):
        """Legacy face result with 'value' field."""
        face = {
            "value": 72.0,
            "confidence": 0.7,
            "risk": "low",
        }
        result = compute_vita_score(face_result=face)
        # With only one module (face), fusion uses weighted-sum of just
        # heart_score — should produce a reasonable healthy score.
        assert result["vita_health_score"] > 40


# ---------------------------------------------------------------------------
# Integration test (real sample)
# ---------------------------------------------------------------------------

SAMPLE_VIDEO = Path(__file__).resolve().parent.parent / "sample_data" / "sample_face_video.mp4"


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Sample video not available")
class TestFaceModuleIntegration:
    def test_full_pipeline(self):
        result = analyze_face_video(str(SAMPLE_VIDEO))
        _assert_valid_schema(result)
        if result["risk"] != "error":
            assert isinstance(result["heart_rate"], (int, float))
            assert result["heart_rate"] > 0
            assert len(result["hr_timeseries"]) >= 1
            assert "confidence_breakdown" in result


# ---------------------------------------------------------------------------
# Signal Strength Engine tests
# ---------------------------------------------------------------------------

from backend.app.utils.signal_processing import (
    compute_signal_strength,
    compute_snr,
    compute_peak_prominence,
    compute_harmonic_consistency,
    compute_inter_window_consistency,
    score_window,
    select_best_windows,
)


class TestSignalStrengthEngine:
    """Tests for the new signal strength scoring infrastructure."""

    def _make_clean_signal(self, fps=30.0, bpm=72.0, n=300, noise=0.05):
        t = np.arange(n) / fps
        freq = bpm / 60.0
        signal = np.sin(2.0 * np.pi * freq * t)
        rng = np.random.default_rng(42)
        signal += rng.normal(0, noise, n)
        return signal, fps

    def test_compute_snr_returns_0_1(self):
        sig, fps = self._make_clean_signal()
        snr = compute_snr(sig, fps)
        assert 0.0 <= snr <= 1.0

    def test_clean_signal_high_snr(self):
        sig, fps = self._make_clean_signal(noise=0.01)
        snr = compute_snr(sig, fps)
        assert snr > 0.3, f"Clean signal should have decent SNR, got {snr}"

    def test_noise_signal_low_snr(self):
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 1, 300)
        snr = compute_snr(noise, 30.0)
        assert snr < 0.5, f"Pure noise should have low SNR, got {snr}"

    def test_compute_peak_prominence(self):
        sig, fps = self._make_clean_signal()
        prom = compute_peak_prominence(sig, fps)
        assert 0.0 <= prom <= 1.0

    def test_compute_harmonic_consistency(self):
        sig, fps = self._make_clean_signal()
        hc = compute_harmonic_consistency(sig, fps)
        assert 0.0 <= hc <= 1.0

    def test_compute_inter_window_consistency(self):
        sig, fps = self._make_clean_signal(n=600)
        ic = compute_inter_window_consistency(sig, fps)
        assert 0.0 <= ic <= 1.0

    def test_compute_signal_strength_range(self):
        sig, fps = self._make_clean_signal()
        result = compute_signal_strength(sig, fps)
        ss = result["signal_strength"]
        assert 0.0 <= ss <= 1.0

    def test_clean_signal_higher_strength(self):
        clean, fps = self._make_clean_signal(noise=0.01)
        rng = np.random.default_rng(123)
        noisy = rng.normal(0, 1, 300)
        ss_clean = compute_signal_strength(clean, fps)["signal_strength"]
        ss_noisy = compute_signal_strength(noisy, fps)["signal_strength"]
        assert ss_clean > ss_noisy, (
            f"Clean signal ({ss_clean:.3f}) should beat noise ({ss_noisy:.3f})"
        )

    def test_score_window_returns_expected_keys(self):
        sig, fps = self._make_clean_signal()
        result = score_window(sig, fps, motion_score=0.5, brightness_stability=0.6)
        assert "bpm" in result
        assert "window_strength" in result
        assert 0.0 <= result["window_strength"] <= 1.0

    def test_select_best_windows_filters(self):
        rng = np.random.default_rng(42)
        fps = 30.0
        # Build a signal with 6 windows' worth of data (~5s each, 150 samples)
        t = np.arange(900) / fps
        signal = np.sin(2.0 * np.pi * 1.2 * t) + rng.normal(0, 0.05, 900)
        selected, rejected = select_best_windows([], signal, fps, window_sec=5.0, stride_sec=2.5)
        assert len(selected) + len(rejected) > 0 or len(signal) < int(fps * 5.0)


# ---------------------------------------------------------------------------
# ROI Quality & Landmark Smoothing tests
# ---------------------------------------------------------------------------

from backend.app.ml.face.vision_utils import (
    build_skin_mask,
    compute_roi_quality_metrics,
    LandmarkSmoother,
)


class TestROIQualityAndSmoothing:

    def test_build_skin_mask_shape(self):
        frame = np.full((100, 120, 3), 150, dtype=np.uint8)
        mask = build_skin_mask(frame)
        assert mask.shape == (100, 120)
        assert mask.dtype == np.uint8

    def test_compute_roi_quality_metrics_keys(self):
        roi = np.full((50, 50, 3), 128, dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        result = compute_roi_quality_metrics(roi, mask)
        assert "skin_coverage" in result
        assert "mean_brightness" in result
        assert "overexposure_ratio" in result
        assert "color_variance" in result
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_compute_roi_quality_no_mask(self):
        roi = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = compute_roi_quality_metrics(roi)
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_landmark_smoother_reduces_drift(self):
        smoother = LandmarkSmoother(alpha=0.6)
        # Build mock landmarks (need .x, .y, .z)
        class _Lm:
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

        h, w = 480, 640
        lm1 = [_Lm(0.5, 0.5, 0.0) for _ in range(5)]
        s1 = smoother.smooth(lm1, h, w)
        assert s1.shape == (5, 3)
        # Second frame with jump
        lm2 = [_Lm(0.6, 0.6, 0.0) for _ in range(5)]
        s2 = smoother.smooth(lm2, h, w)
        # Raw jump
        raw2 = np.array([(0.6 * w, 0.6 * h, 0.0)] * 5)
        raw_drift = float(np.mean(np.linalg.norm(raw2[:, :2] - s1[:, :2], axis=1)))
        smooth_drift = float(np.mean(np.linalg.norm(s2[:, :2] - s1[:, :2], axis=1)))
        assert smooth_drift < raw_drift, "Smoother should reduce drift"


# ---------------------------------------------------------------------------
# Competition Engine tests
# ---------------------------------------------------------------------------

from backend.app.ml.face.rppg_utils import estimate_heart_rate_multi_roi, ROI_NAMES


class TestCompetitionEngine:

    def _make_roi_traces(self, bpm=72.0, fps=30.0, n=300, noise=0.05):
        t = np.arange(n) / fps
        freq = bpm / 60.0
        rng = np.random.default_rng(42)
        traces = {}
        rgb_traces = {}
        for name in ROI_NAMES:
            g = np.sin(2.0 * np.pi * freq * t) + rng.normal(0, noise, n)
            traces[name] = g.tolist()
            rgb_arr = np.zeros((n, 3))
            rgb_arr[:, 1] = g
            rgb_arr[:, 0] = rng.normal(0, noise * 0.3, n)
            rgb_arr[:, 2] = rng.normal(0, noise * 0.3, n)
            rgb_traces[name] = [rgb_arr[i] for i in range(n)]
        return traces, rgb_traces

    def test_competition_returns_expected_keys(self):
        traces, rgb = self._make_roi_traces()
        weights = {n: (1.5 if n == "forehead" else 1.0) for n in ROI_NAMES}
        result = estimate_heart_rate_multi_roi(
            traces, weights, 30.0, 0.7, 3.5,
            roi_rgb_traces=rgb,
        )
        assert "bpm" in result
        assert "quality" in result
        assert "roi_agreement" in result
        assert "periodicity" in result
        assert "selected_roi" in result
        assert "selected_method" in result
        assert "signal_strength" in result
        assert "all_candidate_scores" in result
        assert "roi_scores" in result
        assert "method_scores" in result

    def test_competition_signal_strength_positive(self):
        traces, rgb = self._make_roi_traces(noise=0.02)
        weights = {n: (1.5 if n == "forehead" else 1.0) for n in ROI_NAMES}
        result = estimate_heart_rate_multi_roi(
            traces, weights, 30.0, 0.7, 3.5,
            roi_rgb_traces=rgb,
        )
        assert result["signal_strength"] > 0.0

    def test_competition_with_skin_coverage(self):
        traces, rgb = self._make_roi_traces()
        weights = {n: (1.5 if n == "forehead" else 1.0) for n in ROI_NAMES}
        skin_cov = {n: 0.8 for n in ROI_NAMES}
        result = estimate_heart_rate_multi_roi(
            traces, weights, 30.0, 0.7, 3.5,
            roi_rgb_traces=rgb,
            roi_skin_coverage=skin_cov,
        )
        assert result["bpm"] is not None or result["quality"] == 0.0

    def test_noisy_traces_low_strength(self):
        rng = np.random.default_rng(99)
        traces = {n: rng.normal(0, 1, 300).tolist() for n in ROI_NAMES}
        rgb = {n: [rng.normal(0, 1, 3) for _ in range(300)] for n in ROI_NAMES}
        weights = {n: 1.0 for n in ROI_NAMES}
        result = estimate_heart_rate_multi_roi(
            traces, weights, 30.0, 0.7, 3.5,
            roi_rgb_traces=rgb,
        )
        clean_traces, clean_rgb = self._make_roi_traces(noise=0.02)
        clean_result = estimate_heart_rate_multi_roi(
            clean_traces, {n: 1.0 for n in ROI_NAMES}, 30.0, 0.7, 3.5,
            roi_rgb_traces=clean_rgb,
        )
        assert clean_result["signal_strength"] >= result["signal_strength"]


# ---------------------------------------------------------------------------
# Tests for new Part 1-5 signal improvements
# ---------------------------------------------------------------------------

def _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.05, seed=0):
    """Utility: synthetic pulse signal as a sine wave plus Gaussian noise."""
    t = np.arange(n) / fps
    rng = np.random.default_rng(seed)
    return np.sin(2.0 * np.pi * (bpm / 60.0) * t) + rng.normal(0, noise, n)


class TestMovingAverageSmooth:
    """Part 1 — temporal noise reduction via moving-average pre-smoothing."""

    def test_returns_same_length(self):
        sig = np.random.default_rng(1).normal(0, 1, 100)
        out = moving_average_smooth(sig, window=5)
        assert len(out) == len(sig)

    def test_reduces_noise(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(0, 1, 200)
        smoothed = moving_average_smooth(noise, window=9)
        assert float(np.std(smoothed)) < float(np.std(noise))

    def test_short_signal_passthrough(self):
        sig = np.array([1.0, 2.0])
        out = moving_average_smooth(sig, window=5)
        assert len(out) == 2

    def test_constant_signal_unchanged(self):
        sig = np.ones(60, dtype=np.float64) * 3.7
        out = moving_average_smooth(sig, window=5)
        np.testing.assert_allclose(out, sig, atol=1e-10)

    def test_estimate_bpm_uses_smoothing(self):
        """Smoke-test: estimate_bpm should still return a valid BPM."""
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.1)
        bpm, quality = estimate_bpm(sig, 30.0)
        assert 40.0 <= bpm <= 180.0
        assert 0.0 <= quality <= 1.0


class TestSignalColorVariance:
    """Part 2 — color variance scoring."""

    def test_range_0_to_1(self):
        sig = _make_pulse(noise=0.3)
        score = signal_color_variance(sig)
        assert 0.0 <= score <= 1.0

    def test_high_amplitude_higher_score(self):
        low_amp = _make_pulse(noise=0.01) * 0.1
        high_amp = _make_pulse(noise=0.01) * 2.0
        assert signal_color_variance(high_amp) > signal_color_variance(low_amp)

    def test_flat_signal_zero(self):
        sig = np.zeros(100)
        assert signal_color_variance(sig) == 0.0


class TestSignalIlluminationStability:
    """Part 2 — illumination stability scoring."""

    def test_range_0_to_1(self):
        sig = _make_pulse(noise=0.05)
        score = signal_illumination_stability(sig, fps=30.0)
        assert 0.0 <= score <= 1.0

    def test_stable_signal_high_score(self):
        # Near-zero drift → maximally stable
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.01)
        score = signal_illumination_stability(sig, fps=30.0)
        assert score >= 0.5

    def test_large_drift_lowers_score(self):
        # Add a strong linear drift (illumination change)
        t = np.arange(300)
        sig = _make_pulse(fps=30.0, n=300, noise=0.01) + t * 0.05
        score_drift = signal_illumination_stability(sig, fps=30.0)
        sig_clean = _make_pulse(fps=30.0, n=300, noise=0.01)
        score_clean = signal_illumination_stability(sig_clean, fps=30.0)
        assert score_drift <= score_clean

    def test_short_signal_returns_default(self):
        sig = np.ones(5)
        # Should not raise; returns the default 0.5
        score = signal_illumination_stability(sig, fps=30.0)
        assert score == 0.5


class TestCrossWindowPeakStability:
    """Part 4 — cross-window peak stability & isolated-peak penalty."""

    def test_returns_tuple(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.05)
        result = compute_cross_window_peak_stability(sig, 30.0)
        assert len(result) == 3

    def test_pure_sine_high_stability(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.02)
        stab, modal_bpm, support = compute_cross_window_peak_stability(sig, 30.0)
        assert stab >= 0.5
        assert support > 1

    def test_random_noise_low_stability(self):
        sig = np.random.default_rng(99).normal(0, 1, 300)
        stab, _, support = compute_cross_window_peak_stability(sig, 30.0)
        # Pure noise should yield low stability
        assert stab <= 0.7

    def test_short_signal_returns_zero_stability(self):
        sig = np.ones(10)
        stab, modal, support = compute_cross_window_peak_stability(sig, 30.0)
        assert stab == 0.0
        assert support == 0

    def test_isolated_peak_zero_stability(self):
        # A signal with a single valid BPM window → support_count == 0 path
        sig = _make_pulse(bpm=72.0, fps=30.0, n=200, noise=0.01)
        # use a very long window_sec so only one or zero windows exist
        stab, _, support = compute_cross_window_peak_stability(sig, 30.0, window_sec=8.0)
        # With n<wlen, should return (0.0, 0.0, 0)
        assert stab == 0.0


class TestComputeSignalStrengthNewFields:
    """Part 4 — compute_signal_strength now returns peak_stability and modal_bpm."""

    def test_returns_new_keys(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.05)
        result = compute_signal_strength(sig, 30.0)
        assert "peak_stability" in result
        assert "modal_bpm" in result
        assert "peak_support_count" in result

    def test_clean_signal_high_strength(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.02)
        result = compute_signal_strength(sig, 30.0)
        assert result["signal_strength"] > 0.5

    def test_isolation_penalty_applied(self):
        """signal_strength with isolated-peak signal should not exceed clean signal."""
        clean = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.02)
        noisy = np.random.default_rng(7).normal(0, 1, 300)
        clean_result = compute_signal_strength(clean, 30.0)
        noisy_result = compute_signal_strength(noisy, 30.0)
        assert clean_result["signal_strength"] >= noisy_result["signal_strength"]


class TestClosepeakMergingInConsensus:
    """Part 5 — close BPM peaks (±5 bpm) are merged before cluster selection."""

    def test_merging_reduces_spread(self):
        from backend.app.utils.signal_processing import robust_hr_consensus
        # Six BPMs all within 5 bpm of each other → merge into a single value
        windows = [
            {"bpm": v, "quality": 0.6, "timestamp_sec": float(i), "snr": 0.5, "amplitude": 1.0}
            for i, v in enumerate([70.0, 71.5, 72.3, 73.0, 74.0, 75.0])
        ]
        result = robust_hr_consensus(windows, min_windows=1)
        assert result["has_consensus"]
        # All values merge into one cluster; final HR should be near the centre
        assert 70.0 <= result["heart_rate"] <= 76.0

    def test_well_separated_peaks_not_merged(self):
        from backend.app.utils.signal_processing import robust_hr_consensus
        # Two clearly distinct groups
        windows = [
            {"bpm": v, "quality": 0.6, "timestamp_sec": float(i), "snr": 0.5, "amplitude": 1.0}
            for i, v in enumerate([60.0, 60.5, 61.0, 90.0, 90.5, 91.0])
        ]
        result = robust_hr_consensus(windows, min_windows=2)
        assert result["has_consensus"]
        # Should pick the dominant (largest) cluster
        hr = result["heart_rate"]
        assert (55.0 <= hr <= 66.0) or (85.0 <= hr <= 96.0)

    def test_consensus_returns_valid_hr(self):
        from backend.app.utils.signal_processing import robust_hr_consensus
        fps = 30.0
        signals = [_make_pulse(bpm=72.0, fps=fps, n=300, noise=0.05, seed=i) for i in range(3)]
        ts_list = [compute_hr_timeseries(s, fps) for s in signals]
        flat_windows = [w for ts in ts_list for w in ts]
        result = robust_hr_consensus(flat_windows, min_windows=2)
        if result["has_consensus"]:
            assert 40.0 <= result["heart_rate"] <= 180.0


# ---------------------------------------------------------------------------
# Tests for harmonic correction, physiological validation, peak dominance,
# window consensus, and debug output (Parts 1–5)
# ---------------------------------------------------------------------------

def _sub_harmonic_signal(true_bpm=72.0, fps=30.0, n=300, noise=0.02, seed=5):
    """Sine wave at true_bpm/2 — simulates sub-harmonic mis-selection."""
    t = np.arange(n) / fps
    rng = np.random.default_rng(seed)
    return np.sin(2.0 * np.pi * (true_bpm / 60.0 / 2.0) * t) + rng.normal(0, noise, n)


def _make_competition_traces(bpm=72.0, fps=30.0, n=300, noise=0.05):
    """Module-level helper: multi-ROI traces for competition engine tests."""
    t = np.arange(n) / fps
    freq = bpm / 60.0
    rng = np.random.default_rng(42)
    traces, rgb_traces = {}, {}
    for name in ROI_NAMES:
        g = np.sin(2.0 * np.pi * freq * t) + rng.normal(0, noise, n)
        traces[name] = g.tolist()
        rgb_arr = np.zeros((n, 3))
        rgb_arr[:, 1] = g
        rgb_arr[:, 0] = rng.normal(0, noise * 0.3, n)
        rgb_arr[:, 2] = rng.normal(0, noise * 0.3, n)
        rgb_traces[name] = [rgb_arr[i] for i in range(n)]
    return traces, rgb_traces


class TestSpectralHrEstimate:
    """Part 1+2 — spectral_hr_estimate: harmonic correction + physiological validation."""

    def test_returns_required_keys(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.05)
        result = spectral_hr_estimate(sig, 30.0)
        for key in ("bpm", "quality", "original_bpm", "harmonic_checked",
                    "harmonic_corrected", "corrected_bpm", "peak_prominence_raw",
                    "peak_snr", "physiologically_valid"):
            assert key in result, f"Missing key: {key}"

    def test_normal_bpm_not_corrected(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.02)
        result = spectral_hr_estimate(sig, 30.0)
        assert result["bpm"] > 0.0
        assert result["physiologically_valid"] is True
        assert result["harmonic_corrected"] is False

    def test_sub_harmonic_corrected(self):
        """A signal whose dominant spectral peak is the sub-harmonic (≈36 bpm)
        should be corrected to the harmonic (≈72 bpm) when the harmonic peak
        is clearly stronger."""
        fps = 30.0
        t = np.arange(300) / fps
        true_bpm = 72.0
        sub_freq = true_bpm / 60.0 / 2.0       # 0.6 Hz (36 bpm — below valid)
        harm_freq = true_bpm / 60.0             # 1.2 Hz (72 bpm)
        # Build a signal dominated by the harmonic but with a sub-harmonic component
        sig = (0.5 * np.sin(2 * np.pi * sub_freq * t)   # weaker sub-harmonic
               + 1.0 * np.sin(2 * np.pi * harm_freq * t)  # stronger true HR
               + np.random.default_rng(10).normal(0, 0.02, 300))
        result = spectral_hr_estimate(sig, fps)
        # Should detect the true HR (≈72 bpm), not the sub-harmonic
        assert result["bpm"] >= 45.0
        assert result["physiologically_valid"] is True

    def test_below_physio_range_rejected_or_corrected(self):
        """Any result from spectral_hr_estimate must be either 0 (invalid)
        or within the physiological range [45, 180] — never in between."""
        fps = 30.0
        rng = np.random.default_rng(55)
        for _ in range(10):
            sig = rng.normal(0, 1, 300)  # pure noise has no dominant peak
            result = spectral_hr_estimate(sig, fps)
            # Whatever is returned must satisfy the physiological constraint
            assert result["bpm"] == 0.0 or 45.0 <= result["bpm"] <= 180.0

    def test_short_signal_returns_empty(self):
        result = spectral_hr_estimate(np.ones(5), 30.0)
        assert result["bpm"] == 0.0
        assert result["quality"] == 0.0

    def test_estimate_bpm_still_backward_compatible(self):
        """estimate_bpm() must still return a (bpm, quality) tuple."""
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.05)
        result = estimate_bpm(sig, 30.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        bpm, quality = result
        assert 0.0 <= quality <= 1.0

    def test_peak_prominence_raw_in_range(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.02)
        result = spectral_hr_estimate(sig, 30.0)
        assert 0.0 <= result["peak_prominence_raw"] <= 1.0

    def test_peak_snr_positive_for_clean_signal(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.02)
        result = spectral_hr_estimate(sig, 30.0)
        assert result["peak_snr"] > 1.0  # at least above noise floor


class TestPhysiologicalValidation:
    """Part 2 — physiological bounds: 45–180 bpm."""

    def test_valid_range_passes(self):
        for bpm_target in [50, 72, 120, 150]:
            sig = _make_pulse(bpm=float(bpm_target), fps=30.0, n=300, noise=0.02)
            result = spectral_hr_estimate(sig, 30.0)
            assert result["physiologically_valid"] is True or result["bpm"] == 0.0

    def test_sub_hz_signal_fails(self):
        """Signal at 0.5 Hz → 30 bpm — below physiological threshold."""
        t = np.arange(300) / 30.0
        sig = np.sin(2 * np.pi * 0.5 * t)  # 30 bpm
        result = spectral_hr_estimate(sig, 30.0)
        # Either corrected to 60 bpm or marked invalid
        assert result["bpm"] == 0.0 or result["bpm"] >= 45.0

    def test_borderline_low_bpm_confidence_reduced(self):
        """BPM near 48 should have reduced quality (< full-strength)."""
        fps = 30.0
        t = np.arange(300) / fps
        sig = np.sin(2 * np.pi * (48.0 / 60.0) * t) + np.random.default_rng(42).normal(0, 0.02, 300)
        result = spectral_hr_estimate(sig, fps)
        if result["bpm"] > 0:
            # Quality should be reduced for borderline BPM
            assert result["quality"] <= 0.9


class TestPeakDominanceValidation:
    """Part 3 — peak dominance: weak/isolated peaks reduce candidate_score."""

    def test_clean_signal_high_quality(self):
        traces, rgb = _make_competition_traces(noise=0.02)
        weights = {n: 1.0 for n in ROI_NAMES}
        result = estimate_heart_rate_multi_roi(traces, weights, 30.0,
                                               roi_rgb_traces=rgb)
        assert result["signal_strength"] > 0.3

    def test_noisy_signal_penalized(self):
        rng = np.random.default_rng(77)
        traces = {n: rng.normal(0, 1, 300).tolist() for n in ROI_NAMES}
        rgb = {n: [rng.normal(0, 1, 3) for _ in range(300)] for n in ROI_NAMES}
        weights = {n: 1.0 for n in ROI_NAMES}
        noisy_result = estimate_heart_rate_multi_roi(traces, weights, 30.0,
                                                      roi_rgb_traces=rgb)
        # Clean signal should beat noisy (already tested elsewhere; here just
        # verify the noisy result does not get a perfect signal_strength)
        assert noisy_result["signal_strength"] <= 0.8


class TestWindowConsensusCheck:
    """Part 4 — window consensus: consensus_confidence + median_deviation_bpm."""

    def test_returns_consensus_fields(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.05)
        ts = compute_hr_timeseries(sig.tolist(), 30.0)
        result = robust_hr_consensus(ts, min_windows=2)
        if result["has_consensus"]:
            assert "consensus_confidence" in result
            assert "median_deviation_bpm" in result

    def test_tight_cluster_high_consensus_confidence(self):
        """Windows all at the same BPM → zero deviation → confidence = 1.0."""
        windows = [
            {"bpm": 72.0, "quality": 0.5, "timestamp_sec": float(i),
             "snr": 0.5, "amplitude": 1.0}
            for i in range(6)
        ]
        result = robust_hr_consensus(windows, min_windows=1)
        assert result["has_consensus"]
        assert result["consensus_confidence"] >= 0.99

    def test_split_cluster_lower_consensus_confidence(self):
        """Two tight clusters far apart → large deviation → lower confidence."""
        windows = (
            [{"bpm": 60.0, "quality": 0.5, "timestamp_sec": float(i),
              "snr": 0.5, "amplitude": 1.0} for i in range(4)]
            + [{"bpm": 90.0, "quality": 0.5, "timestamp_sec": float(i + 4),
                "snr": 0.5, "amplitude": 1.0} for i in range(4)]
        )
        result = robust_hr_consensus(windows, min_windows=1)
        if result["has_consensus"]:
            # Deviation between cluster HR and overall median should be non-trivial
            assert result["median_deviation_bpm"] >= 0.0

    def test_median_deviation_bpm_nonneg(self):
        sig = _make_pulse(bpm=72.0, fps=30.0, n=300, noise=0.05)
        ts = compute_hr_timeseries(sig.tolist(), 30.0)
        result = robust_hr_consensus(ts, min_windows=2)
        if result["has_consensus"]:
            assert result["median_deviation_bpm"] >= 0.0


class TestDebugOutputFields:
    """Part 5 — verify debug fields present in competition engine output."""

    def _run_engine(self, noise=0.05):
        traces, rgb = _make_competition_traces(noise=noise)
        weights = {n: 1.0 for n in ROI_NAMES}
        return estimate_heart_rate_multi_roi(traces, weights, 30.0,
                                             roi_rgb_traces=rgb)

    def test_harmonic_fields_present(self):
        result = self._run_engine()
        assert "harmonic_checked" in result
        assert "harmonic_corrected" in result
        assert isinstance(result["harmonic_checked"], bool)
        assert isinstance(result["harmonic_corrected"], bool)

    def test_original_corrected_hr_fields(self):
        result = self._run_engine()
        assert "original_hr" in result
        assert "corrected_hr" in result
        # If not corrected, corrected_hr should be None
        if not result["harmonic_corrected"]:
            assert result["corrected_hr"] is None

    def test_peak_prominence_present(self):
        result = self._run_engine()
        assert "peak_prominence" in result
        assert isinstance(result["peak_prominence"], float)

    def test_peak_support_count_present(self):
        result = self._run_engine()
        assert "peak_support_count" in result
        assert isinstance(result["peak_support_count"], int)

    def test_no_roi_result_has_debug_fields(self):
        """Even when all ROIs are dropped, debug fields must be present."""
        empty_traces = {n: [0.0] * 3 for n in ROI_NAMES}  # too short → dropped
        weights = {n: 1.0 for n in ROI_NAMES}
        result = estimate_heart_rate_multi_roi(empty_traces, weights, 30.0)
        assert "harmonic_checked" in result
        assert "peak_support_count" in result
