"""
Tests for the audio / breathing module.

Tests cover schema validation, error handling, and the breathing-rate
estimation helpers.  When a real audio sample is available, the full
pipeline is also exercised.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


from backend.app.ml.audio.audio_module import (
    _classify_breathing,
    _smooth,
    analyze_audio,
    _fft_breathing_rate,
    _autocorr_breathing_rate,
    _peak_breathing_rate_adaptive,
    _detect_subharmonic,
    _fuse_rate_estimates,
)

# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {"module_name", "metric_name", "value", "unit", "confidence", "risk", "message", "debug"}


def _assert_valid_schema(result: dict) -> None:
    assert isinstance(result, dict)
    for key in REQUIRED_KEYS:
        assert key in result, f"Missing key: {key}"
    assert result["module_name"] == "audio_module"
    assert result["unit"] == "breaths/min"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["risk"] in {"low", "moderate", "high", "error",
                                "normal", "elevated", "very_high", "low_rate", "unreliable"}


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestAudioModuleErrors:
    def test_missing_file(self):
        result = analyze_audio("nonexistent_audio.wav")
        _assert_valid_schema(result)
        assert result["risk"] == "error"

    def test_empty_path(self):
        result = analyze_audio("")
        _assert_valid_schema(result)
        assert result["risk"] == "error"

    def test_none_path(self):
        result = analyze_audio(None)  # type: ignore
        _assert_valid_schema(result)
        assert result["risk"] == "error"


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_classify_normal(self):
        risk, msg = _classify_breathing(16.0)
        assert risk == "normal"
        assert "normal" in msg.lower() or "within" in msg.lower()

    def test_classify_low(self):
        risk, _ = _classify_breathing(8.0)
        assert risk == "low_rate"

    def test_classify_high(self):
        risk, _ = _classify_breathing(28.0)
        assert risk == "elevated"  # 20-30/min is elevated

    def test_classify_zero(self):
        risk, _ = _classify_breathing(0.0)
        assert risk == "error"

    def test_smooth_identity(self):
        signal = np.ones(100)
        smoothed = _smooth(signal, 5)
        assert len(smoothed) == len(signal)
        np.testing.assert_allclose(smoothed[5:-5], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Synthetic audio test
# ---------------------------------------------------------------------------

class TestAudioSynthetic:
    def test_synthetic_wav(self):
        """Create a short synthetic WAV in-memory and run the pipeline."""
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            pytest.skip("soundfile not installed for synthetic WAV creation")

        sr = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Simulate breathing-like amplitude modulation at ~0.27 Hz (16 b/min)
        breathing_freq = 0.27
        envelope = 0.5 * (1 + np.sin(2 * np.pi * breathing_freq * t))
        noise = np.random.randn(len(t)) * 0.01
        audio = envelope * noise

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio.astype(np.float32), sr)
            path = f.name

        result = analyze_audio(path)
        _assert_valid_schema(result)
        # Cleanup
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Integration test (real sample)
# ---------------------------------------------------------------------------

SAMPLE_AUDIO = Path(__file__).resolve().parent.parent / "sample_data" / "sample_breathing.wav"


@pytest.mark.skipif(not SAMPLE_AUDIO.exists(), reason="Sample audio not available")
class TestAudioIntegration:
    def test_full_pipeline(self):
        result = analyze_audio(str(SAMPLE_AUDIO))
        _assert_valid_schema(result)


# ---------------------------------------------------------------------------
# Realism / suppression tests
# ---------------------------------------------------------------------------

class TestAudioRealismGuards:
    """Verify that weak/noisy audio is honestly reported, not
    padded with default values."""

    def test_silence_returns_error_or_none(self):
        """Pure silence should not produce a breathing rate."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        sr = 22050
        silence = np.zeros(int(sr * 5.0))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, silence.astype(np.float32), sr)
            path = f.name
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        # Should return no breathing value
        assert result["value"] is None or result["risk"] in ("error", "unreliable")

    def test_noise_returns_low_confidence(self):
        """Random noise should produce low confidence."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        sr = 22050
        noise = np.random.randn(int(sr * 8.0)) * 0.5
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, noise.astype(np.float32), sr)
            path = f.name
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        if result["value"] is not None:
            assert result["confidence"] < 0.7, (
                f"Random noise got confidence={result['confidence']} (should be low)"
            )

    def test_suppression_reason_in_debug(self):
        """Debug output should include suppression_reason field."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        sr = 22050
        t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)
        # Broadband AM noise at 0.27 Hz: passes pre-norm RMS gate and spectral
        # flatness gate, but only ~2.7 cycles fit in 10 s (< 3 required) so the
        # analyser reaches the suppression_reason path.
        rng = np.random.default_rng(42)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 0.27 * t))
        noise = rng.standard_normal(len(t)) * 0.03
        audio = (envelope * noise).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio.astype(np.float32), sr)
            path = f.name
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert "suppression_reason" in result.get("debug", {})


class TestNoBreathingRejection:
    """Ensure the pipeline refuses to estimate when there is no real breathing.

    These tests simulate the exact scenarios users encounter:
    AGC-amplified ambient noise, steady-state hum, and white noise at realistic
    levels.  Every case must return: no breathing rate, retake_required=True,
    reliability='unreliable'.
    """

    @staticmethod
    def _write_wav(audio: np.ndarray, sr: int) -> str:
        """Write a float32 wav and return the path."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(f.name, audio.astype(np.float32), sr)
        return f.name

    def test_agc_amplified_silence(self):
        """Browser AGC amplifies silence to ~0.03 RMS — must still reject."""
        sr = 22050
        rng = np.random.default_rng(99)
        # Simulate AGC-amplified ambient: low-level broadband noise at ~0.03 RMS
        audio = rng.standard_normal(sr * 15) * 0.03
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"AGC noise got breathing_rate={result['breathing_rate']} — should be None"
        )
        assert result["retake_required"] is True
        assert result["reliability"] == "unreliable"

    def test_agc_amplified_moderate_noise(self):
        """Moderate AGC noise (~0.06 RMS) should still not produce a rate."""
        sr = 22050
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(sr * 20) * 0.06
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"Moderate noise got rate={result['breathing_rate']}"
        )
        assert result["retake_required"] is True

    def test_steady_hum_60hz(self):
        """A 60 Hz hum (fan/HVAC) should not be mistaken for breathing."""
        sr = 22050
        t = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
        audio = np.sin(2 * np.pi * 60.0 * t) * 0.04
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"60 Hz hum produced rate={result['breathing_rate']}"
        )

    def test_low_frequency_rumble(self):
        """Low-freq environmental rumble (faint mechanical vibration) should
        not produce a rate.  Real-world rumble is typically very quiet and
        gets caught by the pre-normalization RMS gate."""
        sr = 22050
        t = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
        # Faint incoherent sum (combined RMS ~ 0.008, below pre-norm gate)
        audio = (
            np.sin(2 * np.pi * 0.9 * t) * 0.005
            + np.sin(2 * np.pi * 1.3 * t + 1.7) * 0.004
            + np.sin(2 * np.pi * 1.8 * t + 3.1) * 0.003
        )
        rng = np.random.default_rng(7)
        audio += rng.standard_normal(len(t)) * 0.004
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"Low-freq rumble produced rate={result['breathing_rate']}"
        )

    def test_real_breathing_still_detected(self):
        """A strong, clean breathing-like signal should still produce a result.

        15 bpm = 0.25 Hz modulation of broadband noise, amplitude 0.4
        with clear periodic peaks.
        """
        sr = 22050
        rng = np.random.default_rng(123)
        t = np.linspace(0, 20.0, int(sr * 20.0), endpoint=False)
        # Simulate breathing: amplitude-modulated noise with strong periodic envelope
        breath_env = 0.5 * (1.0 + np.sin(2 * np.pi * (15.0 / 60.0) * t))
        noise = rng.standard_normal(len(t)) * 0.15
        audio = (breath_env * noise).astype(np.float32)
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is not None, (
            "Clear breathing signal should produce a rate"
        )
        rate = float(result["breathing_rate"])
        assert 8.0 <= rate <= 25.0, f"Expected ~15 bpm, got {rate}"

    def test_no_vita_score_from_invalid_breathing(self):
        """Score engine should return None for breathing when result is invalid."""
        from backend.app.ml.fusion.score_engine import _normalise_breathing
        # Simulate an invalid result from the audio module
        invalid_result = {
            "breathing_rate": None,
            "value": None,
            "risk": "unreliable",
            "reliability": "unreliable",
            "retake_required": True,
            "confidence": 0.0,
        }
        score = _normalise_breathing(invalid_result)
        assert score is None, f"Invalid breathing should not produce a score, got {score}"

    def test_no_vita_score_from_retake_required(self):
        """Score engine should exclude breathing even when rate exists but retake=True."""
        from backend.app.ml.fusion.score_engine import _normalise_breathing
        result_with_retake = {
            "breathing_rate": 40.0,
            "value": 40.0,
            "risk": "high",
            "reliability": "low",
            "retake_required": True,
            "confidence": 0.3,
        }
        score = _normalise_breathing(result_with_retake)
        assert score is None, f"retake_required=True should exclude from scoring, got {score}"


# ---------------------------------------------------------------------------
# Multi-method breathing estimator tests
# ---------------------------------------------------------------------------


class TestBreathingRateMultiMethod:
    """Verify multi-method estimation: FFT, autocorr, peak detection, subharmonic
    correction, and fusion all behave correctly for both fast and slow breathing."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_breathing_audio(
        self, sr: int, duration: float, bpm: float, seed: int = 42
    ) -> np.ndarray:
        """Amplitude-modulated noise at the given breathing rate (breaths/min)."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = bpm / 60.0
        envelope = 0.6 * (1.0 + np.sin(2 * np.pi * freq * t - np.pi / 2))
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(t)) * 0.05
        return (envelope * noise).astype(np.float32)

    def _envelope(self, bpm: float, duration: float = 15.0, env_sr: float = 50.0) -> np.ndarray:
        """Pure sinusoidal envelope at a given breathing rate."""
        t = np.linspace(0, duration, int(env_sr * duration), endpoint=False)
        freq = bpm / 60.0
        return (1.0 + np.sin(2 * np.pi * freq * t)).astype(np.float64)

    # ------------------------------------------------------------------
    # Unit tests — individual helpers
    # ------------------------------------------------------------------

    def test_fft_rate_detects_fast_breathing(self):
        """_fft_breathing_rate should identify 30 bpm from a clean envelope."""
        env = self._envelope(bpm=30.0)
        rate = _fft_breathing_rate(env, env_sr=50.0)
        assert abs(rate - 30.0) < 3.0, f"Expected ~30 bpm, got {rate}"

    def test_fft_rate_detects_slow_breathing(self):
        """_fft_breathing_rate should identify 12 bpm from a clean envelope."""
        env = self._envelope(bpm=12.0)
        rate = _fft_breathing_rate(env, env_sr=50.0)
        assert abs(rate - 12.0) < 3.0, f"Expected ~12 bpm, got {rate}"

    def test_autocorr_rate_detects_fast_breathing(self):
        """_autocorr_breathing_rate should find the dominant period at 24 bpm."""
        env = self._envelope(bpm=24.0)
        rate = _autocorr_breathing_rate(env, env_sr=50.0)
        assert abs(rate - 24.0) < 4.0, f"Expected ~24 bpm, got {rate}"

    def test_peak_adaptive_allows_fast_peaks(self):
        """_peak_breathing_rate_adaptive with min_dist_sec=0.5 handles 30 bpm."""
        env = self._envelope(bpm=30.0)
        peak_rate, peaks, _ = _peak_breathing_rate_adaptive(env, env_sr=50.0, min_dist_sec=0.5)
        assert peak_rate > 0, "Should detect peaks in clean 30-bpm envelope"
        assert abs(peak_rate - 30.0) < 6.0, f"Expected ~30 bpm, got {peak_rate}"

    def test_subharmonic_correction_corrects_half_rate(self):
        """_detect_subharmonic returns rate×2 when FFT/autocorr point to harmonic."""
        corrected, was_corrected = _detect_subharmonic(15.0, other_rates=[29.5, 30.5])
        assert was_corrected, "Expected subharmonic to be corrected"
        assert 28.0 <= corrected <= 32.0, f"Corrected rate {corrected} not near 30 bpm"

    def test_subharmonic_no_correction_when_methods_agree(self):
        """_detect_subharmonic must NOT correct when all methods agree on the rate."""
        corrected, was_corrected = _detect_subharmonic(16.0, other_rates=[15.5, 16.5])
        assert not was_corrected
        assert corrected == 16.0

    def test_fuse_all_agree(self):
        """_fuse_rate_estimates returns high agreement when all three are close."""
        rate, score, count, _ = _fuse_rate_estimates(18.0, 19.0, 18.5)
        assert count == 3
        assert score >= 0.9
        assert 17.0 <= rate <= 20.0

    def test_fuse_two_of_three_agree(self):
        """_fuse_rate_estimates picks the majority cluster, ignoring outlier."""
        rate, score, count, _ = _fuse_rate_estimates(18.0, 18.5, 35.0)
        assert count == 2
        assert 16.0 <= rate <= 21.0, f"Expected ~18, got {rate}"

    def test_fuse_no_valid_rates(self):
        """_fuse_rate_estimates returns zero when no estimate is in valid range."""
        rate, score, count, label = _fuse_rate_estimates(0.0, 0.0, 0.0)
        assert rate == 0.0
        assert count == 0

    # ------------------------------------------------------------------
    # Integration tests — full pipeline
    # ------------------------------------------------------------------

    def test_fast_breathing_rate_higher_than_slow(self):
        """Pipeline must report a higher rate for fast than for slow breathing."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        sr = 22050
        slow_audio = self._make_breathing_audio(sr, 20.0, bpm=12.0)
        fast_audio = self._make_breathing_audio(sr, 20.0, bpm=28.0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_slow:
            sf.write(f_slow.name, slow_audio, sr)
            slow_path = f_slow.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_fast:
            sf.write(f_fast.name, fast_audio, sr)
            fast_path = f_fast.name

        try:
            slow_result = analyze_audio(slow_path)
            fast_result = analyze_audio(fast_path)
        finally:
            Path(slow_path).unlink(missing_ok=True)
            Path(fast_path).unlink(missing_ok=True)

        if slow_result["value"] is None or fast_result["value"] is None:
            pytest.skip("Signal too weak for rate comparison")

        assert fast_result["value"] > slow_result["value"], (
            f"Expected fast ({fast_result['value']}) > slow ({slow_result['value']})"
        )

    def test_oversmoothing_prevention(self):
        """A 30 bpm signal must not be reported as ≤15 bpm (oversmoothing guard)."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        sr = 22050
        audio = self._make_breathing_audio(sr, 20.0, bpm=30.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            path = f.name

        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)

        if result["value"] is None:
            pytest.skip("Signal too weak")
        assert result["value"] > 15.0, (
            f"30 bpm signal reported as {result['value']} bpm (oversmoothing)"
        )

    def test_debug_has_multi_method_fields(self):
        """analyze_audio debug dict must include the new multi-method fields."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        sr = 22050
        audio = self._make_breathing_audio(sr, 15.0, bpm=16.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            path = f.name

        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)

        debug = result.get("debug", {})
        for field in (
            "fft_rate", "autocorr_rate", "peak_rate_raw", "peak_rate_corrected",
            "harmonic_correction_applied", "method_agreement", "detected_peaks",
            "breath_intervals", "smoothing_window_ms",
        ):
            assert field in debug, f"debug dict missing field: {field}"


# ---------------------------------------------------------------------------
# Fan / airflow / mechanical noise rejection tests
# ---------------------------------------------------------------------------

class TestFanNoiseRejection:
    """Ensure the pipeline rejects fan, AC, HVAC and mechanical airflow noise.

    Fan/AC noise is characterised by narrow-band energy (tonal hum) and/or
    mechanically regular amplitude modulation.  None of these should produce
    a breathing rate.
    """

    @staticmethod
    def _write_wav(audio: np.ndarray, sr: int) -> str:
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(f.name, audio.astype(np.float32), sr)
        return f.name

    def test_fan_60hz_pure_hum(self):
        """Pure 60 Hz fan/mains hum → must reject (narrow-band)."""
        sr = 22050
        t = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
        audio = np.sin(2 * np.pi * 60.0 * t) * 0.05
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"Pure 60 Hz hum produced rate={result['breathing_rate']}"
        )
        assert result["risk"] in ("error", "unreliable")

    def test_fan_60hz_with_amplitude_modulation(self):
        """60 Hz hum with 0.5 Hz AM (the main false-positive scenario) → reject."""
        sr = 22050
        t = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
        # Carrier at 60 Hz with 40% AM at 0.5 Hz (= 30 bpm modulation)
        carrier = np.sin(2 * np.pi * 60.0 * t)
        modulation = 1.0 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
        audio = (carrier * modulation) * 0.05
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"Fan AM produced rate={result['breathing_rate']} — should be None"
        )
        assert result["risk"] in ("error", "unreliable")

    def test_ac_hum_with_harmonics(self):
        """AC hum at 60/120/180 Hz (harmonics) → reject as narrow-band."""
        sr = 22050
        t = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
        audio = (
            np.sin(2 * np.pi * 60.0 * t) * 0.04
            + np.sin(2 * np.pi * 120.0 * t) * 0.02
            + np.sin(2 * np.pi * 180.0 * t) * 0.01
        )
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"AC harmonics produced rate={result['breathing_rate']}"
        )

    def test_fan_50hz_european_mains(self):
        """50 Hz mains hum (European AC) with AM → reject."""
        sr = 22050
        t = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
        carrier = np.sin(2 * np.pi * 50.0 * t)
        modulation = 1.0 + 0.3 * np.sin(2 * np.pi * 0.4 * t)
        audio = (carrier * modulation) * 0.06
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"50 Hz mains AM produced rate={result['breathing_rate']}"
        )

    def test_mechanical_low_freq_hum_with_am(self):
        """Low-frequency mechanical hum (e.g. compressor) at 100 Hz with AM → reject."""
        sr = 22050
        t = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
        carrier = np.sin(2 * np.pi * 100.0 * t)
        modulation = 1.0 + 0.35 * np.sin(2 * np.pi * 0.75 * t)  # 45 bpm
        audio = (carrier * modulation) * 0.04
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is None, (
            f"Compressor hum AM produced rate={result['breathing_rate']}"
        )

    def test_speech_still_handled(self):
        """Speech-like content (high ZCR + centroid) should not be breathing."""
        sr = 22050
        rng = np.random.default_rng(77)
        t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)
        # Simulate speech: broadband noise filtered to higher frequencies
        noise = rng.standard_normal(len(t)) * 0.1
        # Add formant-like resonances to raise centroid
        audio = noise * np.sin(2 * np.pi * 800 * t) * 0.2
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        # Should either reject or report low confidence
        if result["breathing_rate"] is not None:
            assert result["confidence"] < 0.5, (
                f"Speech-like got confidence={result['confidence']}"
            )

    def test_real_breathing_not_rejected_by_fan_gate(self):
        """Broadband breathing-like AM noise must NOT be rejected by the fan gate."""
        sr = 22050
        rng = np.random.default_rng(456)
        t = np.linspace(0, 20.0, int(sr * 20.0), endpoint=False)
        breath_env = 0.5 * (1.0 + np.sin(2 * np.pi * (16.0 / 60.0) * t))
        noise = rng.standard_normal(len(t)) * 0.15
        audio = (breath_env * noise).astype(np.float32)
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is not None, (
            "Broadband breathing should not be rejected by fan gate"
        )
        rate = float(result["breathing_rate"])
        assert 8.0 <= rate <= 28.0, f"Expected ~16 bpm, got {rate}"

    def test_fast_breathing_not_rejected_by_fan_gate(self):
        """Fast breathing (40 bpm) in broadband noise must still be detected."""
        sr = 22050
        rng = np.random.default_rng(789)
        t = np.linspace(0, 20.0, int(sr * 20.0), endpoint=False)
        breath_env = 0.5 * (1.0 + np.sin(2 * np.pi * (40.0 / 60.0) * t))
        noise = rng.standard_normal(len(t)) * 0.15
        audio = (breath_env * noise).astype(np.float32)
        path = self._write_wav(audio, sr)
        result = analyze_audio(path)
        Path(path).unlink(missing_ok=True)
        assert result["breathing_rate"] is not None, (
            "Fast breathing should not be rejected by fan gate"
        )
        rate = float(result["breathing_rate"])
        assert 25.0 <= rate <= 55.0, f"Expected ~40 bpm, got {rate}"

    def test_no_vita_score_from_fan_noise(self):
        """Score engine must exclude breathing rejected as fan noise."""
        from backend.app.ml.fusion.score_engine import _normalise_breathing
        fan_result = {
            "breathing_rate": None,
            "value": None,
            "risk": "unreliable",
            "reliability": "unreliable",
            "retake_required": True,
            "confidence": 0.0,
        }
        score = _normalise_breathing(fan_result)
        assert score is None, f"Fan noise should not produce a score, got {score}"
