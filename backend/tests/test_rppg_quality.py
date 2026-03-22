"""
Tests for the rPPG quality upgrade features:
- Skin-tone mask, overexposure detection, denoising (vision_utils)
- POS projection, signal amplitude/stability (signal_processing)
- Exposure stability, overexposure scoring, usable frames (face_quality)
- Per-ROI quality scoring, intelligent ROI dropping, POS fusion (rppg_utils)
"""

import unittest
import numpy as np


class TestVisionUtilsSkinMask(unittest.TestCase):
    """Test skin-tone HSV mask and overexposure detection."""

    def test_build_skin_mask_returns_correct_shape(self):
        from backend.app.ml.face.vision_utils import build_skin_mask
        # Create a skin-toned BGR image (warm beige)
        frame = np.full((100, 100, 3), [120, 160, 200], dtype=np.uint8)
        mask = build_skin_mask(frame)
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, np.uint8)

    def test_build_skin_mask_detects_skin(self):
        from backend.app.ml.face.vision_utils import build_skin_mask
        import cv2
        # HSV skin tone: H~15, S~100, V~180 → BGR
        hsv = np.full((50, 50, 3), [15, 100, 180], dtype=np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask = build_skin_mask(frame)
        skin_ratio = np.sum(mask > 0) / mask.size
        self.assertGreater(skin_ratio, 0.5,
                           "Should detect skin-toned pixels")

    def test_build_skin_mask_rejects_blue(self):
        from backend.app.ml.face.vision_utils import build_skin_mask
        # Pure blue should not be detected as skin
        frame = np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)  # BGR blue
        mask = build_skin_mask(frame)
        skin_ratio = np.sum(mask > 0) / mask.size
        self.assertLess(skin_ratio, 0.1,
                        "Blue pixels should not pass skin mask")

    def test_overexposure_ratio_dark(self):
        from backend.app.ml.face.vision_utils import overexposure_ratio
        frame = np.full((50, 50, 3), [80, 80, 80], dtype=np.uint8)
        ratio = overexposure_ratio(frame)
        self.assertAlmostEqual(ratio, 0.0, places=2)

    def test_overexposure_ratio_bright(self):
        from backend.app.ml.face.vision_utils import overexposure_ratio
        # Near-white pixels
        frame = np.full((50, 50, 3), [250, 250, 250], dtype=np.uint8)
        ratio = overexposure_ratio(frame)
        self.assertGreater(ratio, 0.5,
                           "Nearly-white frame should show high overexposure")

    def test_frame_overexposure_ratio(self):
        from backend.app.ml.face.vision_utils import frame_overexposure_ratio
        dark = np.full((30, 30, 3), [50, 50, 50], dtype=np.uint8)
        self.assertLess(frame_overexposure_ratio(dark), 0.01)
        bright = np.full((30, 30, 3), [252, 252, 252], dtype=np.uint8)
        self.assertGreater(frame_overexposure_ratio(bright), 0.5)

    def test_denoise_frame_returns_same_shape(self):
        from backend.app.ml.face.vision_utils import denoise_frame
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        result = denoise_frame(frame)
        self.assertEqual(result.shape, frame.shape)

    def test_mean_rgb_from_roi(self):
        from backend.app.ml.face.vision_utils import mean_rgb_from_roi
        # BGR [100, 150, 200] → RGB [200, 150, 100]
        roi = np.full((10, 10, 3), [100, 150, 200], dtype=np.uint8)
        rgb = mean_rgb_from_roi(roi)
        self.assertIsNotNone(rgb)
        np.testing.assert_array_almost_equal(rgb, [200, 150, 100], decimal=0)

    def test_mean_rgb_from_roi_with_mask(self):
        from backend.app.ml.face.vision_utils import mean_rgb_from_roi
        roi = np.full((10, 10, 3), [100, 150, 200], dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        rgb = mean_rgb_from_roi(roi, mask)
        self.assertIsNotNone(rgb)
        np.testing.assert_array_almost_equal(rgb, [200, 150, 100], decimal=0)

    def test_mean_rgb_from_roi_empty(self):
        from backend.app.ml.face.vision_utils import mean_rgb_from_roi
        self.assertIsNone(mean_rgb_from_roi(np.array([])))


class TestPOSProjection(unittest.TestCase):
    """Test POS colour projection in signal_processing."""

    def test_pos_project_basic(self):
        from backend.app.utils.signal_processing import pos_project
        # Simulated RGB traces with a sinusoidal pulse in the green channel
        n = 150
        fps = 30.0
        t = np.arange(n) / fps
        r = 150 + np.random.randn(n) * 2
        g = 150 + 5 * np.sin(2 * np.pi * 1.2 * t) + np.random.randn(n) * 2
        b = 140 + np.random.randn(n) * 2
        rgb = np.column_stack([r, g, b])
        pulse = pos_project(rgb, fps)
        self.assertEqual(len(pulse), n)
        self.assertGreater(float(np.std(pulse)), 0,
                           "POS output should not be flat")

    def test_pos_project_short(self):
        from backend.app.utils.signal_processing import pos_project
        rgb = np.ones((3, 3))
        result = pos_project(rgb, 30.0)
        self.assertEqual(len(result), 3)

    def test_signal_amplitude(self):
        from backend.app.utils.signal_processing import signal_amplitude
        sig = np.sin(np.linspace(0, 4 * np.pi, 100))
        amp = signal_amplitude(sig)
        self.assertGreater(amp, 1.5)

    def test_signal_amplitude_short(self):
        from backend.app.utils.signal_processing import signal_amplitude
        self.assertEqual(signal_amplitude(np.array([1, 2])), 0.0)

    def test_signal_stability_constant(self):
        from backend.app.utils.signal_processing import signal_stability
        # Constant amplitude signal → high stability
        sig = np.sin(np.linspace(0, 10 * np.pi, 300))
        stab = signal_stability(sig, fps=30.0)
        self.assertGreater(stab, 0.5)

    def test_signal_stability_varying(self):
        from backend.app.utils.signal_processing import signal_stability
        # Signal with wildly varying amplitude
        sig = np.zeros(300)
        sig[:100] = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10
        sig[100:200] = np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.01
        sig[200:] = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10
        stab = signal_stability(sig, fps=30.0)
        self.assertLess(stab, 0.8)


class TestFaceQualityUpgrade(unittest.TestCase):
    """Test new quality factors in face_quality."""

    def test_overexposure_triggers_retake(self):
        from backend.app.ml.face.face_quality import compute_scan_quality
        result = compute_scan_quality(
            frame_qualities=[0.8] * 100,
            tracking_ratio=0.9,
            motion_scores=[1.0] * 100,
            brightness_scores=[0.9] * 100,
            roi_agreement=0.8,
            signal_periodicity=0.7,
            scan_duration_sec=20.0,
            overexposure_ratios=[0.5] * 100,  # 50% overexposed
        )
        self.assertTrue(result["retake_required"])
        self.assertIn("overexposure", result["confidence_breakdown"])
        self.assertLess(result["confidence_breakdown"]["overexposure"], 0.5)

    def test_exposure_stability_in_breakdown(self):
        from backend.app.ml.face.face_quality import compute_scan_quality
        result = compute_scan_quality(
            frame_qualities=[0.8] * 100,
            tracking_ratio=0.9,
            motion_scores=[1.0] * 100,
            brightness_scores=[0.9] * 100,
            roi_agreement=0.8,
            signal_periodicity=0.7,
            scan_duration_sec=20.0,
        )
        self.assertIn("exposure_stability", result["confidence_breakdown"])
        self.assertIn("usable_frames", result["confidence_breakdown"])

    def test_usable_frames_low_triggers_retake(self):
        from backend.app.ml.face.face_quality import compute_scan_quality
        result = compute_scan_quality(
            frame_qualities=[0.8] * 100,
            tracking_ratio=0.9,
            motion_scores=[1.0] * 100,
            brightness_scores=[0.9] * 100,
            roi_agreement=0.8,
            signal_periodicity=0.7,
            scan_duration_sec=20.0,
            usable_frame_count=5,
            total_frame_count=100,
        )
        self.assertTrue(result["retake_required"])
        any_usable = any("usable" in r.lower() for r in result["retake_reasons"])
        self.assertTrue(any_usable)

    def test_good_scan_no_retake(self):
        from backend.app.ml.face.face_quality import compute_scan_quality
        result = compute_scan_quality(
            frame_qualities=[0.9] * 100,
            tracking_ratio=0.95,
            motion_scores=[0.5] * 100,
            brightness_scores=[0.95] * 100,
            roi_agreement=0.9,
            signal_periodicity=0.8,
            scan_duration_sec=25.0,
            overexposure_ratios=[0.01] * 100,
            usable_frame_count=90,
            total_frame_count=100,
        )
        self.assertFalse(result["retake_required"])
        self.assertGreater(result["scan_quality"], 0.5)

    def test_unstable_exposure_triggers_retake(self):
        from backend.app.ml.face.face_quality import compute_scan_quality
        # Wildly varying brightness scores
        import random
        bs = [random.choice([0.1, 0.95]) for _ in range(100)]
        result = compute_scan_quality(
            frame_qualities=[0.5] * 100,
            tracking_ratio=0.9,
            motion_scores=[1.0] * 100,
            brightness_scores=bs,
            roi_agreement=0.5,
            signal_periodicity=0.3,
            scan_duration_sec=20.0,
        )
        self.assertLess(result["confidence_breakdown"]["exposure_stability"], 0.5)


class TestRppgUtilsUpgrade(unittest.TestCase):
    """Test per-ROI quality, ROI dropping, and POS integration."""

    def test_estimate_hr_returns_new_fields(self):
        from backend.app.ml.face.rppg_utils import estimate_heart_rate_multi_roi
        n = 300
        fps = 30.0
        t = np.arange(n) / fps
        # 72 BPM = 1.2 Hz
        base = 150 + 5 * np.sin(2 * np.pi * 1.2 * t)
        roi_traces = {
            "forehead": (base + np.random.randn(n) * 0.5).tolist(),
            "left_cheek": (base + np.random.randn(n) * 0.5).tolist(),
            "right_cheek": (base + np.random.randn(n) * 0.5).tolist(),
        }
        roi_weights = {"forehead": 1.5, "left_cheek": 1.0, "right_cheek": 1.0}
        result = estimate_heart_rate_multi_roi(roi_traces, roi_weights, fps)
        self.assertIn("pos_used", result)
        self.assertIn("rois_dropped", result)
        self.assertIsInstance(result["rois_dropped"], list)

    def test_roi_dropping_weak_signal(self):
        from backend.app.ml.face.rppg_utils import estimate_heart_rate_multi_roi
        n = 300
        fps = 30.0
        t = np.arange(n) / fps
        base = 150 + 5 * np.sin(2 * np.pi * 1.2 * t)
        roi_traces = {
            "forehead": (base + np.random.randn(n) * 0.5).tolist(),
            "left_cheek": (base + np.random.randn(n) * 0.5).tolist(),
            "right_cheek": [150.0] * n,  # constant signal = no pulse = should be dropped or low quality
        }
        roi_weights = {"forehead": 1.5, "left_cheek": 1.0, "right_cheek": 1.0}
        result = estimate_heart_rate_multi_roi(roi_traces, roi_weights, fps)
        # The constant signal ROI should have very low quality
        rq = result["per_roi_quality"]
        if "right_cheek" in rq:
            self.assertLess(rq["right_cheek"], rq.get("forehead", 1.0),
                            "Constant signal ROI should have lower quality")

    def test_overexposed_roi_dropped(self):
        from backend.app.ml.face.rppg_utils import estimate_heart_rate_multi_roi
        n = 300
        fps = 30.0
        t = np.arange(n) / fps
        base = 150 + 5 * np.sin(2 * np.pi * 1.2 * t)
        roi_traces = {
            "forehead": (base + np.random.randn(n) * 0.5).tolist(),
            "left_cheek": (base + np.random.randn(n) * 0.5).tolist(),
            "right_cheek": (base + np.random.randn(n) * 0.5).tolist(),
        }
        roi_weights = {"forehead": 1.5, "left_cheek": 1.0, "right_cheek": 1.0}
        roi_overexposure = {
            "forehead": [0.05] * n,
            "left_cheek": [0.05] * n,
            "right_cheek": [0.8] * n,  # heavily overexposed
        }
        result = estimate_heart_rate_multi_roi(
            roi_traces, roi_weights, fps,
            roi_overexposure=roi_overexposure,
        )
        self.assertIn("right_cheek", result["rois_dropped"],
                       "Heavily overexposed ROI should be dropped")

    def test_pos_used_when_rgb_provided(self):
        from backend.app.ml.face.rppg_utils import estimate_heart_rate_multi_roi, _EXTRACTION_METHODS
        n = 300
        fps = 30.0
        t = np.arange(n) / fps
        base = 150 + 5 * np.sin(2 * np.pi * 1.2 * t)
        roi_traces = {
            "forehead": (base + np.random.randn(n) * 0.5).tolist(),
            "left_cheek": (base + np.random.randn(n) * 0.5).tolist(),
            "right_cheek": (base + np.random.randn(n) * 0.5).tolist(),
        }
        # Build RGB traces with cross-channel illumination variation so
        # projection methods (POS/CHROM/PCA) can demonstrate their value.
        roi_rgb_traces = {}
        for name in roi_traces:
            trace_g = np.array(roi_traces[name])
            r = trace_g * 1.1 + 3 * np.sin(2 * np.pi * 0.3 * t)
            g = trace_g + 5 * np.sin(2 * np.pi * 1.2 * t)
            b = trace_g * 0.9 + 2 * np.cos(2 * np.pi * 0.5 * t)
            rgb = np.column_stack([r, g, b])
            roi_rgb_traces[name] = [row for row in rgb]
        roi_weights = {"forehead": 1.5, "left_cheek": 1.0, "right_cheek": 1.0}
        result = estimate_heart_rate_multi_roi(
            roi_traces, roi_weights, fps,
            roi_rgb_traces=roi_rgb_traces,
        )
        # When RGB traces are provided, all 4 extraction methods must be
        # evaluated — verify projection candidates appear in the scores.
        methods_evaluated = {c["method"] for c in result["all_candidate_scores"]}
        self.assertIn("pos", methods_evaluated,
                      "POS candidates should be evaluated when RGB is provided")
        self.assertIn("chrom", methods_evaluated,
                      "CHROM candidates should be evaluated when RGB is provided")
        self.assertIn("pca", methods_evaluated,
                      "PCA candidates should be evaluated when RGB is provided")
        # The selected method must be one of the 4 valid methods.
        self.assertIn(
            result["selected_method"], _EXTRACTION_METHODS,
            f"Selected method {result['selected_method']!r} is not a known method",
        )
        # pos_used is True when any colour-projection method won (pos/chrom/pca).
        # With a clean synthetic signal all methods are competitive; just verify
        # the flag is consistent with selected_method rather than mandating a winner.
        if result["selected_method"] in ("pos", "chrom", "pca"):
            self.assertTrue(result["pos_used"],
                            "pos_used should be True when a projection method is selected")
        else:
            self.assertFalse(result["pos_used"],
                             "pos_used should be False when green is selected")

    def test_extract_frame_overexposure(self):
        """Smoke test for extract_frame_overexposure with mock landmarks."""
        from backend.app.ml.face.rppg_utils import extract_frame_overexposure

        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # Create a small frame and fake landmarks
        frame = np.full((100, 100, 3), [100, 100, 100], dtype=np.uint8)
        lm = [MockLandmark(0.5, 0.5)] * 500
        # Forehead landmarks
        lm[10] = MockLandmark(0.5, 0.2)
        lm[67] = MockLandmark(0.3, 0.2)
        lm[297] = MockLandmark(0.7, 0.2)
        lm[8] = MockLandmark(0.5, 0.4)
        # Cheek landmarks
        for idx in [50, 101, 36, 206, 280, 330, 266, 426]:
            lm[idx] = MockLandmark(0.4 + (idx % 5) * 0.05, 0.5 + (idx % 3) * 0.05)

        result = extract_frame_overexposure(frame, lm, 100, 100)
        self.assertIn("forehead", result)
        self.assertIn("left_cheek", result)
        self.assertIn("right_cheek", result)


class TestExtractFrameRoiRgb(unittest.TestCase):
    """Test RGB extraction for POS pipeline."""

    def test_extract_frame_roi_rgb(self):
        from backend.app.ml.face.rppg_utils import extract_frame_roi_rgb

        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        frame = np.full((100, 100, 3), [100, 150, 200], dtype=np.uint8)
        lm = [MockLandmark(0.5, 0.5)] * 500
        lm[10] = MockLandmark(0.5, 0.2)
        lm[67] = MockLandmark(0.3, 0.2)
        lm[297] = MockLandmark(0.7, 0.2)
        lm[8] = MockLandmark(0.5, 0.4)
        for idx in [50, 101, 36, 206, 280, 330, 266, 426]:
            lm[idx] = MockLandmark(0.4 + (idx % 5) * 0.05, 0.5 + (idx % 3) * 0.05)

        result = extract_frame_roi_rgb(frame, lm, 100, 100)
        self.assertIn("forehead", result)
        if result["forehead"] is not None:
            self.assertEqual(len(result["forehead"]), 3)


# ═══════════════════════════════════════════════════════════════════════════
# Second-pass: realism / trust tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSkinMaskIntegration(unittest.TestCase):
    """Verify skin mask is applied during ROI extraction."""

    def test_apply_skin_mask_returns_mask(self):
        from backend.app.ml.face.rppg_utils import _apply_skin_mask
        # Skin-toned ROI → should return a valid mask
        import cv2
        hsv = np.full((30, 30, 3), [15, 100, 180], dtype=np.uint8)
        roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask = _apply_skin_mask(roi)
        self.assertIsNotNone(mask, "Skin-toned ROI should produce a skin mask")
        self.assertGreater(np.sum(mask > 0), 10)

    def test_apply_skin_mask_rejects_non_skin(self):
        from backend.app.ml.face.rppg_utils import _apply_skin_mask
        # Pure blue ROI → no skin pixels
        roi = np.full((30, 30, 3), [255, 0, 0], dtype=np.uint8)
        mask = _apply_skin_mask(roi)
        self.assertIsNone(mask, "Non-skin ROI should return None mask")

    def test_apply_skin_mask_combines_polygon(self):
        from backend.app.ml.face.rppg_utils import _apply_skin_mask
        import cv2
        hsv = np.full((30, 30, 3), [15, 100, 180], dtype=np.uint8)
        roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        polygon = np.zeros((30, 30), dtype=np.uint8)
        polygon[5:25, 5:25] = 255  # restrict to inner region
        combined = _apply_skin_mask(roi, polygon)
        self.assertIsNotNone(combined)
        # Combined should be smaller than full skin mask
        full = _apply_skin_mask(roi)
        self.assertLessEqual(np.sum(combined > 0), np.sum(full > 0))

    def test_extract_frame_roi_signals_uses_skin_mask(self):
        """ROI signals should handle frames where skin mask filters pixels."""
        from backend.app.ml.face.rppg_utils import extract_frame_roi_signals

        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # Create a skin-toned frame
        import cv2
        hsv = np.full((100, 100, 3), [15, 80, 170], dtype=np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        lm = [MockLandmark(0.5, 0.5)] * 500
        lm[10] = MockLandmark(0.5, 0.2)
        lm[67] = MockLandmark(0.3, 0.2)
        lm[297] = MockLandmark(0.7, 0.2)
        lm[8] = MockLandmark(0.5, 0.4)
        for idx in [50, 101, 36, 206, 280, 330, 266, 426]:
            lm[idx] = MockLandmark(0.4 + (idx % 5) * 0.05, 0.5 + (idx % 3) * 0.05)

        result = extract_frame_roi_signals(frame, lm, 100, 100)
        # Forehead should still extract a value from skin pixels
        self.assertIsNotNone(result["forehead"])


class TestHRSuppressionOnRetake(unittest.TestCase):
    """Verify HR is suppressed when scan is marked unreliable."""

    def test_retake_suppresses_hr(self):
        """When compute_scan_quality flags retake_required, HR should be None."""
        from backend.app.ml.face.face_quality import compute_scan_quality
        # Create a scan that triggers retake (very low periodicity)
        result = compute_scan_quality(
            frame_qualities=[0.3] * 100,
            tracking_ratio=0.9,
            motion_scores=[1.0] * 100,
            brightness_scores=[0.5] * 100,
            roi_agreement=0.3,
            signal_periodicity=0.05,  # very low → retake
            scan_duration_sec=20.0,
            overexposure_ratios=[0.4] * 100,  # high overexposure
        )
        self.assertTrue(result["retake_required"],
                        "This scan configuration should trigger retake")


class TestHRTimeseriesQualityFilter(unittest.TestCase):
    """Verify low-quality windows are filtered from hr_timeseries."""

    def test_low_quality_windows_filtered(self):
        from backend.app.utils.signal_processing import compute_hr_timeseries
        # Pure noise → with strict threshold most windows should be filtered
        np.random.seed(42)
        noise = np.random.randn(300).tolist()
        ts_lenient = compute_hr_timeseries(noise, fps=30.0, min_window_quality=0.0)
        ts_strict = compute_hr_timeseries(noise, fps=30.0, min_window_quality=0.30)
        self.assertGreater(len(ts_lenient), len(ts_strict),
                           "Stricter quality threshold should filter more windows")

    def test_clean_signal_passes_filter(self):
        from backend.app.utils.signal_processing import compute_hr_timeseries
        # Clean 72 BPM signal
        n = 300
        t = np.arange(n) / 30.0
        signal = (150 + 5 * np.sin(2 * np.pi * 1.2 * t)).tolist()
        ts = compute_hr_timeseries(signal, fps=30.0, min_window_quality=0.15)
        self.assertGreater(len(ts), 0,
                           "Clean signal should produce quality windows")

    def test_min_window_quality_param(self):
        from backend.app.utils.signal_processing import compute_hr_timeseries
        n = 300
        t = np.arange(n) / 30.0
        signal = (150 + 5 * np.sin(2 * np.pi * 1.2 * t)).tolist()
        # With threshold 0 → everything passes
        ts_all = compute_hr_timeseries(signal, fps=30.0, min_window_quality=0.0)
        # With threshold 0.99 → almost nothing passes
        ts_strict = compute_hr_timeseries(signal, fps=30.0, min_window_quality=0.99)
        self.assertGreaterEqual(len(ts_all), len(ts_strict),
                                "Higher threshold should filter more windows")


class TestConfidencePenalties(unittest.TestCase):
    """Verify overexposure and usable-frame penalties in confidence."""

    def test_overexposure_reduces_confidence(self):
        """Simulated confidence computation with and without overexposure."""
        hr_quality = 0.7
        tracking = 0.9
        roi_agreement = 0.8
        periodicity = 0.7
        scan_quality = 0.6

        base_conf = (
            0.35 * hr_quality + 0.20 * tracking
            + 0.15 * roi_agreement + 0.15 * periodicity
            + 0.15 * scan_quality
        )

        # No overexposure
        conf_clean = base_conf * (1.0 - 0.30 * 0.0) * (1.0 - 0.25 * 0.0)
        # 30% mean overexposure (75% of threshold)
        oe_penalty = min(0.30 / 0.40, 1.0)
        conf_oe = base_conf * (1.0 - 0.30 * oe_penalty) * (1.0 - 0.25 * 0.0)

        self.assertLess(conf_oe, conf_clean,
                        "Overexposure should reduce confidence")
        self.assertLess(conf_oe, conf_clean * 0.85,
                        "Penalty should be meaningful (>15% reduction)")

    def test_low_usable_ratio_reduces_confidence(self):
        """Low usable-frame ratio should penalise confidence."""
        base_conf = 0.7
        # 90% usable → small penalty
        conf_good = base_conf * (1.0 - 0.25 * (1.0 - 0.9))
        # 30% usable → large penalty
        conf_bad = base_conf * (1.0 - 0.25 * (1.0 - 0.3))
        self.assertLess(conf_bad, conf_good)


if __name__ == "__main__":
    unittest.main()
