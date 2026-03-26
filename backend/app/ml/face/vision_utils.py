"""
Vita AI – Vision Utilities
============================
Reusable OpenCV / MediaPipe helpers for ROI extraction, landmark
processing, colour sampling, and frame-level quality metrics.

Kept separate from the face module so teammates can import individual
helpers into other pipelines (e.g. skin-lesion analysis, wound ROI).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ═══════════════════════════════════════════════════════════════════════════
# MediaPipe landmark index groups
# ═══════════════════════════════════════════════════════════════════════════

# Eyes (6-point EAR model)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Iris centres (needs refine_landmarks=True)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Forehead ROI anchors
FOREHEAD_TOP = 10
FOREHEAD_LEFT = 67
FOREHEAD_RIGHT = 297
NOSE_BRIDGE = 8

# Cheek polygon landmarks (left / right)
LEFT_CHEEK = [50, 101, 36, 206]
RIGHT_CHEEK = [280, 330, 266, 426]

# Region landmark groups for muscle-motion tracking
REGION_LANDMARKS: Dict[str, List[int]] = {
    "forehead": [10, 67, 297, 109, 338, 69, 104, 68],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "jaw": [172, 136, 150, 149, 176, 148, 152, 377, 400, 378,
            379, 365, 397, 288],
}


# ═══════════════════════════════════════════════════════════════════════════
# ROI extraction from MediaPipe landmarks
# ═══════════════════════════════════════════════════════════════════════════

def extract_forehead_roi(frame: np.ndarray, lm_list, h: int, w: int) -> Optional[np.ndarray]:
    """Extract forehead ROI from 468-landmark face mesh.

    Trims a few pixels off the top to reduce hairline contamination
    and narrows to the central ~80% to avoid temple/sideburn leakage.
    """
    top_y = int(lm_list[FOREHEAD_TOP].y * h)
    left_x = int(lm_list[FOREHEAD_LEFT].x * w)
    right_x = int(lm_list[FOREHEAD_RIGHT].x * w)
    bottom_y = int(lm_list[NOSE_BRIDGE].y * h)

    # Reduce height to upper ~50% between forehead top and nose bridge;
    # pull top down slightly to dodge the hairline.
    span = bottom_y - top_y
    roi_top = max(top_y + max(int(span * 0.10), 2), 0)  # skip top 10%
    roi_bottom = min(top_y + int(span * 0.52), h)  # extend to 52% of span (was 38%)

    # Narrow horizontally by 10% each side to avoid temples
    width = right_x - left_x
    roi_left = max(left_x + int(width * 0.10), 0)
    roi_right = min(right_x - int(width * 0.10), w)

    if roi_bottom <= roi_top or roi_right <= roi_left:
        return None
    return frame[roi_top:roi_bottom, roi_left:roi_right]


def extract_cheek_roi(frame: np.ndarray, lm_list, indices: List[int],
                      h: int, w: int) -> Optional[np.ndarray]:
    """Extract a cheek ROI defined by a polygon of landmark indices."""
    if not _HAS_CV2:
        return None
    pts = np.array([(int(lm_list[i].x * w), int(lm_list[i].y * h)) for i in indices])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    if cv2.countNonZero(mask) < 10:
        return None
    # Extract pixels within the polygon
    roi = cv2.bitwise_and(frame, frame, mask=mask)
    # Crop to bounding rect for efficiency
    x, y, rw, rh = cv2.boundingRect(pts)
    cropped = roi[y:y + rh, x:x + rw]
    mask_crop = mask[y:y + rh, x:x + rw]
    if cropped.size == 0:
        return None
    return cropped, mask_crop  # type: ignore[return-value]


def mean_green_from_roi(roi: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Extract mean green-channel value from an ROI, optionally masked."""
    if roi is None or roi.size == 0:
        return None
    if mask is not None:
        pixels = roi[mask > 0]
        if pixels.size == 0:
            return None
        # pixels is flattened; reshape to (..., 3)
        if pixels.ndim == 1:
            pixels = pixels.reshape(-1, 3) if len(pixels) % 3 == 0 else None
            if pixels is None:
                return None
        return float(np.mean(pixels[:, 1]))
    return float(np.mean(roi[:, :, 1]))


def mean_rgb_from_roi(
    roi: np.ndarray, mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract mean [R, G, B] from an ROI (for POS/CHROM projection).

    Returns a (3,) float64 array or None.
    """
    if roi is None or roi.size == 0:
        return None
    if mask is not None:
        pixels = roi[mask > 0]
        if pixels.size == 0:
            return None
        if pixels.ndim == 1:
            pixels = pixels.reshape(-1, 3) if len(pixels) % 3 == 0 else None
            if pixels is None:
                return None
        # OpenCV is BGR
        mean_bgr = np.mean(pixels, axis=0).astype(np.float64)
    else:
        mean_bgr = np.mean(roi.reshape(-1, 3), axis=0).astype(np.float64)
    return mean_bgr[::-1].copy()  # BGR → RGB


def _tile_bounds(height: int, width: int, rows: int = 2, cols: int = 3) -> List[Tuple[int, int, int, int]]:
    """Generate tile bounds as (y0, y1, x0, x1)."""
    ys = np.linspace(0, height, rows + 1, dtype=int)
    xs = np.linspace(0, width, cols + 1, dtype=int)
    bounds: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(ys[r]), int(ys[r + 1])
            x0, x1 = int(xs[c]), int(xs[c + 1])
            if y1 > y0 and x1 > x0:
                bounds.append((y0, y1, x0, x1))
    return bounds


def _tile_quality_weight(tile: np.ndarray, tile_mask: np.ndarray) -> float:
    """Quality weight for a tile using exposure, blur, and edge contamination."""
    valid = int(np.sum(tile_mask > 0))
    if valid < 10:
        return 0.0

    skin_cov = float(valid / max(tile_mask.size, 1))
    oe = overexposure_ratio(tile, tile_mask)

    if _HAS_CV2:
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge = cv2.Canny(gray, 80, 160)
        edge_density = float(np.mean(edge[tile_mask > 0] > 0)) if valid > 0 else 0.0
    else:
        lap_var = 80.0
        edge_density = 0.0

    blur_q = float(np.clip(lap_var / 80.0, 0.0, 1.0))
    oe_pen = float(np.clip(1.0 - oe / 0.35, 0.0, 1.0))
    edge_pen = float(np.clip(1.0 - edge_density / 0.35, 0.0, 1.0))
    return float(np.clip(0.45 * skin_cov + 0.25 * blur_q + 0.20 * oe_pen + 0.10 * edge_pen, 0.0, 1.0))


def mean_green_from_roi_tiled(
    roi: np.ndarray,
    mask: Optional[np.ndarray] = None,
    rows: int = 2,
    cols: int = 3,
) -> Optional[float]:
    """Weighted green mean from sub-ROI tiles, keeping only stronger tiles."""
    if roi is None or roi.size == 0:
        return None

    h, w = roi.shape[:2]
    if h < 6 or w < 6:
        return mean_green_from_roi(roi, mask)

    tile_vals: List[float] = []
    tile_weights: List[float] = []
    for y0, y1, x0, x1 in _tile_bounds(h, w, rows=rows, cols=cols):
        t_roi = roi[y0:y1, x0:x1]
        t_mask = None if mask is None else mask[y0:y1, x0:x1]
        g = mean_green_from_roi(t_roi, t_mask)
        if g is None:
            continue
        if t_mask is None:
            t_mask = np.ones(t_roi.shape[:2], dtype=np.uint8) * 255
        w_q = _tile_quality_weight(t_roi, t_mask)
        if w_q < 0.15:
            continue
        tile_vals.append(float(g))
        tile_weights.append(w_q)

    if not tile_vals:
        return mean_green_from_roi(roi, mask)

    w_arr = np.array(tile_weights, dtype=np.float64)
    v_arr = np.array(tile_vals, dtype=np.float64)
    return float(np.dot(v_arr, w_arr) / np.sum(w_arr))


def mean_rgb_from_roi_tiled(
    roi: np.ndarray,
    mask: Optional[np.ndarray] = None,
    rows: int = 2,
    cols: int = 3,
) -> Optional[np.ndarray]:
    """Weighted RGB mean from sub-ROI tiles, rejecting weak tiles."""
    if roi is None or roi.size == 0:
        return None

    h, w = roi.shape[:2]
    if h < 6 or w < 6:
        return mean_rgb_from_roi(roi, mask)

    tile_vals: List[np.ndarray] = []
    tile_weights: List[float] = []
    for y0, y1, x0, x1 in _tile_bounds(h, w, rows=rows, cols=cols):
        t_roi = roi[y0:y1, x0:x1]
        t_mask = None if mask is None else mask[y0:y1, x0:x1]
        rgb = mean_rgb_from_roi(t_roi, t_mask)
        if rgb is None:
            continue
        if t_mask is None:
            t_mask = np.ones(t_roi.shape[:2], dtype=np.uint8) * 255
        w_q = _tile_quality_weight(t_roi, t_mask)
        if w_q < 0.15:
            continue
        tile_vals.append(rgb.astype(np.float64))
        tile_weights.append(w_q)

    if not tile_vals:
        return mean_rgb_from_roi(roi, mask)

    w_arr = np.array(tile_weights, dtype=np.float64)
    v_arr = np.vstack(tile_vals)
    weighted = (v_arr * w_arr[:, None]).sum(axis=0) / max(np.sum(w_arr), 1e-9)
    return weighted.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Skin-tone HSV mask
# ═══════════════════════════════════════════════════════════════════════════

def build_skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Create a binary skin-tone mask using HSV + YCrCb dual thresholds.

    Dual-space masking catches more skin tones across diverse complexions
    while still rejecting non-skin pixels (hair, background, specular).
    Returns a uint8 mask (255 = skin, 0 = non-skin).
    """
    if not _HAS_CV2:
        return np.ones(frame_bgr.shape[:2], dtype=np.uint8) * 255

    # --- HSV mask (broad range for diverse tones) ---
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 25, 50], dtype=np.uint8)
    upper1 = np.array([25, 255, 250], dtype=np.uint8)
    mask_h1 = cv2.inRange(hsv, lower1, upper1)
    # Wrap-around hue for reddish skin
    lower2 = np.array([160, 25, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 250], dtype=np.uint8)
    mask_h2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = mask_h1 | mask_h2

    # --- YCrCb mask (better skin segmentation across lighting) ---
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Union of both colour spaces (catches more skin tones)
    combined = mask_hsv | mask_ycrcb

    # Light morphological cleanup: remove small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Overexposure detection
# ═══════════════════════════════════════════════════════════════════════════

def overexposure_ratio(roi: np.ndarray, mask: Optional[np.ndarray] = None,
                       threshold: int = 245) -> float:
    """Fraction of pixels in an ROI that are overexposed (V ≥ threshold).

    Returns 0.0 (no overexposure) to 1.0 (fully blown out).
    """
    if roi is None or roi.size == 0:
        return 0.0
    if not _HAS_CV2:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    if mask is not None:
        # Resize mask to match roi if needed
        if mask.shape != v_channel.shape:
            return 0.0
        total = int(np.sum(mask > 0))
        if total == 0:
            return 0.0
        overexposed = int(np.sum((v_channel >= threshold) & (mask > 0)))
        return overexposed / total
    total = v_channel.size
    if total == 0:
        return 0.0
    return int(np.sum(v_channel >= threshold)) / total


def frame_overexposure_ratio(frame_bgr: np.ndarray, threshold: int = 245) -> float:
    """Fraction of the entire frame that is overexposed."""
    if frame_bgr is None or frame_bgr.size == 0 or not _HAS_CV2:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return float(np.sum(v >= threshold)) / v.size


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight frame denoising
# ═══════════════════════════════════════════════════════════════════════════

def denoise_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Mild bilateral filter to reduce sensor noise while preserving edges.

    This targets low-quality webcam noise that corrupts the subtle
    green-channel rPPG signal.  The parameters are deliberately gentle
    so that the colour information is not smeared.
    """
    if not _HAS_CV2:
        return frame_bgr
    return cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=25, sigmaSpace=25)


# ═══════════════════════════════════════════════════════════════════════════
# Colour sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_polygon_color(frame_bgr: np.ndarray, lm_list,
                          indices: List[int], h: int, w: int) -> Optional[Dict[str, Any]]:
    """Mean BGR / HSV colour from a polygon defined by landmark indices."""
    if not _HAS_CV2:
        return None
    pts = np.array([(int(lm_list[i].x * w), int(lm_list[i].y * h)) for i in indices])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    if cv2.countNonZero(mask) < 10:
        return None
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    r, g, b = int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])
    hsv_pixel = cv2.cvtColor(
        np.uint8([[[int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])]]]),
        cv2.COLOR_BGR2HSV,
    )
    h_val, s_val, v_val = int(hsv_pixel[0, 0, 0]), int(hsv_pixel[0, 0, 1]), int(hsv_pixel[0, 0, 2])
    return {"rgb": [r, g, b], "hsv": [h_val, s_val, v_val]}


def sample_iris_color(frame_bgr: np.ndarray, lm_list,
                       iris_indices: List[int], h: int, w: int) -> Optional[Dict[str, Any]]:
    """Sample mean colour from the iris landmark region."""
    if not _HAS_CV2:
        return None
    if iris_indices[0] >= len(lm_list):
        return None
    cx = float(np.mean([lm_list[i].x for i in iris_indices])) * w
    cy = float(np.mean([lm_list[i].y for i in iris_indices])) * h
    radius = 3
    x1, y1 = max(int(cx) - radius, 0), max(int(cy) - radius, 0)
    x2, y2 = min(int(cx) + radius, w), min(int(cy) + radius, h)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    mean_bgr = np.mean(patch, axis=(0, 1))
    r, g, b = int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])
    hsv_pixel = cv2.cvtColor(
        np.uint8([[[int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])]]]),
        cv2.COLOR_BGR2HSV,
    )
    h_val, s_val, v_val = int(hsv_pixel[0, 0, 0]), int(hsv_pixel[0, 0, 1]), int(hsv_pixel[0, 0, 2])
    return {"rgb": [r, g, b], "hsv": [h_val, s_val, v_val]}


# ═══════════════════════════════════════════════════════════════════════════
# Frame-level quality metrics
# ═══════════════════════════════════════════════════════════════════════════

def frame_brightness(frame_bgr: np.ndarray) -> float:
    """Return mean V-channel value (0-255) as a brightness proxy."""
    if not _HAS_CV2:
        return 128.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def frame_blur_metrics(frame_bgr: np.ndarray) -> Dict[str, float]:
    """Return blur diagnostics using Laplacian variance."""
    if frame_bgr is None or frame_bgr.size == 0 or not _HAS_CV2:
        return {"laplacian_var": 0.0, "blur_quality": 0.0}
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # <=15: very blurry, >=80: usually sharp enough for ROI photoplethysmography.
    blur_quality = float(np.clip((lap_var - 15.0) / 65.0, 0.0, 1.0))
    return {
        "laplacian_var": round(lap_var, 4),
        "blur_quality": round(blur_quality, 4),
    }


def apply_clahe(frame_bgr: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Equalises the L-channel in LAB colour space so lighting variations
    across the face are reduced without distorting colour information
    needed for rPPG green-channel extraction.
    """
    if not _HAS_CV2:
        return frame_bgr
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correct(frame_bgr: np.ndarray, brightness: float,
                  target: float = 140.0) -> np.ndarray:
    """Adaptive gamma correction to normalise brightness toward *target*.

    When the frame is dark (brightness < target) gamma < 1 brightens it;
    when over-exposed gamma > 1 darkens it.  The correction is mild and
    clamped to avoid artefacts.
    """
    if not _HAS_CV2 or brightness <= 0:
        return frame_bgr
    # Compute gamma: ratio of log(target/255) / log(brightness/255)
    ratio = brightness / target
    gamma = float(np.clip(ratio, 0.5, 2.0))  # clamp for safety
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame_bgr, table)


def brightness_quality(brightness: float) -> float:
    """Map brightness (0-255) to a 0-1 quality score.

    Sweet spot: 80-200.  Too dark (<40) or overexposed (>240) → low quality.
    """
    if brightness < 40:
        return max(brightness / 40.0, 0.0)
    if brightness > 240:
        return max((255 - brightness) / 15.0, 0.0)
    if 80 <= brightness <= 200:
        return 1.0
    if brightness < 80:
        return 0.5 + 0.5 * (brightness - 40) / 40.0
    # 200 < brightness <= 240
    return 0.5 + 0.5 * (240 - brightness) / 40.0


# ═══════════════════════════════════════════════════════════════════════════
# Landmark-based motion metrics
# ═══════════════════════════════════════════════════════════════════════════

def landmark_positions(lm_list, indices: List[int], h: int, w: int) -> np.ndarray:
    """Nx2 array of (x, y) pixel positions for given landmark indices."""
    return np.array([(lm_list[i].x * w, lm_list[i].y * h) for i in indices])


def head_motion_magnitude(prev_landmarks: np.ndarray, curr_landmarks: np.ndarray) -> float:
    """Mean Euclidean displacement between two sets of key landmarks.

    Use a stable subset (nose bridge, outer eye corners) that move mostly
    due to head motion rather than expressions.
    """
    if prev_landmarks.shape != curr_landmarks.shape or len(prev_landmarks) == 0:
        return 0.0
    disp = np.linalg.norm(curr_landmarks - prev_landmarks, axis=1)
    return float(np.mean(disp))


# Stable anchor landmarks for head-motion estimation
HEAD_ANCHORS = [1, 4, 5, 6, 10, 33, 133, 263, 362, 168]


def extract_anchor_positions(lm_list, h: int, w: int) -> np.ndarray:
    """Extract positions of stable head-anchor landmarks."""
    return landmark_positions(lm_list, HEAD_ANCHORS, h, w)


# ═══════════════════════════════════════════════════════════════════════════
# Haar-cascade fallback
# ═══════════════════════════════════════════════════════════════════════════

def extract_face_roi_haar(frame: np.ndarray, cascade) -> Optional[np.ndarray]:
    """Upper 30 % of Haar-detected face rectangle as forehead ROI."""
    if not _HAS_CV2:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    fh = max(int(h * 0.30), 1)
    return frame[y:y + fh, x:x + w]


# ═══════════════════════════════════════════════════════════════════════════
# ROI Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_roi_quality_metrics(
    roi: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute quality metrics for a single ROI crop.

    Returns skin_coverage, mean_brightness, overexposure_ratio,
    and color_variance (temporal proxy from spatial variance).
    """
    if roi is None or roi.size == 0:
        return {
            "skin_coverage": 0.0,
            "mean_brightness": 0.0,
            "overexposure_ratio": 0.0,
            "color_variance": 0.0,
            "quality_score": 0.0,
        }

    total_pixels = roi.shape[0] * roi.shape[1]

    # Skin coverage
    skin_mask = build_skin_mask(roi)
    if mask is not None:
        skin_mask = skin_mask & mask
    skin_pixels = int(np.sum(skin_mask > 0))
    skin_coverage = skin_pixels / max(total_pixels, 1)

    # Brightness from V-channel
    if _HAS_CV2:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        if mask is not None:
            skin_v = v_channel[skin_mask > 0]
            mean_bright = float(np.mean(skin_v)) / 255.0 if len(skin_v) > 0 else 0.0
        else:
            mean_bright = float(np.mean(v_channel)) / 255.0
        # Overexposure
        oe = overexposure_ratio(roi, mask)
    else:
        mean_bright = 0.5
        oe = 0.0

    # Color variance (spatial) — proxy for how homogeneous the skin is
    if _HAS_CV2 and skin_pixels > 10:
        pixels = roi[skin_mask > 0]
        if pixels.ndim == 1:
            pixels = pixels.reshape(-1, 3) if len(pixels) % 3 == 0 else None
        if pixels is not None and len(pixels) > 0:
            color_var = float(np.mean(np.std(pixels.astype(np.float64), axis=0)))
        else:
            color_var = 50.0
    else:
        color_var = 50.0

    # Normalised color variance: low variance = homogeneous skin = good
    color_var_score = float(np.clip(1.0 - color_var / 60.0, 0.0, 1.0))

    # Brightness quality: sweet-spot is 0.3-0.8
    if mean_bright < 0.15:
        bright_q = max(mean_bright / 0.15, 0.0)
    elif mean_bright > 0.90:
        bright_q = max((1.0 - mean_bright) / 0.10, 0.0)
    else:
        bright_q = 1.0

    oe_penalty = float(np.clip(1.0 - oe / 0.30, 0.0, 1.0))

    # Composite quality
    quality_score = (
        0.30 * skin_coverage
        + 0.25 * bright_q
        + 0.25 * oe_penalty
        + 0.20 * color_var_score
    )

    return {
        "skin_coverage": round(skin_coverage, 4),
        "mean_brightness": round(mean_bright, 4),
        "overexposure_ratio": round(oe, 4),
        "color_variance": round(color_var, 4),
        "quality_score": round(float(np.clip(quality_score, 0.0, 1.0)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Landmark Smoothing (EMA-based ROI stabilisation)
# ═══════════════════════════════════════════════════════════════════════════

class LandmarkSmoother:
    """EMA-based smoothing of MediaPipe landmarks across frames.

    Reduces frame-to-frame jitter so ROIs stay locked to the same skin
    patch.  The smoothing factor ``alpha`` controls responsiveness:
    higher = more responsive (less smoothing), lower = more stable.
    """

    def __init__(self, alpha: float = 0.6) -> None:
        self._alpha = alpha
        self._prev_positions: Optional[np.ndarray] = None

    def smooth(self, lm_list, h: int, w: int) -> np.ndarray:
        """Return an Nx3 array of smoothed (x_px, y_px, z) positions.

        ``lm_list`` should be the raw MediaPipe landmark list.  The
        returned array has pixel coordinates for x/y.
        """
        n = len(lm_list)
        raw = np.array(
            [(lm_list[i].x * w, lm_list[i].y * h, getattr(lm_list[i], 'z', 0.0))
             for i in range(n)],
            dtype=np.float64,
        )
        if self._prev_positions is None or self._prev_positions.shape != raw.shape:
            self._prev_positions = raw.copy()
            return raw
        smoothed = self._alpha * raw + (1.0 - self._alpha) * self._prev_positions
        self._prev_positions = smoothed.copy()
        return smoothed

    def smooth_to_landmarks(self, lm_list, h: int, w: int):
        """In-place smooth landmark positions (mutates lm_list coords).

        After calling this, extracting ROIs from lm_list will use
        stabilised coordinates.
        """
        smoothed = self.smooth(lm_list, h, w)
        for i in range(len(lm_list)):
            lm_list[i].x = smoothed[i, 0] / w
            lm_list[i].y = smoothed[i, 1] / h

    def roi_drift(self, lm_list, h: int, w: int) -> float:
        """Mean pixel displacement since last frame (before smoothing).

        High drift = head/camera motion; penalise this window.
        """
        n = len(lm_list)
        raw = np.array(
            [(lm_list[i].x * w, lm_list[i].y * h) for i in range(n)],
            dtype=np.float64,
        )
        if self._prev_positions is None:
            return 0.0
        diff = raw - self._prev_positions[:, :2]
        return float(np.mean(np.linalg.norm(diff, axis=1)))
