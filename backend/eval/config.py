"""
Eval – Configuration
=====================
Central constants for the rPPG evaluation harness.
Override via CLI flags or by editing values here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
RESULTS_DIR = BACKEND_DIR / "data" / "eval_results"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
DEFAULT_FPS = 30.0
MAX_SCAN_SECONDS = 60          # cap per-video processing time
HR_RANGE = (40, 200)           # physiological BPM clamp range

# ---------------------------------------------------------------------------
# Dataset type identifiers
# ---------------------------------------------------------------------------
DATASET_UBFC = "ubfc"
DATASET_COHFACE = "cohface"
DATASET_MAHNOB = "mahnob"
SUPPORTED_DATASETS = [DATASET_UBFC, DATASET_COHFACE, DATASET_MAHNOB]

# ---------------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------------
PLOT_DPI = 150
PLOT_FORMAT = "png"            # png | pdf | svg
FIGSIZE_TIMESERIES = (12, 5)
FIGSIZE_BLAND_ALTMAN = (8, 6)
FIGSIZE_HISTOGRAM = (8, 5)
