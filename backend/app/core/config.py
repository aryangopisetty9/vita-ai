"""
Vita AI - Configuration

Central configuration for the Vita AI health screening system.
All tunable parameters, thresholds, and constants live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives at backend/app/core/config.py → parents[3] = project root
BASE_DIR = Path(__file__).resolve().parents[3]
SAMPLE_DATA_DIR = BASE_DIR / "backend" / "sample_data"
TEMP_DIR = BASE_DIR / "temp"

# ---------------------------------------------------------------------------
# Face Module Defaults
# ---------------------------------------------------------------------------
FACE_MIN_FRAMES = 30          # minimum frames to attempt rPPG
FACE_ROI_MARGIN = 0.15        # fractional margin around forehead ROI
RPPG_LOW_HZ = 0.7             # ~42 BPM lower band-pass bound
RPPG_HIGH_HZ = 3.5            # ~210 BPM upper band-pass bound
RPPG_NORMAL_LOW = 50          # normal adult resting HR low end
RPPG_NORMAL_HIGH = 100        # normal adult resting HR high end
MOTION_REJECT_THRESHOLD = 15.0  # px displacement to reject a frame
SCAN_MIN_DURATION_SEC = 10.0    # minimum useful scan length
SCAN_QUALITY_THRESHOLD = 0.35   # below this → retake required

# ---------------------------------------------------------------------------
# Audio Module Defaults
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 22050     # librosa default SR
AUDIO_MIN_DURATION = 3.0      # seconds - minimum useful recording
BREATHING_NORMAL_LOW = 12     # normal adult breaths / min
BREATHING_NORMAL_HIGH = 20    # normal adult breaths / min
ENERGY_SMOOTH_WINDOW = 0.4    # seconds for energy envelope smoothing

# ---------------------------------------------------------------------------
# Symptom Module Defaults
# ---------------------------------------------------------------------------
SYMPTOM_HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
SYMPTOM_MIN_TEXT_LENGTH = 3   # characters

# High-caution symptom combinations (any pair → escalate risk)
HIGH_CAUTION_PAIRS = [
    {"chest pain", "breathlessness"},
    {"chest pain", "dizziness"},
    {"breathlessness", "fever"},
    {"confusion", "fever"},
]

# ---------------------------------------------------------------------------
# Score Engine Defaults
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "heart": 0.40,
    "breathing": 0.30,
    "symptom": 0.30,
}

SCORE_THRESHOLDS = {
    "low": 70,       # >= 70 → low risk
    "moderate": 40,  # >= 40 → moderate risk
    # < 40 → high risk
}

# ---------------------------------------------------------------------------
# API Settings
# ---------------------------------------------------------------------------
API_TITLE = "Vita AI"
API_VERSION = "0.1.0"
API_DESCRIPTION = (
    "AI-powered preliminary health screening API. "
    "This is NOT a medical diagnostic system."
)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Model Download & Cache
# ---------------------------------------------------------------------------
VITA_AUTO_DOWNLOAD_MODELS = os.getenv("VITA_AUTO_DOWNLOAD_MODELS", "true").lower() in ("true", "1", "yes")
VITA_PRELOAD_MODELS = os.getenv("VITA_PRELOAD_MODELS", "true").lower() in ("true", "1", "yes")
VITA_ENABLE_DISTILBERT = os.getenv("VITA_ENABLE_DISTILBERT", "true").lower() in ("true", "1", "yes")
VITA_ENABLE_BIOBERT = os.getenv("VITA_ENABLE_BIOBERT", "true").lower() in ("true", "1", "yes")
VITA_ENABLE_YAMNET = os.getenv("VITA_ENABLE_YAMNET", "true").lower() in ("true", "1", "yes")
# New primary rPPG face models
VITA_ENABLE_OPEN_RPPG = os.getenv("VITA_ENABLE_OPEN_RPPG", "true").lower() in ("true", "1", "yes")
VITA_OPEN_RPPG_MODEL = os.getenv("VITA_OPEN_RPPG_MODEL", "FacePhys.rlap")
