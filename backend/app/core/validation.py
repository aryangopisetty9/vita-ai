"""
Vita AI – Input Validation
============================
Reusable validators for uploaded files and request data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set

from fastapi import UploadFile

logger = logging.getLogger(__name__)

# Maximum upload sizes (bytes)
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB
MAX_TEXT_LENGTH = 10_000             # characters

ALLOWED_VIDEO_EXTS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_AUDIO_EXTS: Set[str] = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}


def validate_video_upload(file: UploadFile) -> Optional[str]:
    """Validate a video upload.  Returns error message or None."""
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        return f"Unsupported video format: {ext}. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}"

    # Check file size by reading content length header if available
    if file.size is not None and file.size > MAX_VIDEO_SIZE:
        mb = MAX_VIDEO_SIZE / (1024 * 1024)
        return f"Video file too large. Maximum: {mb:.0f} MB."

    return None


def validate_audio_upload(file: UploadFile) -> Optional[str]:
    """Validate an audio upload.  Returns error message or None."""
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        return f"Unsupported audio format: {ext}. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}"

    if file.size is not None and file.size > MAX_AUDIO_SIZE:
        mb = MAX_AUDIO_SIZE / (1024 * 1024)
        return f"Audio file too large. Maximum: {mb:.0f} MB."

    return None


def validate_symptom_text(text: str) -> Optional[str]:
    """Validate symptom text input.  Returns error message or None."""
    if not text or not text.strip():
        return "Symptom text cannot be empty."
    if len(text) > MAX_TEXT_LENGTH:
        return f"Text too long. Maximum: {MAX_TEXT_LENGTH} characters."
    return None
