"""
Eval – Dataset Loaders
=======================
Load benchmark rPPG datasets from local directories.

Supported datasets
------------------
- **UBFC-rPPG** — ``vid.avi`` + ``ground_truth.txt`` per subject.
- **COHFACE** — ``data.avi`` + ``data.hdf5`` (or ``hr.csv``) per subject.
- **MAHNOB-HCI** — ``.avi`` + ``*_BPM.csv`` / ``*.bdf`` per session.

None of these datasets are downloaded automatically.  The user must
place them under the expected directory layout (see README).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.eval.config import (
    DATASET_COHFACE,
    DATASET_MAHNOB,
    DATASET_UBFC,
    DEFAULT_FPS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SampleRecord:
    """One evaluation sample (video + ground truth)."""

    subject_id: str
    video_path: str
    ground_truth_bpm: List[float] = field(default_factory=list)
    ground_truth_mean_bpm: Optional[float] = None
    fps: float = DEFAULT_FPS
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_csv_column(
    path: str, col: int = 0, skip_header: bool = False
) -> List[float]:
    """Read numeric values from a single column of a CSV/TSV file."""
    values: List[float] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        # Detect delimiter
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        reader = csv.reader(fh, dialect)
        if skip_header:
            next(reader, None)
        for row in reader:
            if len(row) <= col:
                continue
            token = row[col].strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


def _read_single_column_txt(path: str) -> List[float]:
    """Read a plain-text file with one number per line."""
    values: List[float] = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            token = line.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


def _mean_bpm(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(np.mean(values)), 2)


# ═══════════════════════════════════════════════════════════════════════════
# UBFC-rPPG Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_ubfc(root: str, limit: Optional[int] = None) -> List[SampleRecord]:
    """Load UBFC-rPPG dataset.

    Expected layout::

        root/
          subject1/
            vid.avi
            ground_truth.txt     ← one BPM value per line (or PPG + timestamps)
          subject2/
            ...

    ``ground_truth.txt`` may contain either:
    - One BPM value per line, **or**
    - Three space-separated columns: PPG-signal, HR(bpm), time(s)
      (UBFC-rPPG dataset 2 format).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        logger.error("UBFC root not found: %s", root)
        return []

    records: List[SampleRecord] = []

    subject_dirs = sorted(
        [d for d in root_path.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    for sdir in subject_dirs:
        if limit is not None and len(records) >= limit:
            break

        video = _find_video(sdir)
        if video is None:
            logger.warning("No video found in %s – skipping.", sdir)
            continue

        gt_file = sdir / "ground_truth.txt"
        if not gt_file.is_file():
            # Try alternate names
            for alt in ("gt.txt", "GT.txt", "ground_truth.csv", "bpm.txt"):
                alt_path = sdir / alt
                if alt_path.is_file():
                    gt_file = alt_path
                    break

        bpm_values: List[float] = []
        if gt_file.is_file():
            bpm_values = _parse_ubfc_gt(str(gt_file))

        records.append(
            SampleRecord(
                subject_id=sdir.name,
                video_path=str(video),
                ground_truth_bpm=bpm_values,
                ground_truth_mean_bpm=_mean_bpm(bpm_values),
                metadata={"dataset": "ubfc", "gt_file": str(gt_file)},
            )
        )

    logger.info("UBFC: loaded %d samples from %s", len(records), root)
    return records


def _parse_ubfc_gt(path: str) -> List[float]:
    """Parse UBFC ground_truth.txt.

    Handles two common formats:
    1. Single column – one BPM per line.
    2. Three columns (PPG, HR_bpm, timestamp) – extract column 1.
    """
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline().strip()
    parts = first.split()
    if len(parts) >= 2:
        # Multi-column format → BPM is second column (index 1)
        return _read_csv_column(path, col=1)
    # Single value per line
    return _read_single_column_txt(path)


# ═══════════════════════════════════════════════════════════════════════════
# COHFACE Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_cohface(root: str, limit: Optional[int] = None) -> List[SampleRecord]:
    """Load COHFACE dataset.

    Expected layout::

        root/
          1/                       ← subject directory
            0/                     ← session directory
              data.avi
              data.hdf5            ← contains /pulse and /bpm groups
          ...

    If ``data.hdf5`` is not readable (h5py missing), falls back to
    ``hr.csv`` if present.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        logger.error("COHFACE root not found: %s", root)
        return []

    records: List[SampleRecord] = []

    for subject_dir in sorted(root_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        # Session sub-dirs (0, 1, 2, 3)
        session_dirs = sorted(
            [d for d in subject_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        if not session_dirs:
            # Flat layout: video directly in subject dir
            session_dirs = [subject_dir]

        for sess in session_dirs:
            if limit is not None and len(records) >= limit:
                break

            video = _find_video(sess, prefer="data.avi")
            if video is None:
                continue

            bpm_values = _load_cohface_gt(sess)
            sid = f"{subject_dir.name}_{sess.name}" if sess != subject_dir else subject_dir.name

            records.append(
                SampleRecord(
                    subject_id=sid,
                    video_path=str(video),
                    ground_truth_bpm=bpm_values,
                    ground_truth_mean_bpm=_mean_bpm(bpm_values),
                    fps=DEFAULT_FPS,
                    metadata={"dataset": "cohface"},
                )
            )

    logger.info("COHFACE: loaded %d samples from %s", len(records), root)
    return records


def _load_cohface_gt(sess_dir: Path) -> List[float]:
    """Try HDF5 first, then CSV fallback."""
    hdf5_path = sess_dir / "data.hdf5"
    if hdf5_path.is_file():
        try:
            import h5py  # type: ignore

            with h5py.File(str(hdf5_path), "r") as hf:
                if "bpm" in hf:
                    return [float(v) for v in np.asarray(hf["bpm"])]
                if "pulse" in hf:
                    # Raw pulse – caller will need to derive BPM externally;
                    # store raw values for now.
                    return [float(v) for v in np.asarray(hf["pulse"])]
        except Exception as exc:
            logger.debug("Could not read HDF5 %s: %s", hdf5_path, exc)

    csv_path = sess_dir / "hr.csv"
    if csv_path.is_file():
        return _read_csv_column(str(csv_path), col=0, skip_header=True)

    return []


# ═══════════════════════════════════════════════════════════════════════════
# MAHNOB-HCI Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_mahnob(root: str, limit: Optional[int] = None) -> List[SampleRecord]:
    """Load MAHNOB-HCI dataset.

    Expected layout::

        root/
          Sessions/
            1/
              video.avi            ← or *.avi
              session_BPM.csv      ← or *_BPM.csv
            ...

    Falls back to a flat layout (``root/<session_id>/``).
    """
    root_path = Path(root)
    sessions_dir = root_path / "Sessions"
    if not sessions_dir.is_dir():
        sessions_dir = root_path  # flat layout

    if not sessions_dir.is_dir():
        logger.error("MAHNOB root not found: %s", root)
        return []

    records: List[SampleRecord] = []

    for sess in sorted(sessions_dir.iterdir()):
        if not sess.is_dir():
            continue
        if limit is not None and len(records) >= limit:
            break

        video = _find_video(sess)
        if video is None:
            continue

        bpm_values = _load_mahnob_gt(sess)
        records.append(
            SampleRecord(
                subject_id=sess.name,
                video_path=str(video),
                ground_truth_bpm=bpm_values,
                ground_truth_mean_bpm=_mean_bpm(bpm_values),
                metadata={"dataset": "mahnob"},
            )
        )

    logger.info("MAHNOB: loaded %d samples from %s", len(records), root)
    return records


def _load_mahnob_gt(sess_dir: Path) -> List[float]:
    """Load BPM CSV (``*_BPM.csv`` or ``bpm.csv``)."""
    for f in sess_dir.iterdir():
        if f.suffix.lower() == ".csv" and "bpm" in f.name.lower():
            return _read_csv_column(str(f), col=0, skip_header=True)
    # Try generic hr.csv
    hr = sess_dir / "hr.csv"
    if hr.is_file():
        return _read_csv_column(str(hr), col=0, skip_header=True)
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def _find_video(directory: Path, prefer: Optional[str] = None) -> Optional[Path]:
    """Return the first video file found in *directory*.

    If *prefer* is set (e.g. ``"vid.avi"``), try that name first.
    """
    if prefer:
        candidate = directory / prefer
        if candidate.is_file():
            return candidate
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in _VIDEO_EXTS and f.is_file():
            return f
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

_LOADERS = {
    DATASET_UBFC: load_ubfc,
    DATASET_COHFACE: load_cohface,
    DATASET_MAHNOB: load_mahnob,
}


def detect_and_load(
    dataset_type: str,
    root: str,
    limit: Optional[int] = None,
) -> List[SampleRecord]:
    """Load a dataset by type identifier.

    Parameters
    ----------
    dataset_type : str
        One of ``"ubfc"``, ``"cohface"``, ``"mahnob"``.
    root : str
        Path to the dataset root directory.
    limit : int or None
        Maximum number of samples to load.

    Returns
    -------
    list[SampleRecord]
    """
    loader = _LOADERS.get(dataset_type.lower())
    if loader is None:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Supported: {list(_LOADERS)}"
        )
    return loader(root, limit=limit)
