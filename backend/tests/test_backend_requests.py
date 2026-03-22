"""
Vita AI Backend Integration Test Script
========================================

Running Instructions
--------------------

Step 1: Start the FastAPI server
    uvicorn api.main:app --reload

Step 2: Run this script (from the project root)
    python tests/test_backend_requests.py

Requirements
------------
- requests library  (pip install requests)
- Backend running at http://127.0.0.1:8000
- Sample files can be absent; the script handles missing files gracefully.

Sample files used (place them in sample_data/ if you have them):
    sample_data/test_face.mp4
    sample_data/test_audio.wav
"""

import json
import os
import sys

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' library not installed. Run: pip install requests")
    sys.exit(1)

BASE_URL = "http://127.0.0.1:8000"
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _divider(title: str) -> None:
    width = 50
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _print_response(resp: requests.Response) -> None:
    if resp.ok:
        try:
            data = resp.json()
            print(json.dumps(data, indent=2, default=str))
        except ValueError:
            print(resp.text)
    else:
        print(f"[HTTP {resp.status_code}]  {resp.reason}")
        try:
            print(json.dumps(resp.json(), indent=2, default=str))
        except ValueError:
            print(resp.text)


def _post(label: str, url: str, **kwargs) -> None:
    """Send a POST request and pretty-print the result."""
    _divider(label)
    try:
        resp = requests.post(url, timeout=120, **kwargs)
        _print_response(resp)
    except requests.exceptions.ConnectionError:
        print(
            f"[CONNECTION ERROR] Could not reach {url}\n"
            "  Make sure the server is running:\n"
            "      uvicorn api.main:app --reload"
        )
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] Request to {url} timed out after 120 s.")
    except Exception as exc:
        print(f"[ERROR] {exc}")


# ---------------------------------------------------------------------------
# Individual endpoint tests
# ---------------------------------------------------------------------------

def test_health() -> None:
    """Verify the server is reachable before running other tests."""
    _divider("HEALTH CHECK")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        _print_response(resp)
    except requests.exceptions.ConnectionError:
        msg = (
            "[CONNECTION ERROR] Server is not running.\n"
            "  Start it with:  uvicorn api.main:app --reload"
        )
        # When executed via pytest, skip instead of hard-exiting
        try:
            import pytest as _pt
            _pt.skip(msg)
        except ImportError:
            print(msg)
            sys.exit(1)


def test_face() -> None:
    """POST /predict/face – upload a sample video."""
    face_path = os.path.join(SAMPLE_DATA_DIR, "test_face.mp4")
    if not os.path.isfile(face_path):
        _divider("FACE RESULT")
        print(
            f"[SKIP] Sample file not found: {face_path}\n"
            "  Place a short (20-30 s) video at sample_data/test_face.mp4 to run this test."
        )
        return

    with open(face_path, "rb") as fh:
        _post(
            "FACE RESULT",
            f"{BASE_URL}/predict/face",
            files={"file": ("test_face.mp4", fh, "video/mp4")},
        )


def test_audio() -> None:
    """POST /predict/audio – upload a sample audio file."""
    audio_path = os.path.join(SAMPLE_DATA_DIR, "test_audio.wav")
    if not os.path.isfile(audio_path):
        _divider("AUDIO RESULT")
        print(
            f"[SKIP] Sample file not found: {audio_path}\n"
            "  Place a .wav recording at sample_data/test_audio.wav to run this test."
        )
        return

    with open(audio_path, "rb") as fh:
        _post(
            "AUDIO RESULT",
            f"{BASE_URL}/predict/audio",
            files={"file": ("test_audio.wav", fh, "audio/wav")},
        )


def test_symptom() -> None:
    """POST /predict/symptom – send example symptom text."""
    payload = {
        "text": (
            "I have been experiencing persistent headaches, fatigue, "
            "shortness of breath, and mild chest tightness for the past week."
        )
    }
    _post(
        "SYMPTOM RESULT",
        f"{BASE_URL}/predict/symptom",
        json=payload,
    )


def test_final_score() -> None:
    """POST /predict/final-score – send mock module outputs."""
    payload = {
        "face_result": {
            "module_name": "face_module",
            "heart_rate": 76.5,
            "heart_rate_confidence": 0.82,
            "blink_rate": 14.0,
            "eye_stability": 0.90,
            "facial_tension_index": 0.20,
            "skin_signal_stability": 0.78,
            "scan_quality": 0.75,
            "retake_required": False,
            "risk": "low",
            "confidence": 0.82,
            "value": 76.5,
            "unit": "bpm",
            "metric_name": "heart_rate",
            "message": "Scan completed successfully.",
        },
        "audio_result": {
            "module_name": "audio_module",
            "metric_name": "breathing_rate",
            "value": 16.0,
            "unit": "breaths/min",
            "confidence": 0.74,
            "risk": "low",
            "message": "Breathing rate within normal range.",
        },
        "symptom_result": {
            "module_name": "symptom_module",
            "metric_name": "symptom_risk",
            "value": 0.35,
            "unit": "score",
            "confidence": 0.80,
            "risk": "moderate",
            "message": "Moderate risk detected from reported symptoms.",
            "detected_symptoms": ["headache", "fatigue", "shortness of breath"],
        },
    }
    _post(
        "FINAL SCORE",
        f"{BASE_URL}/predict/final-score",
        json=payload,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Vita AI Backend Integration Tests")
    print(f"  Target: {BASE_URL}")
    print("=" * 50)

    test_health()
    test_face()
    test_audio()
    test_symptom()
    test_final_score()

    print(f"\n{'=' * 50}")
    print("  All tests completed.")
    print("=" * 50)
