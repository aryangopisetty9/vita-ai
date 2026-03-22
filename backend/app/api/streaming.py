"""
Vita AI – WebSocket Streaming Router
======================================
Enhanced real-time face-scan streaming with session management,
audio streaming, and per-frame metrics.

Mounted as a sub-router in ``api/main.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.services.session_manager import session_manager

logger = logging.getLogger("vita_api.streaming")

router = APIRouter(tags=["Streaming"])

# Pre-check for OpenCV availability
try:
    import cv2 as _cv2_check  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@router.websocket("/ws/audio-stream")
async def ws_audio_stream(websocket: WebSocket):
    """Stream audio chunks for real-time breathing analysis.

    Protocol
    --------
    **Handshake** (client → server, first message, JSON text)::

        {"sample_rate": 22050, "client": "flutter"}

    **Audio chunks** (client → server, binary PCM-16 LE bytes)::

        <raw PCM bytes>

    **Server events** (server → client, JSON text)::

        {"event": "connected", "data": {...}}
        {"event": "interim",   "data": {"breathing_rate": ..., "confidence": ...}}
        {"event": "final",     "data": { <full analyze_audio result> }}
        {"event": "error",     "data": {"message": "..."}}

    **Client stop** (text JSON)::

        {"action": "stop"}
    """
    await websocket.accept()
    session = session_manager.create_session(client_info="audio-stream")
    if session is None:
        await websocket.send_text(json.dumps({
            "event": "error",
            "data": {"message": "Server at streaming capacity. Try again later."},
        }))
        await websocket.close()
        return

    sample_rate = 22050
    try:
        init_msg = await asyncio.wait_for(websocket.receive(), timeout=5.0)
        if init_msg.get("text"):
            meta = json.loads(init_msg["text"])
            sample_rate = int(meta.get("sample_rate", 22050))
    except (asyncio.TimeoutError, Exception):
        pass

    await websocket.send_text(json.dumps({
        "event": "connected",
        "data": {
            "session_id": session.session_id,
            "sample_rate": sample_rate,
            "message": "Ready to receive audio chunks (PCM-16 LE).",
        },
    }))

    audio_buffer = bytearray()
    target_bytes = sample_rate * 2 * 15  # 15 seconds of int16

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive(), timeout=20.0)
            except asyncio.TimeoutError:
                break

            if raw.get("text"):
                try:
                    ctrl = json.loads(raw["text"])
                    if ctrl.get("action") == "stop":
                        break
                except Exception:
                    pass
                continue

            chunk = raw.get("bytes")
            if not chunk:
                continue

            audio_buffer.extend(chunk)
            session_manager.record_frame(session.session_id)

            # Send interim result every ~3 seconds of audio
            if len(audio_buffer) % (sample_rate * 2 * 3) < len(chunk):
                interim = _quick_breathing_estimate(audio_buffer, sample_rate)
                await websocket.send_text(json.dumps({
                    "event": "interim",
                    "data": interim,
                }))

            if len(audio_buffer) >= target_bytes:
                break

        # Final analysis
        final = _analyse_audio_buffer(audio_buffer, sample_rate)
        await websocket.send_text(json.dumps({
            "event": "final",
            "data": final,
        }, default=str))

    except WebSocketDisconnect:
        logger.info("Audio stream: client disconnected (session=%s)", session.session_id)
    except Exception as exc:
        logger.error("Audio stream error: %s", exc)
        try:
            await websocket.send_text(json.dumps({
                "event": "error",
                "data": {"message": str(exc)},
            }))
        except Exception:
            pass
    finally:
        session_manager.close_session(session.session_id)


def _quick_breathing_estimate(buf: bytearray, sr: int) -> dict:
    """Fast breathing-rate estimate from buffered PCM for interim events."""
    try:
        samples = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32)
        samples /= 32768.0
        # Simple energy envelope
        window = int(sr * 0.4)
        if len(samples) < window * 2:
            return {"breathing_rate": None, "confidence": 0.0}

        energy = np.convolve(samples ** 2, np.ones(window) / window, mode="valid")
        # Count peaks in energy envelope
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(energy, distance=int(sr * 1.5))
        duration = len(samples) / sr
        if duration > 0 and len(peaks) > 0:
            rate = len(peaks) / duration * 60.0
            return {"breathing_rate": round(rate, 1), "confidence": 0.4}
    except Exception:
        pass
    return {"breathing_rate": None, "confidence": 0.0}


def _analyse_audio_buffer(buf: bytearray, sr: int) -> dict:
    """Run full audio analysis on the accumulated buffer."""
    import tempfile
    import os

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    samples = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0

    # Write to temp WAV and use the existing audio module
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
            if sf:
                sf.write(tmp, samples, sr)
            else:
                # Fallback: write raw WAV header manually
                import wave
                with wave.open(tmp, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes((samples * 32767).astype(np.int16).tobytes())

        from backend.app.ml.audio.audio_module import analyze_audio
        return analyze_audio(tmp)
    except Exception as exc:
        logger.error("Audio buffer analysis failed: %s", exc)
        return {"error": str(exc), "risk": "error"}
    finally:
        if tmp and os.path.isfile(tmp):
            os.remove(tmp)
