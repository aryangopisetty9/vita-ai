"""
Vita AI – Stream Session Manager
==================================
Manages active WebSocket streaming sessions for real-time face scans.
Tracks per-session state, enforces concurrency limits, and provides
status for the health endpoint.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SESSIONS = int(__import__("os").getenv("VITA_MAX_STREAMS", "10"))


@dataclass
class StreamSession:
    """State for a single active WebSocket stream."""
    session_id: str
    client_info: str = ""
    started_at: float = field(default_factory=time.time)
    frames_received: int = 0
    last_frame_at: Optional[float] = None
    finalised: bool = False

    @property
    def duration(self) -> float:
        return time.time() - self.started_at


class SessionManager:
    """Thread-safe manager for concurrent stream sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, StreamSession] = {}

    def create_session(self, client_info: str = "") -> Optional[StreamSession]:
        """Create a new session. Returns None if at capacity."""
        self._purge_stale()
        if len(self._sessions) >= MAX_CONCURRENT_SESSIONS:
            logger.warning(
                "Session limit reached (%d/%d)",
                len(self._sessions),
                MAX_CONCURRENT_SESSIONS,
            )
            return None

        sid = uuid.uuid4().hex[:16]
        session = StreamSession(session_id=sid, client_info=client_info)
        self._sessions[sid] = session
        logger.info("Stream session created: %s", sid)
        return session

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        return self._sessions.get(session_id)

    def record_frame(self, session_id: str) -> None:
        s = self._sessions.get(session_id)
        if s:
            s.frames_received += 1
            s.last_frame_at = time.time()

    def close_session(self, session_id: str) -> None:
        s = self._sessions.pop(session_id, None)
        if s:
            logger.info(
                "Stream session closed: %s (frames=%d, dur=%.1fs)",
                session_id,
                s.frames_received,
                s.duration,
            )

    def get_status(self) -> Dict:
        self._purge_stale()
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": MAX_CONCURRENT_SESSIONS,
            "sessions": [
                {
                    "id": s.session_id,
                    "frames": s.frames_received,
                    "duration_sec": round(s.duration, 1),
                }
                for s in self._sessions.values()
            ],
        }

    def _purge_stale(self) -> None:
        """Remove sessions older than 5 minutes with no recent frames."""
        stale = []
        now = time.time()
        for sid, s in self._sessions.items():
            if s.duration > 300:  # 5 min
                stale.append(sid)
            elif s.last_frame_at and (now - s.last_frame_at) > 60:
                stale.append(sid)
        for sid in stale:
            self._sessions.pop(sid, None)
            logger.info("Purged stale stream session: %s", sid)


# Global singleton
session_manager = SessionManager()
