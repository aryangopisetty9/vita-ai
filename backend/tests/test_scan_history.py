"""
Tests for the scan history API endpoints.

Covers:
- GET  /user/{user_id}/scans  → returns list (empty if none)
- GET  /user/{user_id}/scans  → returns saved scan records
- DELETE /user/{user_id}/scan/{scan_id}  → owner can delete
- DELETE /user/{user_id}/scan/{scan_id}  → 404 for wrong owner or missing record
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool  # ensures all connections share one SQLite :memory: DB


# ---------------------------------------------------------------------------
# In-memory SQLite fixture shared by all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Build a FastAPI TestClient backed by an in-memory SQLite database.

    main.py imports `engine` by value at import time, so we patch it in
    both `database` and `main` modules before the startup event runs.
    """
    from fastapi.testclient import TestClient
    import backend.app.db.database as db_module
    import backend.app.api.main as main_module
    from backend.app.api.main import app
    from backend.app.db.database import Base, get_db

    # Build a dedicated in-memory engine for this test run.
    # StaticPool ensures every SQLAlchemy connection hits the *same*
    # in-memory SQLite database (without it each new connection gets a
    # fresh empty DB, so tables created in one connection are invisible
    # to others).
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=test_engine)

    # Patch engine in both modules so startup create_all + sessions both
    # hit the same in-memory DB.
    original_db_engine = db_module.engine
    original_main_engine = main_module.engine
    db_module.engine = test_engine
    main_module.engine = test_engine

    # Create all tables explicitly (startup event may have already cached
    # the old engine reference).
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    yield TestClient(app)

    app.dependency_overrides.clear()
    db_module.engine = original_db_engine
    main_module.engine = original_main_engine


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _register_and_get_user_id(client, email: str) -> int:
    """Register a user via POST /auth/signup and return the user id."""
    payload = {"name": "Test User", "email": email, "password": "Str0ng!Pass"}
    resp = client.post("/auth/signup", json=payload)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


def _save_scan(client, user_id: int, scan_type: str, result: dict,
               vita_score: int | None = None, risk_level: str | None = None):
    """Save a scan result via POST /user/{user_id}/scan."""
    # The endpoint reads scan_type and result as query params + form,
    # but the implementation accepts them as JSON body via /save_scan or
    # as query params on /user/{user_id}/scan.
    resp = client.post(
        f"/user/{user_id}/scan",
        params={"scan_type": scan_type},
        json=result,
    )
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScanHistoryEndpoints:
    def test_empty_history_returns_list(self, client):
        """GET /user/{user_id}/scans for a valid-but-never-scanned user → []."""
        user_id = _register_and_get_user_id(client, "hist_empty@test.com")
        resp = client.get(f"/user/{user_id}/scans")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_saved_scan_appears_in_history(self, client):
        """A scan saved via POST /user/{id}/scan shows up in GET /user/{id}/scans."""
        user_id = _register_and_get_user_id(client, "hist_save@test.com")

        save_resp = _save_scan(
            client, user_id, "face",
            {"heart_rate": 72.5, "confidence": 0.88, "vita_health_score": 80, "risk": "low"},
        )
        assert save_resp.status_code == 200, save_resp.text

        history_resp = client.get(f"/user/{user_id}/scans")
        assert history_resp.status_code == 200
        scans = history_resp.json()
        assert len(scans) == 1
        scan = scans[0]
        assert scan["scan_type"] == "face"
        assert scan["vita_score"] == 80
        assert scan["risk_level"] == "low"
        assert "id" in scan

    def test_delete_scan_removes_it(self, client):
        """DELETE /user/{user_id}/scan/{scan_id} removes the scan successfully."""
        user_id = _register_and_get_user_id(client, "hist_del@test.com")

        # Save a scan to delete
        save_resp = _save_scan(
            client, user_id, "audio",
            {"breathing_rate": 16, "vita_health_score": 70, "risk": "moderate"},
        )
        assert save_resp.status_code == 200, save_resp.text
        scan_id = save_resp.json()["id"]

        # Delete it
        del_resp = client.delete(f"/user/{user_id}/scan/{scan_id}")
        assert del_resp.status_code == 200, del_resp.text
        body = del_resp.json()
        assert body["deleted"] is True
        assert body["id"] == scan_id

        # Confirm it's gone
        history_resp = client.get(f"/user/{user_id}/scans")
        assert history_resp.status_code == 200
        assert all(s["id"] != scan_id for s in history_resp.json())

    def test_delete_nonexistent_scan_returns_404(self, client):
        """DELETE with a scan_id that doesn't exist → 404."""
        user_id = _register_and_get_user_id(client, "hist_404@test.com")
        resp = client.delete(f"/user/{user_id}/scan/999999")
        assert resp.status_code == 404

    def test_delete_wrong_owner_returns_404(self, client):
        """DELETE where user_id doesn't own the scan → 404 (ownership check)."""
        owner_id = _register_and_get_user_id(client, "hist_owner@test.com")
        other_id = _register_and_get_user_id(client, "hist_other@test.com")

        # Save scan under owner
        save_resp = _save_scan(
            client, owner_id, "symptom",
            {"score": 5, "vita_health_score": 60, "overall_risk": "high"},
        )
        assert save_resp.status_code == 200
        scan_id = save_resp.json()["id"]

        # Try to delete as other_id → should be 404
        resp = client.delete(f"/user/{other_id}/scan/{scan_id}")
        assert resp.status_code == 404

        # Scan should still exist under owner
        history = client.get(f"/user/{owner_id}/scans").json()
        assert any(s["id"] == scan_id for s in history)
