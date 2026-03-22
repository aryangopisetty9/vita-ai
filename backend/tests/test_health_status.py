"""
Tests for the health and status endpoints with Open-rPPG support.

Verifies that the ``/health`` endpoint includes the ``open_rppg``
status dict and that existing health fields remain intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest



class TestHealthEndpoint:
    @pytest.fixture(autouse=True)
    def _client(self):
        from fastapi.testclient import TestClient
        from backend.app.api.main import app
        self.client = TestClient(app)

    def test_health_returns_ok(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_has_open_rppg_key(self):
        resp = self.client.get("/health")
        data = resp.json()
        assert "open_rppg" in data

    def test_health_open_rppg_shape(self):
        resp = self.client.get("/health")
        orppg = resp.json()["open_rppg"]
        assert isinstance(orppg, dict)
        for key in ("installed", "loaded", "active", "model_name",
                     "supported_models", "error"):
            assert key in orppg, f"Missing open_rppg key: {key}"

    def test_health_models_key_present(self):
        resp = self.client.get("/health")
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_status_endpoint_still_works(self):
        resp = self.client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "streaming" in data

    def test_models_endpoint_still_works(self):
        resp = self.client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "rppg_models" in data
