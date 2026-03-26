"""
Vita AI – FastAPI Application
==============================
Exposes the Vita AI health-screening modules via REST endpoints.

Endpoints
---------
- ``GET  /``             → project info
- ``GET  /health``       → health check
- ``POST /predict/face`` → heart-rate estimation from uploaded video
- ``POST /predict/audio``  → breathing analysis from uploaded audio
- ``POST /predict/symptom`` → symptom risk from text
- ``POST /predict/final-score`` → combined Vita Health Score

Running
-------
::

    uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000 --reload

Integration notes
-----------------
* **Flutter / mobile**: Point HTTP client at these endpoints;
  use ``multipart/form-data`` for file uploads and ``application/json``
  for text / score requests.
* **Authentication**: Add an auth middleware or dependency
  (e.g. ``fastapi.security.OAuth2PasswordBearer``) when deploying.
* **Database**: Inject a DB session dependency to persist results
  (e.g. via SQLAlchemy / Supabase client).
* **Monitoring / logging**: Plug in structured logging middleware
  (e.g. ``starlette-exporter`` for Prometheus, or JSON logs) for
  production observability.
* **Real-time streaming**: Add a ``/ws/stream`` WebSocket endpoint
  that receives camera/audio chunks and streams partial results back.
* **Model registry**: Use a config flag or environment variable to
  select which model version each module loads at startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse
from typing import Any

import numpy as np

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    Request,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# are importable regardless of the working directory.

from backend.app.core.config import API_DESCRIPTION, API_TITLE, API_VERSION, CORS_ORIGINS, BASE_DIR
from backend.app.ml.audio.audio_module import analyze_audio
from backend.app.ml.face.face_module import FaceStreamProcessor, analyze_face_video
from backend.app.ml.face.live_signal import analyze_live_face_signal
from backend.app.ml.fusion.score_engine import compute_vita_score
from backend.app.ml.nlp.symptom_module import analyze_symptoms, analyze_symptoms_structured
from backend.app.ml.face.rppg_models import get_available_models as get_rppg_models
from backend.app.ml.face.open_rppg_backend import get_open_rppg_status, load_open_rppg
from backend.app.ml.audio.audio_models import get_available_models as get_audio_models
from backend.app.ml.nlp.nlp_models import get_available_models as get_nlp_models
from backend.app.ml.registry.model_registry import get_all_status as get_registry_status
from backend.app.ml.fusion.fusion_model import get_fusion_status
from backend.app.core.validation import validate_video_upload, validate_audio_upload, validate_symptom_text
from backend.app.api.middleware import RequestLoggingMiddleware, register_exception_handlers
from backend.app.api.streaming import router as streaming_router
from backend.app.services.session_manager import session_manager
from backend.app.db.schemas import FaceScanResult, FinalScoreRequest, SymptomRequest
from backend.app.db.schemas import (
    SignupRequest, LoginRequest, UserResponse,
    UpdateProfileRequest, ChangePasswordRequest,
    FaceLiveSignalRequest,
    HealthDataRequest, HealthDataResponse, ScanResultResponse,
    PatientCreate, PatientUpdate, PatientResponse, PatientRecordResponse,
    AnalyzeRequest, LoginResponse,
    EmergencyContactCreate, EmergencyContactResponse, SOSTriggerRequest,
)
from backend.app.db.database import engine, get_db, Base
from backend.app.db.db_models import User, HealthData, ScanResult, Patient, PatientRecord
from backend.app.services.auth_service import (
    hash_password, verify_password, validate_password,
    create_access_token, get_current_user,
)
from backend.app.services import patient_crud
from backend.app.services import sos_service  # SOS feature integrated from Manogna
from backend.app.services.encryption_service import encrypt_field, decrypt_field  # Security/encryption logic adapted from Manogna's backend
from sqlalchemy.orm import Session
import requests
import base64
import uuid as _uuid
from jose import jwt as jose_jwt
import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("vita_api")

# In-memory async jobs for live face scan processing.
_face_live_jobs: dict[str, dict] = {}
_face_live_jobs_lock = asyncio.Lock()

# Flag for the WebSocket handler – needs cv2 to decode JPEG frames
try:
    import cv2 as _cv2_check  # noqa: F401
    _HAS_CV2_WS = True
except ImportError:
    _HAS_CV2_WS = False

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# CORS – allow all origins by default; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Exception handlers for typed Vita errors
register_exception_handlers(app)

# Mount streaming sub-router (audio WebSocket)
app.include_router(streaming_router)

# Simple in-memory OAuth state store (production: use persistent store/redis)
_oauth_state_store: dict = {}
_OAUTH_STATE_TTL = 300  # seconds

def _set_oauth_state(
    state: str,
    provider: str,
    created_at: float | None = None,
    frontend_url: str | None = None,
) -> None:
    _oauth_state_store[state] = {
        "provider": provider,
        "created_at": created_at or time.monotonic(),
        "frontend_url": frontend_url,
    }

def _pop_oauth_state(state: str, provider: str) -> dict | None:
    entry = _oauth_state_store.get(state)
    if not entry or entry.get("provider") != provider:
        return None
    created = entry.get("created_at", 0)
    # expire stale states
    if time.monotonic() - created > _OAUTH_STATE_TTL:
        _oauth_state_store.pop(state, None)
        return None
    _oauth_state_store.pop(state, None)
    return entry


def _normalise_frontend_url(raw: str | None) -> str | None:
    """Return a safe frontend origin/path URL or None if invalid."""
    if not raw:
        return None
    val = raw.strip()
    if not val:
        return None
    parsed = urlparse(val)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/") if parsed.path and parsed.path != "/" else ""
    return base + path


def _frontend_from_request(request: Request) -> str | None:
    """Infer frontend URL from request headers for same-window OAuth return."""
    origin = request.headers.get("origin")
    parsed_origin = _normalise_frontend_url(origin)
    if parsed_origin:
        return parsed_origin
    referer = request.headers.get("referer")
    return _normalise_frontend_url(referer)


# ═══════════════════════════════════════════════════════════════════════════
# Startup: auto-download & preload
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def _on_startup():
    """Auto-download models if enabled, then preload all models."""
    from backend.app.core.config import VITA_PRELOAD_MODELS

    # Auto-download auto-downloadable models (DistilBERT, BioBERT, YAMNet)
    try:
        from backend.app.ml.registry.model_download import is_auto_download_enabled, download_all_supported
        if is_auto_download_enabled():
            logger.info("Auto-downloading supported models on startup …")
            downloaded = download_all_supported()
            if downloaded:
                logger.info("Auto-downloaded models: %s", downloaded)
    except Exception as exc:
        logger.warning("Model auto-download failed: %s", exc)

    # Refresh registry after all downloads
    try:
        from backend.app.ml.registry.model_registry import refresh_registry
        refresh_registry()
    except Exception:
        pass

    # Preload all models eagerly (default: enabled)
    if VITA_PRELOAD_MODELS:
        logger.info("VITA_PRELOAD_MODELS=true — loading all models at startup …")
        try:
            from backend.app.ml.registry.model_registry import preload_all_models
            preload_all_models()
        except Exception as exc:
            logger.warning("Model preload failed: %s", exc)

        # Log Open-rPPG status explicitly
        orppg = get_open_rppg_status()
        if orppg.get("active"):
            logger.info("Open-rPPG initialized successfully — model=%s", orppg.get("model_name"))
        else:
            logger.warning("Open-rPPG failed — using classical fallback. Error: %s",
                           orppg.get("error", "unknown"))
    else:
        logger.info("VITA_PRELOAD_MODELS=false — models will load on first request.")

    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        logger.warning("Database table creation: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

def _save_upload(upload: UploadFile, suffix: str) -> str:
    """Persist an uploaded file to a temp directory and return its path.

    The caller is responsible for cleanup.
    """
    tmp_dir = Path(__file__).resolve().parents[4] / "temp"
    tmp_dir.mkdir(exist_ok=True)
    filename = f"{uuid.uuid4().hex}{suffix}"
    filepath = tmp_dir / filename
    contents = upload.file.read()
    filepath.write_bytes(contents)
    return str(filepath)


def _cleanup(path: str) -> None:
    """Remove a temporary file if it exists."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

# Path to the Flutter web build output — used by several route handlers below.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "build" / "web"

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/", tags=["General"], include_in_schema=False)
def root():
    """Serve Flutter web index.html with no-cache headers."""
    index = _FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(str(index), headers=_NO_CACHE_HEADERS)
    return {
        "project": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
    }


@app.get("/index.html", include_in_schema=False)
def index_html():
    """Explicit /index.html route with no-cache headers."""
    f = _FRONTEND_DIR / "index.html"
    if f.is_file():
        return FileResponse(str(f), headers=_NO_CACHE_HEADERS)
    raise HTTPException(status_code=404)


@app.get("/flutter_service_worker.js", include_in_schema=False)
def service_worker():
    """Service worker with no-cache so browsers always fetch the latest version."""
    f = _FRONTEND_DIR / "flutter_service_worker.js"
    if f.is_file():
        return FileResponse(
            str(f),
            media_type="application/javascript",
            headers=_NO_CACHE_HEADERS,
        )
    raise HTTPException(status_code=404)


@app.get("/flutter_bootstrap.js", include_in_schema=False)
def bootstrap_js():
    """Flutter bootstrap JS with no-cache headers."""
    f = _FRONTEND_DIR / "flutter_bootstrap.js"
    if f.is_file():
        return FileResponse(
            str(f),
            media_type="application/javascript",
            headers=_NO_CACHE_HEADERS,
        )
    raise HTTPException(status_code=404)


@app.get('/__force_update', include_in_schema=False)
def force_update():
        """Serve a tiny page that unregisters service workers and reloads —
        useful to force the browser to drop cached Flutter service workers
        and fetch the latest `index.html`/assets from the backend.
        """
        html = '''<!doctype html>
<html>
    <head><meta charset="utf-8"><title>Force update</title></head>
    <body>
        <div style="font-family:sans-serif;padding:24px;">Forcing update… If this page hangs, reload manually.</div>
        <script>
            (async function(){
                try {
                    if ('serviceWorker' in navigator) {
                        const regs = await navigator.serviceWorker.getRegistrations();
                        for (const r of regs) {
                            try { await r.unregister(); } catch(e) {}
                        }
                    }
                } catch(e) {}
                // Navigate to root to fetch latest index.html
                window.location.href = '/';
            })();
        </script>
    </body>
</html>'''
        return HTMLResponse(content=html, status_code=200, headers=_NO_CACHE_HEADERS)


@app.get("/api", tags=["General"])
def api_info():
    """Project info and available endpoints."""
    return {
        "project": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "endpoints": [
            "GET  /api",
            "GET  /health",
            "GET  /status",
            "GET  /models",
            "POST /auth/signup",
            "POST /auth/login",
            "GET  /auth/me",
            "POST /predict/face",
            "POST /predict/video",
            "POST /predict/face-live",
            "POST /predict/audio",
            "POST /predict/symptom",
            "POST /predict/final-score",
            "POST /analyze",
            "POST /patients/",
            "GET  /patients/",
            "GET  /patients/{id}",
            "PUT  /patients/{id}",
            "DELETE /patients/{id}",
            "POST /user/{id}/health-data",
            "GET  /user/{id}/health-data",
            "POST /user/{id}/scan",
            "GET  /user/{id}/scans",
            "POST /sos/contacts/{user_id}",
            "GET  /sos/contacts/{user_id}",
            "DELETE /sos/contacts/{user_id}/{contact_id}",
            "POST /sos/trigger/{user_id}",
            "WS   /ws/face-scan",
            "WS   /ws/audio-stream",
        ],
    }


# ── OAuth helpers and endpoints for Google / Facebook / Apple
def _oauth_debug_enabled() -> bool:
    return os.getenv("VITA_OAUTH_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}


def _oauth_log(message: str, **kwargs) -> None:
    if not _oauth_debug_enabled():
        return
    safe = {}
    for k, v in kwargs.items():
        if any(s in k.lower() for s in ("secret", "token", "key", "password")):
            safe[k] = "***"
        else:
            safe[k] = v
    logger.info("[OAuth] %s | %s", message, safe)


def _load_env_kv_file(path: Path) -> None:
    """Minimal KEY=VALUE loader for local .env fallback."""
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.getenv(key):
                os.environ[key] = value
    except OSError:
        pass


def _ensure_oauth_env_loaded() -> None:
    """Load local .env files on-demand if OAuth vars are not in process env."""
    required = (
        "VITA_GOOGLE_CLIENT_ID",
        "VITA_GOOGLE_CLIENT_SECRET",
        "VITA_GOOGLE_REDIRECT_URI",
        "VITA_FRONTEND_URL",
    )
    if all(os.getenv(k) for k in required):
        return
    _load_env_kv_file(BASE_DIR / ".env")
    _load_env_kv_file(BASE_DIR / "backend" / ".env")


def _google_client_id() -> str | None:
    return os.getenv("VITA_GOOGLE_CLIENT_ID") or os.getenv("GOOGLE_CLIENT_ID")


def _google_client_secret() -> str | None:
    return os.getenv("VITA_GOOGLE_CLIENT_SECRET") or os.getenv("GOOGLE_CLIENT_SECRET")


def _google_redirect_uri() -> str:
    return (
        os.getenv("VITA_GOOGLE_REDIRECT_URI")
        or os.getenv("GOOGLE_REDIRECT_URI")
        or "http://localhost:8000/auth/oauth/google/callback"
    )


def _looks_like_placeholder(value: str | None) -> bool:
    if not value:
        return True
    v = value.strip().lower()
    return (
        not v
        or "paste_" in v
        or "your_" in v
        or "_here" in v
        or "example" in v
        or "changeme" in v
    )


def _frontend_target() -> str | None:
    """Frontend URL that OAuth callbacks should redirect to (optional).

    Set the environment variable `VITA_FRONTEND_URL` in deployments to the
    public URL of the frontend (e.g. https://app.example.com). If not set,
    the callback handler will return a small HTML page that stores the JWT
    in localStorage and redirects to '/'.
    """
    frontend_env = os.getenv("VITA_FRONTEND_URL") or os.getenv("FRONTEND_URL")
    return _normalise_frontend_url(frontend_env) or "http://localhost:8000"


def _make_html_token_response(token: str, redirect_to: str | None = None, error: str | None = None) -> HTMLResponse:
    """Return a tiny HTML page that stores the token in localStorage and
    redirects the browser to the frontend.
    """
    target = redirect_to or "/"
    if token:
        joiner = "&" if "?" in target else "?"
        target = f"{target}{joiner}token={quote(token)}"
    if error:
        joiner = "&" if "?" in target else "?"
        target = f"{target}{joiner}oauth_error={quote(error)}"
    html = f"""<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Signing you in…</title></head>
  <body>
    <script>
      try {{
                if ('{token}') {{
                    localStorage.setItem('vita_token', '{token}');
                }}
      }} catch(e) {{}}
      window.location.href = '{target}';
    </script>
    <noscript>
      <p>Sign-in successful. Navigate to <a href="{target}">{target}</a></p>
    </noscript>
  </body>
</html>"""
    return HTMLResponse(content=html, status_code=200)


def _oauth_error_response(message: str, status_code: int = 400, frontend: str | None = None):
    frontend = _normalise_frontend_url(frontend) or _frontend_target()
    if frontend:
        dest = frontend.rstrip('/') + f'/?oauth_error={quote(message)}'
        return RedirectResponse(dest, status_code=302)
    return _make_html_token_response('', redirect_to='/', error=message)


def _get_or_create_user_by_email(db: Session, email: str, name: str):
    """Find an existing user by email or create a new one with a random
    password hash so that the account is usable by the rest of the system.
    """
    user = db.query(User).filter(User.email == email).first()
    if user:
        if name and user.name != name:
            user.name = name
            db.commit()
            db.refresh(user)
        return user

    # Create a user with a random password hash
    random_pw = _uuid.uuid4().hex
    user = User(name=name or 'User', email=email, password_hash=hash_password(random_pw))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get('/auth/oauth/{provider}/login', include_in_schema=False)
def oauth_login(provider: str, request: Request):
    """Redirect the user to the provider's authorization page.

    Supported providers: google, facebook, apple
    """
    provider = provider.lower().strip()
    _ensure_oauth_env_loaded()
    state = _uuid.uuid4().hex
    frontend_hint = _frontend_from_request(request) or _frontend_target()
    _set_oauth_state(state, provider, frontend_url=frontend_hint)
    _oauth_log("oauth start", provider=provider, state=state)

    if provider == 'google':
        client_id = _google_client_id()
        client_secret = _google_client_secret()
        configured_redirect = _google_redirect_uri()
        if _looks_like_placeholder(client_id) or _looks_like_placeholder(client_secret):
            missing = [n for n, v in (
                ('VITA_GOOGLE_CLIENT_ID', client_id),
                ('VITA_GOOGLE_CLIENT_SECRET', client_secret),
            ) if _looks_like_placeholder(v)]
            return _oauth_error_response(
                f'Missing required Google OAuth env vars: {", ".join(missing)}',
                status_code=500,
                frontend=frontend_hint,
            )

        params = {
            'client_id': client_id,
            'redirect_uri': configured_redirect,
            'response_type': 'code',
            'scope': 'openid email profile',
            'prompt': 'select_account',
            'access_type': 'offline',
            'include_granted_scopes': 'true',
            'state': state,
        }
        auth_url = 'https://accounts.google.com/o/oauth2/v2/auth'

    elif provider == 'facebook':
        client_id = os.getenv('VITA_FACEBOOK_CLIENT_ID')
        client_secret = os.getenv('VITA_FACEBOOK_CLIENT_SECRET')
        configured_redirect = os.getenv('VITA_FACEBOOK_REDIRECT_URI') or 'http://localhost:8000/auth/oauth/facebook/callback'
        if not client_id or not client_secret:
            missing = [n for n, v in (
                ('VITA_FACEBOOK_CLIENT_ID', client_id),
                ('VITA_FACEBOOK_CLIENT_SECRET', client_secret),
            ) if not v]
            raise HTTPException(status_code=500, detail=f'Missing required Facebook OAuth env vars: {", ".join(missing)}')

        params = {
            'client_id': client_id,
            'redirect_uri': configured_redirect,
            'response_type': 'code',
            'scope': 'email',
            'state': state,
        }
        auth_url = 'https://www.facebook.com/v12.0/dialog/oauth'

    elif provider == 'apple':
        client_id = os.getenv('VITA_APPLE_CLIENT_ID')
        configured_redirect = os.getenv('VITA_APPLE_REDIRECT_URI') or 'http://localhost:8000/auth/oauth/apple/callback'
        if not client_id:
            missing = [n for n, v in (
                ('VITA_APPLE_CLIENT_ID', client_id),
            ) if not v]
            raise HTTPException(status_code=500, detail=f'Missing required Apple OAuth env vars: {", ".join(missing)}')

        params = {
            'client_id': client_id,
            'redirect_uri': configured_redirect,
            'response_type': 'code',
            'scope': 'name email',
            'response_mode': 'form_post',
            'state': state,
        }
        auth_url = 'https://appleid.apple.com/auth/authorize'

    else:
        raise HTTPException(status_code=400, detail='Unsupported OAuth provider')

    url = auth_url + '?' + urlencode(params)
    _oauth_log("oauth redirect", provider=provider, redirect_uri=params.get('redirect_uri'))
    logger.info('OAuth %s login URL generated', provider)
    return RedirectResponse(url)
@app.api_route('/auth/oauth/{provider}/callback', methods=['GET','POST'], name='oauth_callback', include_in_schema=False)
async def oauth_callback(provider: str, request: Request, db: Session = Depends(get_db)):
    """Handle provider callback: exchange code for token, fetch user info,
    create/login local user, and return a token to the frontend.
    """
    provider = provider.lower()
    _ensure_oauth_env_loaded()
    # Accept code/state from query params (GET) or form (POST, e.g. Apple)
    if request.method == 'POST':
        form = await request.form()
        code = form.get('code') or request.query_params.get('code')
        state = form.get('state') or request.query_params.get('state')
    else:
        code = request.query_params.get('code')
        state = request.query_params.get('state')

    if not code:
        return _oauth_error_response('Missing code from provider', status_code=400)

    # Validate CSRF state
    state_entry = _pop_oauth_state(state, provider) if state else None
    if not state or not state_entry:
        return _oauth_error_response('Invalid or missing OAuth state parameter', status_code=400)

    callback_frontend = _normalise_frontend_url((state_entry or {}).get("frontend_url")) or _frontend_target()

    _oauth_log("oauth callback received", provider=provider, has_code=bool(code), has_state=bool(state))

    try:
        if provider == 'google':
            # Ensure required env vars are present and use configured redirect URI
            client_id = _google_client_id()
            client_secret = _google_client_secret()
            configured = _google_redirect_uri()
            if _looks_like_placeholder(client_id) or _looks_like_placeholder(client_secret):
                missing = [n for n, v in (
                    ('VITA_GOOGLE_CLIENT_ID', client_id),
                    ('VITA_GOOGLE_CLIENT_SECRET', client_secret),
                ) if _looks_like_placeholder(v)]
                return _oauth_error_response(
                    f'Missing required Google OAuth env vars: {", ".join(missing)}',
                    status_code=500,
                    frontend=callback_frontend,
                )

            # Exchange code for token (async)
            async with httpx.AsyncClient(timeout=10.0) as client:
                token_resp = await client.post(
                    'https://oauth2.googleapis.com/token',
                    data={
                        'code': code,
                        'client_id': client_id,
                        'client_secret': client_secret,
                        'redirect_uri': configured,
                        'grant_type': 'authorization_code',
                    },
                )
                token_resp.raise_for_status()
                token_json = token_resp.json()
                access_token = token_json.get('access_token')
                if not access_token:
                    logger.warning('Google token exchange failed: %s', token_json)
                    return _oauth_error_response('Failed to obtain access token from Google', status_code=400, frontend=callback_frontend)
                userinfo_resp = await client.get('https://www.googleapis.com/oauth2/v2/userinfo', headers={'Authorization': f'Bearer {access_token}'})
                userinfo_resp.raise_for_status()
                userinfo = userinfo_resp.json()
                email = userinfo.get('email')
                name = userinfo.get('name') or userinfo.get('given_name') or (email.split('@')[0] if email else 'GoogleUser')
                picture = userinfo.get('picture')
                _oauth_log(
                    "google userinfo received",
                    provider=provider,
                    email_present=bool(email),
                    name_present=bool(name),
                    picture_present=bool(picture),
                )

        elif provider == 'facebook':
            fb_client_id = os.getenv('VITA_FACEBOOK_CLIENT_ID')
            fb_client_secret = os.getenv('VITA_FACEBOOK_CLIENT_SECRET')
            fb_redirect = os.getenv('VITA_FACEBOOK_REDIRECT_URI') or 'http://localhost:8000/auth/oauth/facebook/callback'
            if not fb_client_id or not fb_client_secret:
                missing = [n for n, v in (
                    ('VITA_FACEBOOK_CLIENT_ID', fb_client_id),
                    ('VITA_FACEBOOK_CLIENT_SECRET', fb_client_secret),
                ) if not v]
                raise HTTPException(status_code=500, detail=f'Missing required Facebook OAuth env vars: {", ".join(missing)}')
            async with httpx.AsyncClient(timeout=10.0) as client:
                token_resp = await client.get('https://graph.facebook.com/v12.0/oauth/access_token', params={
                    'client_id': fb_client_id,
                    'client_secret': fb_client_secret,
                    'redirect_uri': fb_redirect,
                    'code': code,
                })
                token_resp.raise_for_status()
                token_json = token_resp.json()
                access_token = token_json.get('access_token')
                if not access_token:
                    logger.warning('Facebook token exchange failed: %s', token_json)
                    return _oauth_error_response('Failed to obtain access token from Facebook', status_code=400)
                userinfo_resp = await client.get('https://graph.facebook.com/me', params={'access_token': access_token, 'fields': 'id,name,email'})
                userinfo_resp.raise_for_status()
                userinfo = userinfo_resp.json()
                email = userinfo.get('email')
                name = userinfo.get('name') or (email.split('@')[0] if email else 'FacebookUser')
                _oauth_log("facebook userinfo received", provider=provider, email_present=bool(email), name_present=bool(name))

        elif provider == 'apple':
            client_id = os.getenv('VITA_APPLE_CLIENT_ID')
            team_id = os.getenv('VITA_APPLE_TEAM_ID')
            key_id = os.getenv('VITA_APPLE_KEY_ID')
            private_key = os.getenv('VITA_APPLE_PRIVATE_KEY')
            configured_redirect = os.getenv('VITA_APPLE_REDIRECT_URI') or 'http://localhost:8000/auth/oauth/apple/callback'
            if not (client_id and team_id and key_id and private_key and configured_redirect):
                missing = [n for n, v in (
                    ('VITA_APPLE_CLIENT_ID', client_id),
                    ('VITA_APPLE_TEAM_ID', team_id),
                    ('VITA_APPLE_KEY_ID', key_id),
                    ('VITA_APPLE_PRIVATE_KEY', private_key),
                ) if not v]
                raise HTTPException(status_code=500, detail=f'Missing Apple OAuth env vars: {", ".join(missing)}')

            now = int(datetime.utcnow().timestamp())
            client_secret = jose_jwt.encode(
                {
                    'iss': team_id,
                    'iat': now,
                    'exp': now + 86400 * 180,
                    'aud': 'https://appleid.apple.com',
                    'sub': client_id,
                },
                private_key,
                algorithm='ES256',
                headers={'kid': key_id},
            )

            async with httpx.AsyncClient(timeout=10.0) as client:
                token_resp = await client.post('https://appleid.apple.com/auth/token', data={
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'code': code,
                    'grant_type': 'authorization_code',
                    'redirect_uri': configured_redirect,
                })
                token_resp.raise_for_status()
                token_json = token_resp.json()
                access_token = token_json.get('access_token')
                id_token = token_json.get('id_token')
                if id_token:
                    try:
                        claims = jose_jwt.decode(id_token, options={"verify_signature": False})
                        email = claims.get('email')
                        name = claims.get('name') or (email.split('@')[0] if email else 'AppleUser')
                    except Exception:
                        email = None
                        name = 'AppleUser'
                else:
                    email = None
                    name = 'AppleUser'
                _oauth_log("apple token processed", provider=provider, email_present=bool(email), has_id_token=bool(id_token))

        else:
            raise HTTPException(status_code=400, detail='Unsupported OAuth provider')

        if not email:
            # Email is required by the application to create a local user
            return _oauth_error_response('OAuth provider did not return an email address', status_code=400, frontend=callback_frontend)

        # Create or find user
        user = _get_or_create_user_by_email(db, email=email, name=name)
        token = create_access_token({"sub": user.email})
        _oauth_log("oauth local auth complete", provider=provider, user_id=user.id, email=user.email)

        # Redirect to frontend or return HTML that stores token in localStorage
        frontend = callback_frontend
        if frontend:
            dest = frontend.rstrip('/') + f'/?token={token}'
            return RedirectResponse(dest)

        return _make_html_token_response(token)

    except HTTPException as exc:
        return _oauth_error_response(str(exc.detail), status_code=exc.status_code, frontend=callback_frontend)
    except httpx.HTTPStatusError as exc:
        logger.warning('OAuth %s HTTP error: %s', provider, exc)
        return _oauth_error_response(f'OAuth HTTP error from {provider}', status_code=400, frontend=callback_frontend)
    except Exception as exc:
        logger.exception('OAuth callback handling failed: %s', exc)
        return _oauth_error_response('OAuth processing failed', status_code=500, frontend=callback_frontend)


@app.get('/auth/config', tags=['Auth'])
def auth_config():
    """Return non-sensitive auth provider configuration status for diagnostics.

    Does not return secrets. Shows whether required env vars are present and
    the configured redirect URIs.
    """
    _ensure_oauth_env_loaded()
    google_configured = (
        not _looks_like_placeholder(_google_client_id())
        and not _looks_like_placeholder(_google_client_secret())
    )
    cfg = {
        'google_configured': google_configured,
        'google': {
            'configured': google_configured,
            'redirect_uri': _google_redirect_uri(),
        },
        'facebook': {
            'configured': all([os.getenv('VITA_FACEBOOK_CLIENT_ID'), os.getenv('VITA_FACEBOOK_CLIENT_SECRET')]),
            'redirect_uri': os.getenv('VITA_FACEBOOK_REDIRECT_URI') or 'http://localhost:8000/auth/oauth/facebook/callback',
        },
        'apple': {
            'configured': all([os.getenv('VITA_APPLE_CLIENT_ID'), os.getenv('VITA_APPLE_TEAM_ID'), os.getenv('VITA_APPLE_KEY_ID'), os.getenv('VITA_APPLE_PRIVATE_KEY')]),
            'redirect_uri': os.getenv('VITA_APPLE_REDIRECT_URI') or 'http://localhost:8000/auth/oauth/apple/callback',
        },
        'frontend_url': _frontend_target(),
    }
    return cfg


@app.get("/health", tags=["General"])
def health_check():
    """Liveness probe with model status summary."""
    try:
        from backend.app.ml.registry.model_status import get_all_model_status
        model_status = get_all_model_status()
    except Exception:
        model_status = {}
    # Add Open-rPPG status at the top level for easy visibility
    open_rppg = get_open_rppg_status()
    # MediaPipe presence
    try:
        import importlib
        mediapipe_installed = importlib.util.find_spec("mediapipe") is not None
    except Exception:
        mediapipe_installed = False
    # Fusion model status (honest reporting)
    fusion = get_fusion_status()
    return {
        "status": "ok",
        "open_rppg": open_rppg,
        "models": model_status,
        "mediapipe": {"installed": mediapipe_installed},
        "fusion": fusion,
    }


@app.get("/status", tags=["General"])
def detailed_status():
    """Readiness probe with model and streaming status."""
    try:
        from backend.app.ml.registry.model_status import get_all_model_status
        runtime_status = get_all_model_status()
    except Exception:
        runtime_status = {}
    return {
        "status": "ok",
        "models": get_registry_status(),
        "runtime_model_status": runtime_status,
        "fusion": get_fusion_status(),
        "streaming": session_manager.get_status(),
    }


@app.get("/models", tags=["General"])
def list_models():
    """List all available pretrained models per module."""
    fusion = get_fusion_status()
    return {
        "rppg_models": get_rppg_models(),
        "audio_models": get_audio_models(),
        "nlp_models": get_nlp_models(),
        "score_fusion": fusion.get("method", "weighted_sum"),
        "fusion_detail": fusion,
        "note": (
            "Models listed under each module are those whose weights were "
            "found and loaded successfully. Missing models fall back to "
            "the baseline signal-processing pipeline automatically."
        ),
    }


# ── Face ──────────────────────────────────────────────────────────────────

async def _predict_face_from_upload(file: UploadFile) -> JSONResponse:
    """Shared heavy video-processing path for face scan endpoints."""
    """Upload a short (20-30s) face video for comprehensive health screening.

    Returns heart rate estimate, behavioral features (blink rate, eye
    stability, facial tension), skin signal analysis, explainable
    confidence scores, and retake guidance if scan quality is low.

    Accepts common video formats (mp4, avi, mov, mkv, webm).

    Integration notes
    -----------------
    * Flutter: use ``http.MultipartRequest`` with field name ``file``.
    * Check ``retake_required`` – if true, prompt the user to rescan.
    * Render ``hr_timeseries`` as a line chart for heart rate over time.
    * Display ``confidence_breakdown`` for transparency.
    * Store full JSON for longitudinal tracking.
    """
    err = validate_video_upload(file)
    if err:
        raise HTTPException(status_code=400, detail=err)

    print(f"Video received: {file.filename}")

    ext = Path(file.filename or "").suffix.lower() or ".mp4"
    path = _save_upload(file, ext)
    try:
        result = analyze_face_video(path)
        print(f"Result: {result}")
    finally:
        _cleanup(path)

    # Keep full rich payload and add stable aliases for video clients.
    response_payload = dict(result)
    response_payload.setdefault("bpm", result.get("heart_rate"))
    response_payload.setdefault("stress", result.get("stress", result.get("risk")))
    response_payload.setdefault(
        "confidence", result.get("confidence", result.get("hr_confidence", 0.0))
    )
    response_payload.setdefault(
        "quality",
        result.get("scan_quality", result.get("signal_strength", 0.0)),
    )
    return JSONResponse(content=response_payload)


@app.post("/predict/face", tags=["Prediction"])
async def predict_face(file: UploadFile = File(...)):
    """Backward-compatible face video endpoint."""
    return await _predict_face_from_upload(file)


@app.post("/predict/video", tags=["Prediction"])
async def predict_video(file: UploadFile = File(...)):
    """Primary video-first face scan endpoint for stable record-then-process flow."""
    return await _predict_face_from_upload(file)


async def _run_face_live_job(job_id: str, payload: dict) -> None:
    """Run heavy live-signal analysis in background without blocking API thread."""
    t_total_start = time.perf_counter()
    try:
        logger.info(
            "[FaceLiveJob] started job_id=%s signal_len=%s duration_sec=%.2f sampling_hz=%.3f",
            job_id,
            len(payload.get("signal") or []),
            float(payload.get("duration_sec") or 0.0),
            float(payload.get("sampling_hz") or 0.0),
        )

        t_analysis_start = time.perf_counter()
        result = await asyncio.to_thread(
            analyze_live_face_signal,
            signal=payload.get("signal") or [],
            duration_sec=float(payload.get("duration_sec") or 0.0),
            sampling_hz=float(payload.get("sampling_hz") or 0.0),
            brightness_mean=payload.get("brightness_mean"),
            frames_seen=payload.get("frames_seen"),
            frames_processed=payload.get("frames_processed"),
            frames_skipped=payload.get("frames_skipped"),
        )
        t_analysis = time.perf_counter() - t_analysis_start

        total_elapsed = time.perf_counter() - t_total_start
        async with _face_live_jobs_lock:
            job = _face_live_jobs.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = result
                job["error"] = None
                job["analysis_sec"] = round(float(t_analysis), 4)
                job["processing_sec"] = round(float(total_elapsed), 4)
                job["updated_at"] = datetime.utcnow().isoformat()

        logger.info(
            "[FaceLiveJob] completed job_id=%s bpm=%s analysis_sec=%.4f total_sec=%.4f",
            job_id,
            result.get("heart_rate"),
            float(t_analysis),
            float(total_elapsed),
        )
    except Exception as exc:
        total_elapsed = time.perf_counter() - t_total_start
        async with _face_live_jobs_lock:
            job = _face_live_jobs.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["result"] = None
                job["processing_sec"] = round(float(total_elapsed), 4)
                job["updated_at"] = datetime.utcnow().isoformat()
        logger.exception("[FaceLiveJob] failed job_id=%s after %.4fs: %s", job_id, total_elapsed, exc)


@app.post("/predict/face-live", tags=["Prediction"])
async def predict_face_live(body: FaceLiveSignalRequest, background_tasks: BackgroundTasks):
    """Submit live face signal for background processing and return a job id."""
    signal_len = len(body.signal)
    logger.info("[FaceLive] received signal length: %s", signal_len)

    if signal_len < 8:
        raise HTTPException(status_code=400, detail="Signal is too short for live analysis")

    job_id = uuid.uuid4().hex
    now_iso = datetime.utcnow().isoformat()
    payload = body.model_dump()

    async with _face_live_jobs_lock:
        _face_live_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "result": None,
            "error": None,
            "created_at": now_iso,
            "updated_at": now_iso,
            "analysis_sec": None,
            "processing_sec": None,
            "signal_len": signal_len,
        }

    logger.info("[FaceLive] job created job_id=%s", job_id)
    background_tasks.add_task(_run_face_live_job, job_id, payload)
    return JSONResponse(content={"job_id": job_id, "status": "processing"})


@app.get("/predict/face-live/job-status/{job_id}", tags=["Prediction"])
async def predict_face_live_job_status(job_id: str):
    """Poll status for a previously submitted live face processing job."""
    async with _face_live_jobs_lock:
        job = _face_live_jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Live face job not found")

    payload = {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "processing_sec": job["processing_sec"],
        "analysis_sec": job["analysis_sec"],
    }
    if job["status"] == "done":
        payload["result"] = job["result"]
    elif job["status"] == "failed":
        payload["error"] = job["error"] or "Live face processing failed"
    return JSONResponse(content=payload)


# ── Audio ─────────────────────────────────────────────────────────────────

@app.post("/predict/audio", tags=["Prediction"])
async def predict_audio(file: UploadFile = File(...)):
    """Upload a breathing audio recording for respiratory analysis.

    Accepts wav, mp3, ogg, flac, m4a.
    """
    err = validate_audio_upload(file)
    if err:
        raise HTTPException(status_code=400, detail=err)

    ext = Path(file.filename or "").suffix.lower() or ".wav"
    path = _save_upload(file, ext)
    try:
        result = analyze_audio(path)
    finally:
        _cleanup(path)

    return JSONResponse(content=result)


# ── Symptom ───────────────────────────────────────────────────────────────

@app.post("/predict/symptom", tags=["Prediction"])
async def predict_symptom(body: SymptomRequest):
    """Analyse symptom inputs using a hybrid NLP + structured-field engine.

    Accepts either:
    - Legacy: ``{"text": "I have fever and cough"}``
    - Structured: full form payload with ``major_symptom``, ``minor_symptoms``,
      ``severity``, ``fever``, ``pain``, ``difficulty_breathing``, ``age``,
      ``gender``, ``days_suffering``, ``symptom_category``
    - Mixed: both text and structured fields

    When ``major_symptom`` or ``minor_symptoms`` are provided the NLP layer
    runs on those fields (not on the concatenated label string), giving
    better symptom extraction.  All structured fields are then used to
    compute score adjustments above and beyond the text-only risk tier.
    """
    has_structured_text = bool(
        (body.major_symptom and body.major_symptom.strip()) or
        (body.minor_symptoms and body.minor_symptoms.strip())
    )
    has_structured_flags = body.fever or body.pain or body.difficulty_breathing

    # Text validation: only required when structured fields are absent
    if not has_structured_text and not has_structured_flags:
        err = validate_symptom_text(body.text)
        if err:
            raise HTTPException(status_code=400, detail=err)

    result = analyze_symptoms_structured(
        text=body.text,
        major_symptom=body.major_symptom,
        minor_symptoms=body.minor_symptoms,
        age=body.age,
        gender=body.gender,
        days_suffering=body.days_suffering,
        symptom_category=body.symptom_category,
        fever=body.fever,
        pain=body.pain,
        difficulty_breathing=body.difficulty_breathing,
        severity=body.severity,
    )
    return JSONResponse(content=result)


# ── WebSocket – Real-time face scan ──────────────────────────────────────

@app.websocket("/ws/face-scan")
async def ws_face_scan(websocket: WebSocket):
    """Stream real-time face-scan results over a WebSocket connection.

    Protocol
    --------
    **Handshake** (client → server, first message, JSON)::

        {"fps": 30, "client": "flutter"}  # optional metadata

    **Frame messages** (client → server, binary JPEG bytes)::

        <raw JPEG bytes>

    **Server events** (server → client, JSON text)::

        {"event": "connected",  "data": {"message": "..."}}
        {"event": "interim",   "data": {"heart_rate": 72.3, "confidence": 0.61, ...}}
        {"event": "final",     "data": { <full analyze_face_video-compatible dict> }}
        {"event": "error",     "data": {"message": "..."}}

    **Client close signal** (optional, JSON text)::

        {"action": "stop"}

    Integration notes (Flutter / web)
    ----------------------------------
    * Use ``web_socket_channel`` (Flutter) or the browser ``WebSocket`` API.
    * Send JPEG-compressed camera frames as ``Uint8List`` bytes.
    * Render ``interim.heart_rate`` live on a gauge.
    * On ``final`` event, display the full scan result and close the socket.
    * If ``final.retake_required`` is true, prompt the user to rescan.
    * ``progress_pct`` on ``interim`` events drives a progress bar (0-99).
    * The server auto-finalises after ~25 seconds; clients may also send
      ``{"action": "stop"}`` to request early finalisation.
    """
    await websocket.accept()
    logger.info("WS /ws/face-scan: client connected from %s", websocket.client)

    # ── Negotiate FPS with the client ────────────────────────────────────
    fps = 30.0
    try:
        init_msg = await asyncio.wait_for(websocket.receive(), timeout=5.0)
        if init_msg.get("type") == "websocket.receive":
            raw = init_msg.get("text") or (init_msg.get("bytes") or b"").decode()
            if raw:
                meta = json.loads(raw)
                fps = float(meta.get("fps", 30.0))
    except (asyncio.TimeoutError, Exception):
        pass  # no handshake → use default fps

    await websocket.send_text(json.dumps({
        "event": "connected",
        "data": {
            "message": "Ready to receive JPEG frames.",
            "fps": fps,
            "target_dur_sec": 25,
            "interim_every_n_frames": 15,
        },
    }))

    # ── Set up the stream processor ──────────────────────────────────────
    proc = FaceStreamProcessor(fps=fps)

    try:
        while not proc.is_done:
            try:
                raw_msg = await asyncio.wait_for(websocket.receive(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("WS /ws/face-scan: client timed out (no frame for 10 s)")
                break

            # Client may send a stop signal as text JSON
            if raw_msg.get("type") == "websocket.receive" and raw_msg.get("text"):
                try:
                    ctrl = json.loads(raw_msg["text"])
                    if ctrl.get("action") == "stop":
                        logger.info("WS /ws/face-scan: client requested early stop.")
                        break
                except Exception:
                    pass
                continue

            # Binary frame bytes
            frame_bytes = raw_msg.get("bytes")
            if not frame_bytes:
                continue

            # Decode JPEG → BGR numpy
            if not _HAS_CV2_WS:
                await websocket.send_text(json.dumps({
                    "event": "error",
                    "data": {"message": "OpenCV not installed on server."},
                }))
                return

            import cv2 as _cv2  # already checked above
            buf = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            if frame is None:
                logger.debug("WS /ws/face-scan: could not decode frame, skipping.")
                continue

            event = proc.push_frame(frame)
            if event is not None:
                await websocket.send_text(json.dumps(event, default=str))
                if event["event"] == "final":
                    return

    except WebSocketDisconnect:
        logger.info("WS /ws/face-scan: client disconnected.")
    except Exception as exc:
        logger.error("WS /ws/face-scan: unexpected error: %s", exc)
        try:
            await websocket.send_text(json.dumps({
                "event": "error",
                "data": {"message": str(exc)},
            }))
        except Exception:
            pass
    finally:
        # Force-finalise if the client disconnected before we sent the final result
        if not proc.is_done or proc._finalised_result is None:
            final_data = proc.finalise()
            try:
                await websocket.send_text(json.dumps(
                    {"event": "final", "data": final_data}, default=str
                ))
            except Exception:
                pass
        proc.close()
        logger.info("WS /ws/face-scan: session closed.")


# ── Final Score ───────────────────────────────────────────────────────────

@app.post("/predict/final-score", tags=["Prediction"])
async def predict_final_score(body: FinalScoreRequest):
    """Compute the combined **Vita Health Score**.

    Accepts optional module results from face, audio, and symptom
    endpoints.  Any field can be ``null`` / omitted — the score engine
    automatically rebalances weights.

    Body example::

        {
          "face_result":    { ... },
          "audio_result":   { ... },
          "symptom_result": { ... }
        }

    Integration notes
    -----------------
    * Flutter: collect outputs from the three individual endpoints and
      POST them here as a single JSON body.
    * Dashboard: display the ``vita_health_score`` gauge and
      ``component_scores`` breakdown.
    """
    result = compute_vita_score(
        face_result=body.face_result,
        audio_result=body.audio_result,
        symptom_result=body.symptom_result,
    )
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════════════════════════
# Auth Endpoints – Code contributed by Manogna (JWT + bcrypt upgrade)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/auth/signup", tags=["Auth"], response_model=UserResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    """Register a new user account (bcrypt-hashed password)."""
    # Password validation – Code contributed by Manogna
    valid, message = validate_password(body.password)
    if not valid:
        raise HTTPException(status_code=400, detail=message)

    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", tags=["Auth"], response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and return a JWT access token."""
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # JWT token creation – Code contributed by Manogna
    token = create_access_token({"sub": user.email})
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(id=user.id, name=user.name, email=user.email),
    )


# ═══════════════════════════════════════════════════════════════════════════
# User Health Data Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/user/{user_id}/health-data", tags=["User"], response_model=HealthDataResponse)
def save_health_data(user_id: int, body: HealthDataRequest, db: Session = Depends(get_db)):
    """Create or update health profile for a user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    record = db.query(HealthData).filter(HealthData.user_id == user_id).first()
    bmi = None
    if body.height and body.weight and body.height > 0:
        bmi = round(body.weight / ((body.height / 100) ** 2), 2)

    if record:
        if body.name is not None:
            record.name = body.name
        if body.age is not None:
            record.age = body.age
        if body.height is not None:
            record.height = body.height
        if body.weight is not None:
            record.weight = body.weight
        if bmi is not None:
            record.bmi = bmi
    else:
        record = HealthData(
            user_id=user_id,
            name=body.name,
            age=body.age,
            height=body.height,
            weight=body.weight,
            bmi=bmi,
        )
        db.add(record)

    db.commit()
    db.refresh(record)
    return record


@app.get("/user/{user_id}/health-data", tags=["User"], response_model=HealthDataResponse)
def get_health_data(user_id: int, db: Session = Depends(get_db)):
    """Retrieve health profile for a user."""
    record = db.query(HealthData).filter(HealthData.user_id == user_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="No health data found")
    return record


# ═══════════════════════════════════════════════════════════════════════════
# Scan History Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/user/{user_id}/scan", tags=["User"])
def save_scan_result(
    user_id: int,
    scan_type: str,
    result: dict,
    db: Session = Depends(get_db),
):
    """Persist a scan result for a user (called internally or by client)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    vita_score = result.get("vita_health_score")
    risk_level = result.get("overall_risk") or result.get("risk")

    record = ScanResult(
        user_id=user_id,
        scan_type=scan_type,
        result_json=json.dumps(result, default=str),
        vita_score=vita_score,
        risk_level=risk_level,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "saved": True}


@app.get("/user/{user_id}/scans", tags=["User"], response_model=list[ScanResultResponse])
def get_scan_history(user_id: int, db: Session = Depends(get_db)):
    """Get all scan results for a user, newest first."""
    records = (
        db.query(ScanResult)
        .filter(ScanResult.user_id == user_id)
        .order_by(ScanResult.created_at.desc())
        .all()
    )
    for r in records:
        if r.created_at:
            r.created_at = r.created_at.isoformat()
    return records


@app.delete("/user/{user_id}/scan/{scan_id}", tags=["User"])
def delete_scan_result(user_id: int, scan_id: int, db: Session = Depends(get_db)):
    """Delete a single scan result owned by the given user.

    Returns 404 if the scan does not exist or belongs to a different user
    (ownership check prevents cross-user deletion).
    """
    record = (
        db.query(ScanResult)
        .filter(ScanResult.id == scan_id, ScanResult.user_id == user_id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Scan not found or access denied")
    db.delete(record)
    db.commit()
    return {"deleted": True, "id": scan_id}


# ═══════════════════════════════════════════════════════════════════════════
# Patient CRUD Endpoints – Code contributed by Manogna
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/patients/", tags=["Patients"], response_model=PatientResponse)
def create_patient_endpoint(body: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient record. Code contributed by Manogna."""
    return patient_crud.create_patient(
        db, body.name, body.dob, body.gender, body.phone
    )


@app.get("/patients/", tags=["Patients"], response_model=list[PatientResponse])
def get_patients_endpoint(db: Session = Depends(get_db)):
    """List all patients. Code contributed by Manogna."""
    return patient_crud.get_patients(db)


@app.get("/patients/{patient_id}", tags=["Patients"], response_model=PatientResponse)
def get_patient_endpoint(patient_id: int, db: Session = Depends(get_db)):
    """Get a single patient by ID. Code contributed by Manogna."""
    p = patient_crud.get_patient(db, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p


@app.put("/patients/{patient_id}", tags=["Patients"], response_model=PatientResponse)
def update_patient_endpoint(
    patient_id: int, body: PatientUpdate, db: Session = Depends(get_db)
):
    """Update a patient record. Code contributed by Manogna."""
    updated = patient_crud.update_patient(
        db, patient_id, body.name, body.dob, body.gender, body.phone
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Patient not found")
    return updated


@app.delete("/patients/{patient_id}", tags=["Patients"])
def delete_patient_endpoint(patient_id: int, db: Session = Depends(get_db)):
    """Delete a patient record. Code contributed by Manogna."""
    deleted = patient_crud.delete_patient(db, patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"detail": "Patient deleted successfully"}


# ═══════════════════════════════════════════════════════════════════════════
# Auth – Protected Profile Endpoint
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/auth/me", tags=["Auth"], response_model=UserResponse)
def get_current_profile(
    current_email: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the authenticated user's profile.

    Requires a valid JWT Bearer token.  Adapted from Manogna's protected
    route pattern (add_patient dependency).
    """
    # Code integrated from Manogna — protected route pattern
    user = db.query(User).filter(User.email == current_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.put("/auth/me", tags=["Auth"], response_model=UserResponse)
def update_current_profile(
    body: UpdateProfileRequest,
    current_email: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update authenticated user's profile fields supported by current schema."""
    user = db.query(User).filter(User.email == current_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    new_name = body.name.strip()
    if len(new_name) < 2:
        raise HTTPException(status_code=400, detail="Name must be at least 2 characters")

    user.name = new_name
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/change-password", tags=["Auth"])
def change_password(
    body: ChangePasswordRequest,
    current_email: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change password for authenticated user with current-password verification."""
    user = db.query(User).filter(User.email == current_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    if body.new_password != body.confirm_password:
        raise HTTPException(status_code=400, detail="New password and confirm password do not match")

    if body.current_password == body.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from current password")

    valid, message = validate_password(body.new_password)
    if not valid:
        raise HTTPException(status_code=400, detail=message)

    user.password_hash = hash_password(body.new_password)
    db.commit()
    return {"detail": "Password updated successfully"}


# ═══════════════════════════════════════════════════════════════════════════
# SOS / Emergency Contact Endpoints – SOS feature integrated from Manogna
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/sos/contacts/{user_id}",
    tags=["SOS"],
    response_model=EmergencyContactResponse,
)
def add_sos_contact(
    user_id: int,
    body: EmergencyContactCreate,
    db: Session = Depends(get_db),
):
    """Save an emergency contact for a user.

    SOS feature integrated from Manogna's backend.
    The phone number is encrypted at rest via Fernet encryption.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Security/encryption logic adapted from Manogna's backend
    encrypted_phone = encrypt_field(body.phone)

    contact = sos_service.add_emergency_contact(
        db, user_id, body.name, encrypted_phone, body.relationship,
    )
    # Return decrypted phone to caller
    contact.phone = body.phone
    if contact.created_at:
        contact.created_at = contact.created_at.isoformat()
    return contact


@app.get(
    "/sos/contacts/{user_id}",
    tags=["SOS"],
    response_model=list[EmergencyContactResponse],
)
def list_sos_contacts(user_id: int, db: Session = Depends(get_db)):
    """Return all emergency contacts for a user.

    SOS feature integrated from Manogna.
    Phone numbers are decrypted before returning.
    """
    contacts = sos_service.get_emergency_contacts(db, user_id)
    results = []
    for c in contacts:
        results.append(EmergencyContactResponse(
            id=c.id,
            name=c.name,
            phone=decrypt_field(c.phone) or c.phone,  # decrypt; fallback to raw if key mismatch
            relationship=c.relationship,
            created_at=c.created_at.isoformat() if c.created_at else None,
        ))
    return results


@app.delete("/sos/contacts/{user_id}/{contact_id}", tags=["SOS"])
def remove_sos_contact(
    user_id: int,
    contact_id: int,
    db: Session = Depends(get_db),
):
    """Delete a specific emergency contact.

    SOS feature integrated from Manogna.
    """
    deleted = sos_service.delete_emergency_contact(db, contact_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"detail": "Emergency contact deleted"}


@app.post("/sos/trigger/{user_id}", tags=["SOS"])
def trigger_sos_alert(
    user_id: int,
    body: SOSTriggerRequest,
    db: Session = Depends(get_db),
):
    """Trigger an SOS alert: log the event and return emergency contacts.

    SOS feature integrated from Manogna's backend (server.js /sos route).
    Records the SOS event with optional GPS coordinates and a message,
    then returns the user's emergency contacts so the frontend / mobile
    layer can initiate actual calls or SMS.

    In production, this could also fire push notifications or Twilio SMS.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    result = sos_service.trigger_sos(
        db, user_id, body.latitude, body.longitude, body.message,
    )

    # Decrypt phone numbers before returning to caller
    for contact in result.get("contacts", []):
        contact["phone"] = decrypt_field(contact["phone"]) or contact["phone"]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Combined Analyze Endpoint – Code contributed by Manogna
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/analyze", tags=["Prediction"])
async def analyze_endpoint(body: AnalyzeRequest):
    """Run all available analysis modules and return a combined Vita Health Score.

    Code contributed by Manogna — adapted to use Vita AI's sophisticated
    ML pipeline (BioBERT/DistilBERT, YAMNet, Open-rPPG, fusion engine)
    instead of the simpler face_scan/audio_reco/symptoms modules.

    Accepts pre-computed face and audio results (from individual endpoints)
    and an optional symptom text. Runs symptom analysis on-the-fly if text
    is provided, then feeds everything into the fusion score engine.

    Body example::

        {
            "symptom_text": "headache and mild fever",
            "face_result": { ... },
            "audio_result": { ... }
        }
    """
    symptom_result = None
    if body.symptom_text:
        symptom_result = analyze_symptoms(body.symptom_text)

    score = compute_vita_score(
        face_result=body.face_result,
        audio_result=body.audio_result,
        symptom_result=symptom_result,
    )

    return JSONResponse(content={
        "face_result": body.face_result,
        "audio_result": body.audio_result,
        "symptom_result": symptom_result,
        "vita_score": score,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Serve Flutter web frontend (same-origin → no CORS issues)
# ═══════════════════════════════════════════════════════════════════════════
# _FRONTEND_DIR is defined near the top of the Endpoints section above.


class _SPAStaticFiles(StaticFiles):
    """StaticFiles subclass that returns index.html for any unknown path.

    Standard StaticFiles raises HTTP 404 for files that don't exist.
    This subclass catches those 404s and serves ``index.html`` instead,
    which is the standard SPA fallback pattern.  All actual static assets
    (JS, CSS, images, …) are still served normally.

    FastAPI registered routes (``/health``, ``/predict/*``, etc.) are matched
    *before* any mount, so API endpoints are never shadowed.
    """

    async def get_response(self, path: str, scope: Any) -> Any:
        from fastapi import HTTPException as _HTTPException

        try:
            return await super().get_response(path, scope)
        except _HTTPException as exc:
            if exc.status_code == 404:
                # SPA fallback: serve index.html for unknown frontend routes
                return await super().get_response("index.html", scope)
            raise


if _FRONTEND_DIR.is_dir():
    # Mount Flutter build output at /.
    #   • Static assets   → served directly by _SPAStaticFiles
    #   • Unknown routes  → index.html (SPA fallback for Flutter path routing)
    #   • API endpoints   → handled by FastAPI routes above (take priority)
    app.mount("/", _SPAStaticFiles(directory=str(_FRONTEND_DIR), html=True), name="flutter_web")
else:
    logger.warning("Frontend build not found at %s – UI will not be served", _FRONTEND_DIR)
