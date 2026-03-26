"""
Vita AI - Pydantic Schemas

Request / response models shared by the API layer and modules.
Consistent field names across all modules:
    module_name, metric_name, value, unit, confidence, risk, message, debug
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Standardised Module Output
# ---------------------------------------------------------------------------

class ModuleResult(BaseModel):
    """Base output schema returned by every analysis module."""
    module_name: str
    metric_name: str
    value: Any
    unit: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk: str  # "low" | "moderate" | "high" | "error"
    message: str
    debug: Dict[str, Any] = {}


class FaceScanResult(BaseModel):
    """Extended result for the face scan module (v2 pipeline)."""
    module_name: str = "face_module"
    scan_duration_sec: float = 0
    heart_rate: Optional[float] = None
    heart_rate_unit: str = "bpm"
    heart_rate_confidence: float = 0.0
    blink_rate: Optional[float] = None
    blink_rate_unit: str = "blinks/min"
    eye_stability: Optional[float] = None
    facial_tension_index: Optional[float] = None
    skin_signal_stability: Optional[float] = None
    scan_quality: float = 0.0
    retake_required: bool = True
    retake_reasons: List[str] = []
    message: str = ""
    risk: str = "error"
    confidence: float = 0.0
    confidence_breakdown: Dict[str, Any] = {}
    debug: Dict[str, Any] = {}
    hr_timeseries: List[Dict[str, Any]] = []
    blink_analysis: Dict[str, Any] = {}
    eye_movement: Dict[str, Any] = {}
    facial_motion: Dict[str, Any] = {}
    eye_color: Dict[str, Any] = {}
    skin_color: Dict[str, Any] = {}
    # Legacy compat
    metric_name: str = "heart_rate"
    value: Optional[float] = None
    unit: str = "bpm"


class SymptomResult(ModuleResult):
    """Extended result for the symptom module."""
    detected_symptoms: List[str] = []


# ---------------------------------------------------------------------------
# Score Engine Output
# ---------------------------------------------------------------------------

class VitaScoreResult(BaseModel):
    """Combined health score output."""
    vita_health_score: int = Field(..., ge=0, le=100)
    overall_risk: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendations: List[str]
    component_scores: Dict[str, float]


# ---------------------------------------------------------------------------
# API Request Bodies
# ---------------------------------------------------------------------------

class SymptomRequest(BaseModel):
    """Body for POST /predict/symptom

    Accepts either free text alone (legacy) or a structured payload from the
    symptom form.  When structured fields are provided the NLP layer runs on
    ``major_symptom`` + ``minor_symptoms`` so that structural labels such as
    "Age: 25. Gender: Male." do not confuse keyword extraction.
    At least one of ``text`` or ``major_symptom`` must contain useful content.
    """
    # Free-text fallback (legacy path).  May be empty when structured fields
    # are provided.
    text: str = Field("", description="Free-text symptom description (optional when structured fields are provided)")

    # ── Structured form fields ──────────────────────────────────────────────
    major_symptom: Optional[str] = Field(None, max_length=500,
        description="Primary symptom reported by the user")
    minor_symptoms: Optional[str] = Field(None, max_length=1000,
        description="Comma-separated or free-text secondary symptoms")

    age: Optional[int] = Field(None, ge=0, le=120,
        description="Patient age in years")
    gender: Optional[str] = Field(None, max_length=20)
    days_suffering: Optional[int] = Field(None, ge=0, le=730,
        description="Number of days the patient has had these symptoms")
    symptom_category: Optional[str] = Field(None, max_length=50,
        description="Self-reported category: General, Respiratory, Digestive, Neurological, Skin")

    # ── Boolean flag toggles ────────────────────────────────────────────────
    fever: bool = Field(False, description="Patient reports fever")
    pain: bool = Field(False, description="Patient reports pain")
    difficulty_breathing: bool = Field(False,
        description="Patient reports difficulty breathing (strong red-flag signal)")

    # ── Severity slider ─────────────────────────────────────────────────────
    severity: Optional[int] = Field(None, ge=0, le=10,
        description="Self-reported severity on a 0–10 scale")


class FinalScoreRequest(BaseModel):
    """Body for POST /predict/final-score

    Accepts the raw dict outputs from each module.  Any of them can be
    ``None`` (omitted); the score engine will rebalance weights.
    """
    face_result: Optional[Dict[str, Any]] = None
    audio_result: Optional[Dict[str, Any]] = None
    symptom_result: Optional[Dict[str, Any]] = None


class FaceLiveSignalRequest(BaseModel):
    """Body for POST /predict/face-live

    Lightweight payload sent after a 30-second local camera scan.
    """
    signal: List[float] = Field(..., min_length=8)
    duration_sec: float = Field(..., gt=0, le=120)
    sampling_hz: float = Field(..., gt=0, le=30)
    frames_seen: Optional[int] = Field(None, ge=0)
    frames_processed: Optional[int] = Field(None, ge=0)
    frames_skipped: Optional[int] = Field(None, ge=0)
    brightness_mean: Optional[float] = Field(None, ge=0)


# ---------------------------------------------------------------------------
# Auth & User Schemas
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    """Body for POST /auth/signup"""
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    """Body for POST /auth/login"""
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=1)


class UpdateProfileRequest(BaseModel):
    """Body for PUT /auth/me"""
    name: str = Field(..., min_length=2, max_length=100)


class ChangePasswordRequest(BaseModel):
    """Body for POST /auth/change-password"""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)
    confirm_password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    """Public user info returned after login/signup."""
    id: int
    name: str
    email: str

    model_config = {"from_attributes": True}


class HealthDataRequest(BaseModel):
    """Body for POST /user/health-data"""
    name: Optional[str] = None
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None


class HealthDataResponse(BaseModel):
    """Response for GET /user/health-data"""
    id: int
    name: Optional[str] = None
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    bmi: Optional[float] = None
    health_score: Optional[float] = None

    model_config = {"from_attributes": True}


class ScanResultResponse(BaseModel):
    """Single scan history entry."""
    id: int
    scan_type: str
    result_json: str
    vita_score: Optional[int] = None
    risk_level: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Patient Schemas – Code contributed by Manogna
# ---------------------------------------------------------------------------

class PatientCreate(BaseModel):
    """Body for POST /patients/"""
    name: str = Field(..., min_length=1, max_length=100)
    dob: date
    gender: str = Field(..., min_length=1, max_length=20)
    phone: Optional[str] = None


class PatientUpdate(BaseModel):
    """Body for PUT /patients/{id}"""
    name: Optional[str] = None
    dob: Optional[date] = None
    gender: Optional[str] = None
    phone: Optional[str] = None


class PatientResponse(BaseModel):
    """Response for patient endpoints."""
    id: int
    name: str
    dob: date
    gender: str
    phone: Optional[str] = None

    model_config = {"from_attributes": True}


class PatientRecordResponse(BaseModel):
    """Response for prediction records."""
    id: int
    patient_id: Optional[int] = None
    name: Optional[str] = None
    symptoms: Optional[str] = None
    result: Optional[str] = None


# ---------------------------------------------------------------------------
# SOS / Emergency Schemas – SOS feature integrated from Manogna
# ---------------------------------------------------------------------------

class EmergencyContactCreate(BaseModel):
    """Body for POST /sos/contacts"""
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(..., min_length=3, max_length=50)
    relationship: Optional[str] = Field(None, max_length=50)


class EmergencyContactResponse(BaseModel):
    """Response for emergency contact endpoints."""
    id: int
    name: str
    phone: str
    relationship: Optional[str] = None
    created_at: Optional[str] = None

    model_config = {"from_attributes": True}


class SOSTriggerRequest(BaseModel):
    """Body for POST /sos/trigger"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    message: Optional[str] = Field(None, max_length=500)

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Combined Analyze Endpoint – Code contributed by Manogna
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Body for POST /analyze — runs all modules and returns combined result."""
    symptom_text: Optional[str] = None
    face_result: Optional[Dict[str, Any]] = None
    audio_result: Optional[Dict[str, Any]] = None


class LoginResponse(BaseModel):
    """Response for /auth/login with JWT token."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
