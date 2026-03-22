"""
Vita AI – SQLAlchemy ORM Models

Database models for users and health scan history.
Integrates the teammate's HealthData model with user authentication
and full scan result persistence.
"""

from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship as sa_relationship

from backend.app.db.database import Base


class User(Base):
    """Registered user account."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    health_records = sa_relationship("HealthData", back_populates="user")
    scan_history = sa_relationship("ScanResult", back_populates="user")
    emergency_contacts = sa_relationship("EmergencyContact", back_populates="user")  # SOS feature integrated from Manogna
    sos_events = sa_relationship("SOSEvent", back_populates="user")  # SOS feature integrated from Manogna


class HealthData(Base):
    """User health profile (from teammate's Vita_AI models.py, extended)."""
    __tablename__ = "health_data"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100))
    age = Column(Integer)
    height = Column(Float)
    weight = Column(Float)
    bmi = Column(Float)
    health_score = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = sa_relationship("User", back_populates="health_records")


class ScanResult(Base):
    """Persisted scan result from any module (face, audio, symptom, final)."""
    __tablename__ = "scan_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scan_type = Column(String(50), nullable=False)  # face | audio | symptom | final
    result_json = Column(Text, nullable=False)       # full JSON blob
    vita_score = Column(Integer, nullable=True)      # only for final scans
    risk_level = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = sa_relationship("User", back_populates="scan_history")


# ── Code contributed by Manogna ──────────────────────────────────────────

class Patient(Base):
    """Patient demographic record (from Manogna's models.py)."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    dob = Column(Date, nullable=False)
    gender = Column(String(20), nullable=False)
    phone = Column(String(20), nullable=True)

    predictions = sa_relationship("PatientRecord", back_populates="patient")


class PatientRecord(Base):
    """Prediction history tied to a patient (from Manogna's models.py)."""
    __tablename__ = "patient_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    name = Column(String(100))
    symptoms = Column(String)
    result = Column(String)

    patient = sa_relationship("Patient", back_populates="predictions")


# ── SOS feature integrated from Manogna ──────────────────────────────────

class EmergencyContact(Base):
    """Emergency contact for SOS feature.

    SOS feature integrated from Manogna's backend (server.js /sos route).
    """
    __tablename__ = "emergency_contacts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    phone = Column(String(50), nullable=False)    # encrypted at rest via encryption_service
    relationship = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = sa_relationship("User", back_populates="emergency_contacts")


class SOSEvent(Base):
    """Log of SOS trigger events.

    SOS feature integrated from Manogna's backend.
    """
    __tablename__ = "sos_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    message = Column(String(500), nullable=True)
    triggered_at = Column(DateTime, default=datetime.utcnow)

    user = sa_relationship("User", back_populates="sos_events")
