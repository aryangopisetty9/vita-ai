"""
Vita AI – Patient CRUD Service

Full CRUD operations for patient records and prediction history.
Code contributed by Manogna — adapted from crud.py to fit Vita AI's
service layer and SQLAlchemy patterns.
"""

from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.db.db_models import Patient, PatientRecord


# ── Code contributed by Manogna ──────────────────────────────────────────


def create_patient(
    db: Session, name: str, dob: date, gender: str, phone: Optional[str] = None
) -> Patient:
    new_patient = Patient(name=name, dob=dob, gender=gender, phone=phone)
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient


def get_patients(db: Session) -> list[Patient]:
    return db.query(Patient).all()


def get_patient(db: Session, patient_id: int) -> Optional[Patient]:
    return db.query(Patient).filter(Patient.id == patient_id).first()


def update_patient(
    db: Session,
    patient_id: int,
    name: Optional[str] = None,
    dob: Optional[date] = None,
    gender: Optional[str] = None,
    phone: Optional[str] = None,
) -> Optional[Patient]:
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        return None
    if name:
        patient.name = name
    if dob:
        patient.dob = dob
    if gender:
        patient.gender = gender
    if phone is not None:
        patient.phone = phone
    db.commit()
    db.refresh(patient)
    return patient


def delete_patient(db: Session, patient_id: int) -> Optional[Patient]:
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        return None
    db.delete(patient)
    db.commit()
    return patient


def save_prediction(
    db: Session, name: str, symptoms: str, result: str, patient_id: Optional[int] = None
) -> PatientRecord:
    """Persist a prediction record, optionally linked to a patient."""
    record = PatientRecord(
        patient_id=patient_id,
        name=name,
        symptoms=symptoms,
        result=result,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
