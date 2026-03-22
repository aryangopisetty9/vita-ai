"""
Vita AI – SOS / Emergency Contact Service

SOS feature integrated from Manogna's backend (server.js /sos endpoint).
Adapted to Python/FastAPI with proper database persistence, emergency
contact management, and SOS event logging.

Provides:
- Emergency contact CRUD (save, list, delete)
- SOS trigger (log event + return contacts)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.app.db.db_models import EmergencyContact, SOSEvent

logger = logging.getLogger("vita_sos")


# ── Emergency Contact CRUD ───────────────────────────────────────────────
# SOS feature integrated from Manogna

def add_emergency_contact(
    db: Session,
    user_id: int,
    name: str,
    phone: str,
    relationship: Optional[str] = None,
) -> EmergencyContact:
    """Save a new emergency contact for a user."""
    contact = EmergencyContact(
        user_id=user_id,
        name=name,
        phone=phone,
        relationship=relationship,
    )
    db.add(contact)
    db.commit()
    db.refresh(contact)
    logger.info("Emergency contact added for user %d: %s", user_id, name)
    return contact


def get_emergency_contacts(db: Session, user_id: int) -> list[EmergencyContact]:
    """Return all emergency contacts for a user."""
    return (
        db.query(EmergencyContact)
        .filter(EmergencyContact.user_id == user_id)
        .order_by(EmergencyContact.created_at.asc())
        .all()
    )


def delete_emergency_contact(db: Session, contact_id: int, user_id: int) -> bool:
    """Delete a specific emergency contact. Returns True if deleted."""
    contact = (
        db.query(EmergencyContact)
        .filter(
            EmergencyContact.id == contact_id,
            EmergencyContact.user_id == user_id,
        )
        .first()
    )
    if not contact:
        return False
    db.delete(contact)
    db.commit()
    logger.info("Emergency contact %d deleted for user %d", contact_id, user_id)
    return True


# ── SOS Trigger ──────────────────────────────────────────────────────────
# SOS feature integrated from Manogna

def trigger_sos(
    db: Session,
    user_id: int,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    message: Optional[str] = None,
) -> dict:
    """Trigger an SOS alert: log the event and return emergency contacts.

    In a production deployment this would also fire push notifications,
    SMS (via Twilio etc.), or other alerting integrations.  For now the
    backend records the event and returns the contacts so the frontend /
    mobile layer can initiate the actual call or SMS.
    """
    # Log the SOS event
    event = SOSEvent(
        user_id=user_id,
        latitude=latitude,
        longitude=longitude,
        message=message,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    logger.warning(
        "🚨 SOS TRIGGERED — user=%d  lat=%s  lon=%s  msg=%s",
        user_id, latitude, longitude, message,
    )

    # Retrieve contacts for caller to act on
    contacts = get_emergency_contacts(db, user_id)

    return {
        "sos_event_id": event.id,
        "triggered_at": event.triggered_at.isoformat() if event.triggered_at else None,
        "contacts": [
            {
                "id": c.id,
                "name": c.name,
                "phone": c.phone,
                "relationship": c.relationship,
            }
            for c in contacts
        ],
        "message": "SOS alert recorded. Emergency contacts returned for action.",
    }
