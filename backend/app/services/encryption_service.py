"""
Vita AI – Field-Level Encryption Utility

Security/encryption logic adapted from Manogna's backend.
Provides Fernet symmetric encryption for sensitive fields (PII)
stored in the database — phone numbers, addresses, etc.

The encryption key is loaded from the VITA_ENCRYPTION_KEY environment
variable.  If the variable is not set a key is auto-generated at startup
and logged as a warning (suitable for development only).

Usage::

    from backend.app.services.encryption_service import encrypt_field, decrypt_field

    encrypted = encrypt_field("555-0123")
    plain     = decrypt_field(encrypted)
"""

from __future__ import annotations

import base64
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("vita_encryption")

# ── Key management ───────────────────────────────────────────────────────
# Security/encryption logic adapted from Manogna's backend

_ENV_KEY = os.getenv("VITA_ENCRYPTION_KEY")

if _ENV_KEY:
    # Accept either raw-Fernet-base64 or a plain passphrase (we hash it)
    try:
        _FERNET = Fernet(_ENV_KEY.encode() if isinstance(_ENV_KEY, str) else _ENV_KEY)
    except Exception:
        # Treat the env value as a passphrase → derive a valid Fernet key
        import hashlib
        _derived = base64.urlsafe_b64encode(
            hashlib.sha256(_ENV_KEY.encode()).digest()
        )
        _FERNET = Fernet(_derived)
else:
    # Auto-generate for development — NOT suitable for production
    _auto_key = Fernet.generate_key()
    _FERNET = Fernet(_auto_key)
    logger.warning(
        "VITA_ENCRYPTION_KEY not set — using auto-generated key. "
        "Encrypted data will NOT survive restarts. Set the env var in production."
    )


def encrypt_field(value: str | None) -> str | None:
    """Encrypt a plaintext string. Returns base64-encoded ciphertext."""
    if value is None:
        return None
    return _FERNET.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_field(token: str | None) -> str | None:
    """Decrypt a Fernet token back to plaintext.

    Returns ``None`` if *token* is ``None`` or cannot be decrypted
    (e.g. key rotated or corrupted data).
    """
    if token is None:
        return None
    try:
        return _FERNET.decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception) as exc:
        logger.warning("Failed to decrypt field: %s", exc)
        return None
