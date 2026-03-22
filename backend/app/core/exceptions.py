"""
Vita AI – Custom Exceptions
=============================
Typed exceptions for cleaner error handling across the backend.
"""

from __future__ import annotations


class VitaError(Exception):
    """Base exception for Vita AI."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        super().__init__(message)
        self.code = code


class ValidationError(VitaError):
    """Input validation failed."""
    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR")


class ModelLoadError(VitaError):
    """A model failed to load."""
    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Failed to load {model_name}: {reason}", code="MODEL_LOAD_ERROR")
        self.model_name = model_name


class InferenceError(VitaError):
    """Inference failed."""
    def __init__(self, module: str, reason: str):
        super().__init__(f"Inference failed in {module}: {reason}", code="INFERENCE_ERROR")
        self.module = module


class FileValidationError(VitaError):
    """Uploaded file validation failed."""
    def __init__(self, message: str):
        super().__init__(message, code="FILE_VALIDATION_ERROR")
