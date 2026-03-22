"""
Vita AI – Request Middleware
==============================
Logging, timing, and exception-handling middleware for the FastAPI app.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from backend.app.core.exceptions import VitaError, ValidationError, FileValidationError


def _clean_pydantic_message(loc: list, msg: str) -> str:
    """Convert a raw Pydantic validation message into a user-friendly string."""
    field = str(loc[-1]) if loc else ""
    lower = msg.lower()

    # Password-specific messages
    if field == "password":
        if "at least" in lower or "min_length" in lower or "short" in lower:
            return "Password must be at least 6 characters"
        if "too long" in lower or "max_length" in lower:
            return "Password must be at most 20 characters"
        return f"Password: {msg}"

    # Email
    if field == "email":
        if "at least" in lower or "short" in lower:
            return "Please enter a valid email address"
        if "value" in lower or "valid" in lower:
            return "Please enter a valid email address"
        return f"Email: {msg}"

    # Name
    if field == "name":
        if "at least" in lower or "short" in lower:
            return "Name must be at least 1 character"
        return f"Name: {msg}"

    # Generic fallback
    if "at least" in lower and "character" in lower:
        return f"{field.capitalize() or 'Field'} is too short"
    if "too long" in lower:
        return f"{field.capitalize() or 'Field'} is too long"
    return msg or "Invalid input"

logger = logging.getLogger("vita_api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info(
            "→ %s %s  [rid=%s]",
            request.method,
            request.url.path,
            request_id,
        )

        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception(
                "✖ %s %s  500  %.1fms  [rid=%s]",
                request.method,
                request.url.path,
                elapsed,
                request_id,
            )
            raise

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "← %s %s  %d  %.1fms  [rid=%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Attach typed exception handlers to the app."""

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error(_request: Request, exc: RequestValidationError):
        """Convert Pydantic request validation errors into clean user messages."""
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = list(first.get("loc", []))
            # Strip the leading 'body' segment FastAPI adds
            if loc and loc[0] == "body":
                loc = loc[1:]
            msg = first.get("msg", "Invalid input")
            clean = _clean_pydantic_message(loc, msg)
            return JSONResponse(
                status_code=422,
                content={"detail": clean},
            )
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request data. Please check your input."},
        )

    @app.exception_handler(ValidationError)
    async def _validation_error(_request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=400,
            content={"error": "validation_error", "detail": str(exc)},
        )

    @app.exception_handler(FileValidationError)
    async def _file_validation_error(_request: Request, exc: FileValidationError):
        return JSONResponse(
            status_code=400,
            content={"error": "file_validation_error", "detail": str(exc)},
        )

    @app.exception_handler(VitaError)
    async def _vita_error(_request: Request, exc: VitaError):
        return JSONResponse(
            status_code=500,
            content={
                "error": exc.code,
                "detail": str(exc),
            },
        )
