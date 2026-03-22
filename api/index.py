# Vercel serverless entrypoint
# Vercel's Python runtime looks for an ASGI/WSGI app named `app` (or `handler`)
# inside api/index.py.  We simply re-export the existing FastAPI instance so
# that no existing logic needs to change.
from backend.app.api.main import app  # noqa: F401 – re-exported for Vercel

__all__ = ["app"]
