"""Backward-compatible ASGI entry point.

Prefer: uvicorn persian_rag.main:app --app-dir src
"""

from persian_rag.main import app

__all__ = ["app"]
