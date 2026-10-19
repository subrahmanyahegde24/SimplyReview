"""Entrypoint for Vercel (``main:app``) and local dev (``uv run uvicorn main:app --reload``)."""

from simplyreview.api import app

__all__ = ["app"]
