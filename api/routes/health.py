"""
RxPilot — Health check endpoint.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from api.database import check_db_health
from observability.tracing import get_langfuse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """
    Health check — reports status of all dependencies.
    """
    db_ok = check_db_health()
    lf = get_langfuse()
    langfuse_ok = lf is not None
    claude_ok = bool(os.getenv("ANTHROPIC_API_KEY"))

    all_ok = db_ok and claude_ok
    status = "healthy" if all_ok else "degraded"

    return {
        "status": status,
        "services": {
            "database": "connected" if db_ok else "unavailable",
            "langfuse": "connected" if langfuse_ok else "unavailable",
            "claude_api": "configured" if claude_ok else "missing_key",
        },
        "vision_provider": os.getenv("VISION_PROVIDER", "claude"),
        "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
    }
