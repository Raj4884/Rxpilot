"""
RxPilot — FastAPI application entry point.

Sets up the FastAPI app with CORS, lifespan handlers, route registration,
and health/root endpoints. All agent calls are traced via Langfuse from
day one.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.database import check_db_health
from observability.tracing import get_langfuse, flush_langfuse

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # ── Startup ──
    logger.info("=" * 60)
    logger.info("💊 RxPilot starting up...")
    logger.info("=" * 60)

    # Ensure upload directory exists
    upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Upload directory: %s", upload_dir.resolve())

    # Check database connectivity
    db_ok = check_db_health()
    if db_ok:
        logger.info("✅ Database connected — tables verified")
    else:
        logger.warning("⚠️  Database not ready or tables missing")

    # Initialize Langfuse
    lf = get_langfuse()
    if lf is not None:
        logger.info("✅ Langfuse tracing initialized")
    else:
        logger.warning("⚠️  Langfuse not available — tracing disabled")

    # Check Claude API key
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("✅ Anthropic API key configured")
        logger.info("   Vision model: %s", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
    else:
        logger.warning("⚠️  ANTHROPIC_API_KEY not set — extraction will fail")

    logger.info("=" * 60)
    logger.info("✅ RxPilot ready!")
    logger.info("=" * 60)

    yield

    # ── Shutdown ──
    logger.info("Shutting down RxPilot...")
    flush_langfuse()
    logger.info("Done.")


# ── Create the FastAPI app ──
app = FastAPI(
    title="RxPilot",
    description=(
        "A multi-agent AI system for pharmacy operations. "
        "Extracts structured data from photographed bills, validates against "
        "historical records, checks drug-safety risk via RAG, and forecasts "
        "reorder needs. **Portfolio project — not for real clinical use.**"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register route modules ──
from api.routes.upload import router as upload_router  # noqa: E402
from api.routes.health import router as health_router  # noqa: E402
from api.routes.voice import router as voice_router    # noqa: E402

app.include_router(upload_router, prefix="/v1")
app.include_router(health_router)
app.include_router(voice_router, prefix="/v1")

# ── Serve uploaded files (for frontend image display) ──
upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "name": "RxPilot",
        "version": "0.3.0",
        "phase": "Phase 3 — Voice Interface + CI Eval Gate",
        "description": "Multi-agent AI system for pharmacy operations",
        "disclaimer": "Portfolio project — not for real clinical use",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /v1/upload",
            "voice": "POST /v1/voice",
            "bills": "GET /v1/bills",
            "bill_detail": "GET /v1/bills/{bill_id}",
            "health": "GET /health",
        },
    }
