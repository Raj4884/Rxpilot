"""
RxPilot — Voice Query API Endpoint.

POST /v1/voice
  Accepts: audio file (WAV, MP3, WEBM, OGG, FLAC) up to 25MB
  Returns: transcript, intent, drug_name, answer, processing metadata

The endpoint runs the full voice pipeline:
  1. Save audio file to disk
  2. Transcribe with Whisper (or stub)
  3. Parse intent with Claude Haiku (or heuristics)
  4. Answer query from DB/corpus/LLM
  5. Return structured JSON response
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from agents.state import PharmacyState
from graph import compiled_graph
from observability.tracing import flush_langfuse

logger = logging.getLogger(__name__)

router = APIRouter()

VOICE_UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")) / "voice"
VOICE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOICE_SIZE = 25 * 1024 * 1024  # 25MB
ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "audio/x-flac",
    "audio/m4a",
    "audio/mp4",
    "video/webm",   # Chrome sometimes sends WebM audio with video MIME
}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a"}


# ── Response Model ──

class VoiceResponse(BaseModel):
    """Response from the voice query endpoint."""
    trace_id: str
    transcript: str
    intent: str
    drug_name: str | None = None
    answer: str
    answer_source: str
    answer_confidence: float
    processing_time_ms: float
    error: str | None = None


# ── Endpoint ──

@router.post("/voice", response_model=VoiceResponse, summary="Submit voice query")
async def voice_query(
    file: UploadFile = File(..., description="Audio file (WAV/MP3/WEBM/OGG/FLAC, max 25MB)"),
):
    """
    Submit a voice query about pharmacy operations.

    The audio is transcribed with Whisper, the intent is parsed by Claude,
    and the query is answered using database records and the drug interaction corpus.

    **This is a portfolio demonstration — not for real clinical decisions.**
    """
    start_time = time.monotonic()
    file_id = str(uuid.uuid4())[:8]
    trace_id = str(uuid.uuid4())

    # ── Validate file ──
    filename = file.filename or f"audio_{file_id}.wav"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS and file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported audio format: {ext or file.content_type}. "
                f"Supported: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Save audio file ──
    safe_name = f"{file_id}_{Path(filename).stem[:40]}{ext or '.wav'}"
    file_path = VOICE_UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        if len(content) > MAX_VOICE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Maximum size is 25MB.",
            )
        with open(file_path, "wb") as f_out:
            f_out.write(content)
        logger.info("Voice file saved: %s (%d bytes)", safe_name, len(content))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save voice file: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # ── Run voice pipeline via LangGraph ──
    initial_state = PharmacyState(
        input_type="voice",
        raw_input_ref=str(file_path),
        trace_id=trace_id,
    )

    try:
        result = await compiled_graph.ainvoke(initial_state.model_dump())
    except Exception as e:
        logger.error("Voice pipeline failed: %s", e)
        elapsed = (time.monotonic() - start_time) * 1000
        return VoiceResponse(
            trace_id=trace_id,
            transcript="",
            intent="general_query",
            answer="The voice pipeline encountered an error. Please try again.",
            answer_source="stub",
            answer_confidence=0.0,
            processing_time_ms=round(elapsed, 2),
            error=str(e),
        )
    finally:
        flush_langfuse()

    # ── Parse result ──
    elapsed_ms = (time.monotonic() - start_time) * 1000
    transcript = result.get("transcript") or ""
    voice_query_data = result.get("voice_query") or {}
    voice_answer_data = result.get("voice_answer") or {}
    error = result.get("error")

    return VoiceResponse(
        trace_id=trace_id,
        transcript=transcript,
        intent=voice_query_data.get("intent", "general_query"),
        drug_name=voice_query_data.get("drug_name"),
        answer=voice_answer_data.get("answer_text", "No answer produced."),
        answer_source=voice_answer_data.get("source", "stub"),
        answer_confidence=voice_answer_data.get("confidence", 0.0),
        processing_time_ms=round(elapsed_ms, 2),
        error=error,
    )
