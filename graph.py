"""
RxPilot — LangGraph StateGraph wiring.

Orchestrates the multi-agent pipeline using LangGraph's StateGraph.

Phase 3 topology:
    START → route_by_input_type
        → "image" → extract → validate → safety → END
        → "voice" → transcribe → parse_intent → answer → END
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.extraction_agent import run_extraction
from agents.validation_agent import run_validation
from agents.safety_agent import run_safety_check
from agents.state import PharmacyState
from voice.transcription import transcribe_audio
from voice.intent_parser import parse_intent
from voice.answer_agent import answer_query
from observability.tracing import create_trace, traced_span

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Image pipeline nodes
# ──────────────────────────────────────────────


async def extraction_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: run the extraction agent on a bill image.

    LangGraph passes state as a dict; we convert to/from PharmacyState.
    """
    pharmacy_state = PharmacyState(**state)
    trace = create_trace(
        trace_id=pharmacy_state.trace_id,
        name="rxpilot-pipeline",
        input_data={
            "input_type": pharmacy_state.input_type,
            "raw_input_ref": pharmacy_state.raw_input_ref,
        },
    )

    with traced_span(trace, "extraction-agent", input_data={"image": pharmacy_state.raw_input_ref}) as ctx:
        result = await run_extraction(pharmacy_state, trace=trace)
        ctx["output"] = {
            "items_count": len(result.extracted_fields.items) if result.extracted_fields else 0,
            "error": result.error,
        }
        ctx["metadata"]["processing_time_ms"] = result.processing_time_ms
        ctx["metadata"]["estimated_cost_usd"] = result.estimated_cost_usd

    try:
        trace.update(
            output={
                "items_count": len(result.extracted_fields.items) if result.extracted_fields else 0,
                "processing_time_ms": result.processing_time_ms,
                "estimated_cost_usd": result.estimated_cost_usd,
                "status": "error" if result.error else "success",
            },
        )
    except Exception as e:
        logger.warning("Failed to update Langfuse trace: %s", e)

    return result.model_dump()


async def validation_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: run the validation agent on extracted items.

    Checks for duplicate batches, expired medicines, price anomalies,
    date inconsistencies, and missing critical fields.
    """
    pharmacy_state = PharmacyState(**state)
    trace = create_trace(
        trace_id=pharmacy_state.trace_id,
        name="rxpilot-pipeline",
    )

    with traced_span(trace, "validation-agent", input_data={
        "items_count": len(pharmacy_state.extracted_fields.items) if pharmacy_state.extracted_fields else 0,
    }) as ctx:
        result = await run_validation(pharmacy_state, trace=trace)
        ctx["output"] = {
            "validation_flags": result.validation_flags,
            "flags_count": len(result.validation_flags),
        }

    logger.info(
        "Validation node complete — %d flags",
        len(result.validation_flags),
    )

    return result.model_dump()


async def safety_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: run the safety RAG agent on extracted items.

    Searches drug interaction corpus and flags dangerous combinations.
    """
    pharmacy_state = PharmacyState(**state)
    trace = create_trace(
        trace_id=pharmacy_state.trace_id,
        name="rxpilot-pipeline",
    )

    with traced_span(trace, "safety-agent", input_data={
        "items_count": len(pharmacy_state.extracted_fields.items) if pharmacy_state.extracted_fields else 0,
    }) as ctx:
        result = await run_safety_check(pharmacy_state, trace=trace)
        ctx["output"] = {
            "safety_flags": [
                {"drug_pair": f.drug_pair, "severity": f.severity}
                for f in result.safety_flags
            ],
            "flags_count": len(result.safety_flags),
        }

    logger.info(
        "Safety node complete — %d flags",
        len(result.safety_flags),
    )

    return result.model_dump()


# ──────────────────────────────────────────────
# Voice pipeline nodes
# ──────────────────────────────────────────────


async def transcribe_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: transcribe audio to text using Whisper.

    Reads raw_input_ref (audio file path) → populates transcript.
    """
    pharmacy_state = PharmacyState(**state)
    trace = create_trace(
        trace_id=pharmacy_state.trace_id,
        name="rxpilot-voice-pipeline",
        input_data={"audio_file": pharmacy_state.raw_input_ref},
    )

    start = time.monotonic()

    with traced_span(trace, "transcription", input_data={"file": pharmacy_state.raw_input_ref}) as ctx:
        try:
            transcript = transcribe_audio(pharmacy_state.raw_input_ref)
            pharmacy_state = PharmacyState(
                **{**pharmacy_state.model_dump(), "transcript": transcript}
            )
            ctx["output"] = {"transcript": transcript, "length": len(transcript)}
            logger.info(
                "Transcription complete: %d chars in %.1fs",
                len(transcript), time.monotonic() - start,
            )
        except Exception as e:
            logger.error("Transcription node failed: %s", e)
            pharmacy_state = PharmacyState(
                **{**pharmacy_state.model_dump(), "error": f"Transcription failed: {e}"}
            )
            ctx["output"] = {"error": str(e)}

    return pharmacy_state.model_dump()


async def parse_intent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: parse transcript into structured VoiceQuery.
    """
    pharmacy_state = PharmacyState(**state)

    if pharmacy_state.error or not pharmacy_state.transcript:
        logger.warning("Skipping intent parsing — no transcript available")
        return pharmacy_state.model_dump()

    trace = create_trace(
        trace_id=pharmacy_state.trace_id,
        name="rxpilot-voice-pipeline",
    )

    with traced_span(trace, "intent-parser", input_data={"transcript": pharmacy_state.transcript}) as ctx:
        voice_query = await parse_intent(pharmacy_state.transcript)
        pharmacy_state = PharmacyState(
            **{**pharmacy_state.model_dump(), "voice_query": voice_query.model_dump()}
        )
        ctx["output"] = {
            "intent": voice_query.intent,
            "drug_name": voice_query.drug_name,
        }
        logger.info(
            "Intent parsed: %s (drug: %s)",
            voice_query.intent, voice_query.drug_name,
        )

    return pharmacy_state.model_dump()


async def answer_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: answer the parsed voice query.
    """
    pharmacy_state = PharmacyState(**state)

    if pharmacy_state.error or not pharmacy_state.voice_query:
        logger.warning("Skipping answer node — no voice query available")
        return pharmacy_state.model_dump()

    trace = create_trace(
        trace_id=pharmacy_state.trace_id,
        name="rxpilot-voice-pipeline",
    )

    with traced_span(trace, "answer-agent", input_data={
        "intent": pharmacy_state.voice_query.intent,
        "drug_name": pharmacy_state.voice_query.drug_name,
    }) as ctx:
        voice_answer = await answer_query(pharmacy_state.voice_query)
        pharmacy_state = PharmacyState(
            **{**pharmacy_state.model_dump(), "voice_answer": voice_answer.model_dump()}
        )
        ctx["output"] = {
            "answer": voice_answer.answer_text[:100],
            "source": voice_answer.source,
            "confidence": voice_answer.confidence,
        }
        logger.info(
            "Answer produced (source=%s, confidence=%.2f)",
            voice_answer.source, voice_answer.confidence,
        )

    return pharmacy_state.model_dump()


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────


def route_by_input_type(state: dict[str, Any]) -> str:
    """Route to the appropriate first agent based on input type."""
    input_type = state.get("input_type", "image")
    if input_type == "voice":
        return "transcribe"
    return "extract"


# ──────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the RxPilot pipeline.

    Phase 3 topology:
        START → route_by_input_type
            → "image" → extract → validate → safety → END
            → "voice" → transcribe → parse_intent → answer → END
    """
    graph = StateGraph(dict)

    # Image pipeline nodes
    graph.add_node("extract", extraction_node)
    graph.add_node("validate", validation_node)
    graph.add_node("safety", safety_node)

    # Voice pipeline nodes
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("answer", answer_node)

    # Conditional entry based on input type
    graph.add_conditional_edges(
        START,
        route_by_input_type,
        {
            "extract": "extract",
            "transcribe": "transcribe",
        },
    )

    # Image pipeline: extract → validate → safety → END
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "safety")
    graph.add_edge("safety", END)

    # Voice pipeline: transcribe → parse_intent → answer → END
    graph.add_edge("transcribe", "parse_intent")
    graph.add_edge("parse_intent", "answer")
    graph.add_edge("answer", END)

    return graph


# Compiled graph — import and invoke this
compiled_graph = build_graph().compile()
