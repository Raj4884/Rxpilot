"""
RxPilot — LangGraph StateGraph wiring.

Orchestrates the multi-agent pipeline using LangGraph's StateGraph.
Phase 1 has a single node (extraction); subsequent phases add validation,
safety, and forecast nodes with conditional routing.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.extraction_agent import run_extraction
from agents.state import PharmacyState
from observability.tracing import create_trace, traced_span

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Node functions
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

    # Update trace with final output
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


async def placeholder_node(state: dict[str, Any]) -> dict[str, Any]:
    """Placeholder node for agents not yet implemented."""
    logger.warning("Placeholder node called — this agent is not yet implemented")
    return state


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────


def route_by_input_type(state: dict[str, Any]) -> str:
    """Route to the appropriate first agent based on input type."""
    input_type = state.get("input_type", "image")
    if input_type == "voice":
        return "voice_placeholder"
    return "extract"


# ──────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the RxPilot pipeline.

    Phase 1 topology:
        START → route_by_input_type
            → "image"  → extract → END
            → "voice"  → voice_placeholder → END
    """
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("extract", extraction_node)
    graph.add_node("voice_placeholder", placeholder_node)

    # Conditional entry based on input type
    graph.add_conditional_edges(
        START,
        route_by_input_type,
        {
            "extract": "extract",
            "voice_placeholder": "voice_placeholder",
        },
    )

    # Terminal edges
    graph.add_edge("extract", END)
    graph.add_edge("voice_placeholder", END)

    return graph


# Compiled graph — import and invoke this
compiled_graph = build_graph().compile()
