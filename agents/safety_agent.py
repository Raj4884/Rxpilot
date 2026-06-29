"""
RxPilot — Safety RAG Agent.

For each pair of drugs found on the bill, queries the drug interaction
corpus (pgvector or CSV fallback) and flags dangerous combinations.

Pipeline:
  1. Extract unique medicine names from state.extracted_fields.items
  2. Generate all drug pairs (combinations of 2)
  3. Search the RAG corpus for relevant interactions
  4. For found interactions, optionally call Claude for a human-readable
     safety assessment (lightweight text call, not vision)
  5. Populate state.safety_flags: list[SafetyFlag]

The agent degrades gracefully:
  - Without DB: uses CSV fallback search
  - Without Anthropic key: uses raw corpus data without LLM refinement
"""

from __future__ import annotations

import logging
import os
import re
from itertools import combinations
from typing import Any

from agents.state import PharmacyState, SafetyFlag
from rag import search_interactions, _search_csv_fallback

logger = logging.getLogger(__name__)


def _normalize_drug_name(name: str) -> str:
    """
    Normalize a medicine name for matching.

    Strips dosage info, common suffixes, and extra whitespace.
    'Paracetamol 500mg tablets' -> 'paracetamol'
    'Amoxicillin 250mg Capsules' -> 'amoxicillin'
    """
    # Lowercase
    name = name.lower().strip()
    # Remove dosage patterns (e.g., 500mg, 250 mg, 10ml)
    name = re.sub(r'\d+\s*(mg|ml|mcg|g|iu|units?)\b', '', name)
    # Remove common dosage forms
    name = re.sub(
        r'\b(tablets?|capsules?|syrup|injection|cream|ointment|drops?|'
        r'suspension|solution|gel|patch|inhaler|spray|powder|sachets?)\b',
        '', name
    )
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _get_unique_drug_names(state: PharmacyState) -> list[str]:
    """Extract unique normalized drug names from the extraction result."""
    if not state.extracted_fields or not state.extracted_fields.items:
        return []

    seen = set()
    names = []
    for item in state.extracted_fields.items:
        normalized = _normalize_drug_name(item.medicine_name)
        if normalized and normalized not in seen:
            seen.add(normalized)
            names.append(normalized)

    return names


def _format_safety_description(
    interaction: dict[str, Any],
    drug_a: str,
    drug_b: str,
) -> str:
    """Format a human-readable safety description from interaction data."""
    parts = [interaction.get("description", "Potential drug interaction detected.")]

    mechanism = interaction.get("mechanism", "")
    if mechanism:
        parts.append(f"Mechanism: {mechanism}")

    return " ".join(parts)


async def _llm_safety_assessment(
    interactions: list[dict[str, Any]],
    drug_names: list[str],
) -> list[dict[str, Any]]:
    """
    Optionally call Claude for a refined safety assessment.

    If ANTHROPIC_API_KEY is not set, returns the interactions as-is.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or not interactions:
        return interactions

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Build a concise prompt
        interaction_text = "\n".join(
            f"- {i['drug_a']} + {i['drug_b']} ({i['severity']}): {i['description']}"
            for i in interactions
        )

        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=500,
            system=(
                "You are a pharmacy safety assistant. Given a list of potential "
                "drug interactions found on a bill, provide a brief clinical summary "
                "for each interaction. Be concise (1-2 sentences per interaction). "
                "This is for a PORTFOLIO PROJECT demonstration, not real clinical advice."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Drugs on this bill: {', '.join(drug_names)}\n\n"
                    f"Potential interactions found:\n{interaction_text}\n\n"
                    "Provide a brief clinical summary for each."
                ),
            }],
        )

        # The LLM assessment enhances but doesn't replace the structured data
        llm_summary = response.content[0].text
        for i, interaction in enumerate(interactions):
            interaction["llm_assessment"] = llm_summary

        logger.info("LLM safety assessment completed for %d interactions", len(interactions))

    except Exception as e:
        logger.warning("LLM safety assessment failed (using raw data): %s", e)

    return interactions


async def run_safety_check(
    state: PharmacyState,
    trace: Any = None,
) -> PharmacyState:
    """
    Run the safety RAG agent on extracted bill items.

    Searches for drug interactions between all pairs of medicines found
    on the bill. Populates state.safety_flags with any interactions found.

    Args:
        state: PharmacyState with extracted_fields populated.
        trace: Langfuse trace object for observability.

    Returns:
        Updated PharmacyState with safety_flags populated.
    """
    drug_names = _get_unique_drug_names(state)

    if len(drug_names) < 2:
        logger.info(
            "Safety agent: %d drugs found — need at least 2 for interaction check",
            len(drug_names),
        )
        return state

    logger.info(
        "Safety agent starting — checking %d drugs: %s",
        len(drug_names), drug_names,
    )

    # Search for interactions
    try:
        interactions = search_interactions(drug_names)
    except Exception as e:
        logger.warning("Vector search failed, falling back to CSV: %s", e)
        interactions = _search_csv_fallback(drug_names)

    if not interactions:
        logger.info("Safety agent: no interactions found")
        return state

    logger.info(
        "Safety agent found %d potential interactions", len(interactions)
    )

    # Optionally enhance with LLM assessment
    interactions = await _llm_safety_assessment(interactions, drug_names)

    # Convert to SafetyFlag objects
    safety_flags: list[SafetyFlag] = []
    for interaction in interactions:
        try:
            severity = interaction.get("severity", "moderate")
            if severity not in ("low", "moderate", "high", "critical"):
                severity = "moderate"

            flag = SafetyFlag(
                drug_pair=(interaction["drug_a"], interaction["drug_b"]),
                severity=severity,
                description=_format_safety_description(
                    interaction,
                    interaction["drug_a"],
                    interaction["drug_b"],
                ),
                source=interaction.get("source", "RxPilot drug interaction corpus"),
            )
            safety_flags.append(flag)
        except Exception as e:
            logger.warning("Failed to create SafetyFlag: %s", e)

    state.safety_flags = safety_flags

    logger.info(
        "Safety agent complete — %d flags: %s",
        len(safety_flags),
        [(f.drug_pair, f.severity) for f in safety_flags],
    )

    return state
