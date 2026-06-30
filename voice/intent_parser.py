"""
RxPilot — Voice Intent Parser.

Parses a transcript string into a structured VoiceQuery using Claude.

The parser uses a lightweight text-only Claude call (~100-200 tokens) to:
  1. Identify the intent (stock_query, expiry_query, interaction_query, general_query)
  2. Extract any drug name mentioned in the query
  3. Produce a clean reformulation of the question

Falls back to rule-based heuristics if the Anthropic API key is not configured.

Usage:
    from voice.intent_parser import parse_intent

    query = await parse_intent("How much Metformin do we have in stock?")
    # VoiceQuery(intent="stock_query", drug_name="Metformin", question="...")
"""

from __future__ import annotations

import json
import logging
import os
import re

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from agents.state import VoiceQuery

logger = logging.getLogger(__name__)

# Claude system prompt for intent classification
_SYSTEM_PROMPT = """You are a pharmacy assistant intent parser. Given a query from a pharmacist,
extract:
1. intent: one of "stock_query" | "expiry_query" | "interaction_query" | "general_query"
   - stock_query: asking about stock levels, inventory, quantities received
   - expiry_query: asking about expiry dates, what medicines are expiring soon
   - interaction_query: asking about drug interactions, contraindications
   - general_query: any other pharmacy question
2. drug_name: the primary drug/medicine mentioned, or null if none
3. question: a clean, concise reformulation of the question (1 sentence)

Respond ONLY with valid JSON, no other text:
{"intent": "...", "drug_name": "...", "question": "..."}"""


async def parse_intent(transcript: str) -> VoiceQuery:
    """
    Parse a voice transcript into a structured VoiceQuery.

    Uses Claude for high-quality intent classification. Falls back to
    rule-based heuristics if the API is unavailable.

    Args:
        transcript: The transcribed voice query text.

    Returns:
        VoiceQuery with intent, drug_name, and question fields.
    """
    transcript = transcript.strip()

    if not transcript:
        return VoiceQuery(
            intent="general_query",
            drug_name=None,
            question="(empty transcript)",
        )

    # Try Claude first
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            result = await _claude_parse(transcript, api_key)
            if result:
                logger.info(
                    "Claude intent: %s, drug: %s",
                    result.intent, result.drug_name,
                )
                return result
        except Exception as e:
            logger.warning("Claude intent parsing failed, using heuristics: %s", e)

    # Heuristic fallback
    return _heuristic_parse(transcript)


async def _claude_parse(transcript: str, api_key: str) -> VoiceQuery | None:
    """Parse intent using Claude text API."""
    if anthropic is None:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-20250514"),  # Use cheapest model
        max_tokens=150,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )

    raw = response.content[0].text.strip()

    # Parse JSON (handle markdown fences if present)
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse Claude JSON response: %s — %s", raw, e)
        return None

    # Validate intent
    valid_intents = {"stock_query", "expiry_query", "interaction_query", "general_query"}
    intent = data.get("intent", "general_query")
    if intent not in valid_intents:
        intent = "general_query"

    return VoiceQuery(
        intent=intent,
        drug_name=data.get("drug_name") or None,
        question=data.get("question", transcript),
    )


# ── Heuristic Patterns ──

_STOCK_PATTERNS = [
    r"\b(stock|inventory|level|quantity|how much|how many|available|supply)\b",
    r"\b(stock level|in stock|on hand|current supply)\b",
]
_EXPIRY_PATTERNS = [
    r"\bexpir",          # matches: expiry, expiry, expiration, expire, expiring
    r"\bbest before\b",
    r"\bshelf life\b",
]
_INTERACTION_PATTERNS = [
    r"\binteract",       # matches: interaction, interactions, interacts, interacting
    r"\bcontraindic",    # matches: contraindicated, contraindication
    r"\b(combine|mix|together|safe to take)\b",
    r"\bdrug.drug\b",
]

_DRUG_NAME_PREFIXES = [
    r"\bof\s+([A-Z][a-zA-Z]+(?:\s+\d+\s*mg)?)",
    r"\bfor\s+([A-Z][a-zA-Z]+(?:\s+\d+\s*mg)?)",
    r"\babout\s+([A-Z][a-zA-Z]+(?:\s+\d+\s*mg)?)",
    r"\bwith\s+([A-Z][a-zA-Z]+(?:\s+\d+\s*mg)?)",
    r"\bof\s+([a-zA-Z]+(?:\s+\d+\s*mg)?)",
]


def _heuristic_parse(transcript: str) -> VoiceQuery:
    """
    Rule-based intent classification for use without Claude.

    Checks regex patterns to determine intent and extract drug name.
    """
    text = transcript.lower()

    # Determine intent
    intent = "general_query"

    if any(re.search(p, text) for p in _INTERACTION_PATTERNS):
        intent = "interaction_query"
    elif any(re.search(p, text) for p in _EXPIRY_PATTERNS):
        intent = "expiry_query"
    elif any(re.search(p, text) for p in _STOCK_PATTERNS):
        intent = "stock_query"

    # Extract drug name (heuristic: capitalized word after common prepositions)
    drug_name = None
    for pattern in _DRUG_NAME_PREFIXES:
        match = re.search(pattern, transcript)
        if match:
            candidate = match.group(1).strip()
            # Filter out common non-drug words
            stop_words = {"the", "our", "any", "some", "all", "more", "we", "I", "you"}
            if candidate.lower() not in stop_words and len(candidate) > 2:
                drug_name = candidate
                break

    logger.info(
        "Heuristic intent: %s, drug: %s (transcript: %r)",
        intent, drug_name, transcript[:60],
    )

    return VoiceQuery(
        intent=intent,
        drug_name=drug_name,
        question=transcript,
    )
