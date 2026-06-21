"""
RxPilot — Extraction Agent.

Uses Claude's vision API to extract structured pharmacy bill data from
photographed paper bills. Handles multi-language bills (Hindi/Marathi/English),
validates output against the ExtractedBillFields pydantic schema, and retries
once on parse failure before failing the node.

Every call is traced via Langfuse (input image ref, output fields, token usage,
latency, estimated cost).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import anthropic

from agents.state import ExtractionResult, ExtractedBillFields, PharmacyState
from observability.tracing import traced_generation

logger = logging.getLogger(__name__)

# ── Constants ──
MAX_RETRIES = 2
# Approximate pricing for Claude Sonnet (per 1M tokens, USD)
INPUT_COST_PER_M = 3.0
OUTPUT_COST_PER_M = 15.0

EXTRACTION_SYSTEM_PROMPT = """You are RxPilot, an AI assistant specialized in extracting structured data from photographed pharmacy bills and prescriptions.

TASK: Analyze the provided bill/prescription image and extract ALL medicine line items with their details.

RULES:
1. Extract every medicine/drug line item visible on the bill.
2. For each item, extract as many of these fields as you can read:
   - medicine_name (REQUIRED): Full name of the medicine/drug
   - batch_number: Batch or lot number
   - expiry_date: Expiry date in YYYY-MM-DD format
   - manufacture_date: Manufacturing date in YYYY-MM-DD format
   - quantity: Number of units
   - unit: Unit of measure (tablets, strips, ml, etc.)
   - supplier_name: Supplier, distributor, or manufacturer name
   - price: Price (numeric value only, no currency symbol)
   - currency: Currency code (default "INR")
3. Handle multi-language bills: text may be in English, Hindi (हिंदी), or Marathi (मराठी), or a mix.
4. If a field is not visible or illegible, set it to null.
5. Dates may appear in various formats (DD/MM/YYYY, MM/YYYY, etc.) — normalize to YYYY-MM-DD.
6. For partially legible text, include your best reading with a note.

OUTPUT FORMAT: Return ONLY valid JSON matching this exact schema:
{
  "items": [
    {
      "medicine_name": "string (required)",
      "batch_number": "string or null",
      "expiry_date": "YYYY-MM-DD or null",
      "manufacture_date": "YYYY-MM-DD or null",
      "quantity": integer_or_null,
      "unit": "string or null",
      "supplier_name": "string or null",
      "price": number_or_null,
      "currency": "INR"
    }
  ]
}

Return an empty items array if no medicines are found.
Do NOT include any explanation or markdown — return ONLY the JSON object."""


def _get_client() -> anthropic.Anthropic:
    """Create an Anthropic client from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Set it in your .env file to use the extraction agent."
        )
    return anthropic.Anthropic(api_key=api_key)


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read and base64-encode an image file. Returns (base64_data, media_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def _parse_extraction_response(raw_text: str) -> list[dict[str, Any]]:
    """Parse the LLM JSON response, handling markdown fences."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
        else:
            raise

    # Handle both {"items": [...]} and bare [...]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data:
        return data["items"]

    raise ValueError(f"Unexpected response shape: {type(data)}")


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts."""
    return (
        (input_tokens / 1_000_000) * INPUT_COST_PER_M
        + (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    )


async def run_extraction(
    state: PharmacyState,
    trace: Any = None,
) -> PharmacyState:
    """
    Run the extraction agent on a bill image.

    Sends the image to Claude's vision API, parses the structured output,
    validates against the pydantic schema, and retries once on failure.

    Args:
        state: Current PharmacyState with input_type="image" and raw_input_ref set.
        trace: Langfuse trace object for observability.

    Returns:
        Updated PharmacyState with extracted_fields populated.
    """
    image_path = state.raw_input_ref
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    client = _get_client()

    logger.info("Extraction agent starting — image=%s, model=%s", image_path, model)

    last_error = ""
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.perf_counter()

    for attempt in range(MAX_RETRIES):
        try:
            # Encode image (inside retry loop so FileNotFoundError is caught)
            image_data, media_type = _encode_image(image_path)

            # Build messages
            user_content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all medicine line items from this pharmacy bill image."
                        if attempt == 0
                        else (
                            f"Previous extraction attempt failed: {last_error}\n"
                            "Please fix the issue and return valid JSON matching the schema."
                        )
                    ),
                },
            ]

            # Create Langfuse generation span
            generation = traced_generation(
                trace=trace,
                name=f"extraction-attempt-{attempt + 1}",
                model=model,
                input_data={"image_path": image_path, "attempt": attempt + 1},
            )

            # Call Claude vision API
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            raw_text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            logger.debug(
                "Claude response (attempt %d): %s...",
                attempt + 1,
                raw_text[:200],
            )

            # Update Langfuse generation
            generation.update(
                output=raw_text[:500],
                usage={
                    "input": input_tokens,
                    "output": output_tokens,
                    "unit": "TOKENS",
                },
            )
            generation.end()

            # Parse and validate
            raw_items = _parse_extraction_response(raw_text)
            validated_items: list[ExtractedBillFields] = []

            for item_data in raw_items:
                validated_items.append(ExtractedBillFields(**item_data))

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            cost = _estimate_cost(total_input_tokens, total_output_tokens)

            result = ExtractionResult(
                items=validated_items,
                raw_llm_output=raw_text,
                parse_retries=attempt,
            )

            logger.info(
                "Extraction succeeded — %d items, %d tokens, %.1f ms, $%.4f",
                len(validated_items),
                total_input_tokens + total_output_tokens,
                elapsed_ms,
                cost,
            )

            # Update state
            state.extracted_fields = result
            state.processing_time_ms = round(elapsed_ms, 2)
            state.estimated_cost_usd = round(cost, 6)

            return state

        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Extraction attempt %d failed: %s", attempt + 1, last_error
            )

    # All retries exhausted
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    cost = _estimate_cost(total_input_tokens, total_output_tokens)

    state.extracted_fields = ExtractionResult(
        items=[],
        raw_llm_output=f"FAILED after {MAX_RETRIES} attempts: {last_error}",
        parse_retries=MAX_RETRIES,
    )
    state.error = f"Extraction failed after {MAX_RETRIES} attempts: {last_error}"
    state.processing_time_ms = round(elapsed_ms, 2)
    state.estimated_cost_usd = round(cost, 6)

    logger.error("Extraction failed after all retries: %s", last_error)
    return state
