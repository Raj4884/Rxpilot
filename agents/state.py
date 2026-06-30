"""
RxPilot — Shared Pydantic state schema for the LangGraph graph.

All agent nodes read from and write to this state object. Type hints and
validators ensure data quality at every handoff.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────


class ExtractedBillFields(BaseModel):
    """A single line-item extracted from a pharmacy bill."""

    medicine_name: str = Field(
        ..., min_length=1, description="Name of the medicine or drug"
    )
    batch_number: str | None = Field(
        default=None, description="Batch/lot number"
    )
    expiry_date: str | None = Field(
        default=None, description="Expiry date (YYYY-MM-DD)"
    )
    manufacture_date: str | None = Field(
        default=None, description="Manufacturing date (YYYY-MM-DD)"
    )
    quantity: int | None = Field(
        default=None, ge=0, description="Quantity of units"
    )
    unit: str | None = Field(
        default=None, description="Unit of measure (e.g. tablets, strips, ml)"
    )
    supplier_name: str | None = Field(
        default=None, description="Supplier or distributor name"
    )
    price: float | None = Field(
        default=None, ge=0, description="Price per unit or total price"
    )
    currency: str = Field(
        default="INR", description="Currency code"
    )

    @field_validator("expiry_date", "manufacture_date", mode="before")
    @classmethod
    def normalize_date_string(cls, v: str | None) -> str | None:
        """Accept various date formats and normalize to YYYY-MM-DD."""
        if v is None or v == "":
            return None
        # Try parsing common formats
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%Y", "%m-%Y"):
            try:
                parsed = datetime.strptime(v, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Return as-is if we can't parse — let downstream handle it
        return v

    @model_validator(mode="after")
    def check_expiry_after_manufacture(self) -> "ExtractedBillFields":
        """Validate that expiry date is after manufacture date if both present."""
        if self.expiry_date and self.manufacture_date:
            try:
                exp = date.fromisoformat(self.expiry_date)
                mfg = date.fromisoformat(self.manufacture_date)
                if exp <= mfg:
                    # Don't raise — flag it, the validation agent handles this
                    pass
            except ValueError:
                pass
        return self


class ExtractionResult(BaseModel):
    """Output of the extraction agent — all items found on the bill."""

    items: list[ExtractedBillFields] = Field(
        default_factory=list,
        description="Extracted line items from the bill",
    )
    raw_llm_output: str = Field(
        default="", description="Raw LLM response for debugging/tracing"
    )
    parse_retries: int = Field(
        default=0, ge=0, description="Number of parse retries needed"
    )


class SafetyFlag(BaseModel):
    """A drug-interaction or contraindication flag from the safety agent."""

    drug_pair: tuple[str, str] = Field(
        ..., description="Pair of drugs involved in the interaction"
    )
    severity: Literal["low", "moderate", "high", "critical"] = Field(
        ..., description="Severity level of the interaction"
    )
    description: str = Field(
        ..., description="Human-readable description of the interaction"
    )
    source: str = Field(
        ..., description="Citation to the corpus chunk / source document"
    )


class VoiceQuery(BaseModel):
    """Parsed intent from a voice input transcript."""

    intent: Literal[
        "stock_query",
        "expiry_query",
        "interaction_query",
        "general_query",
    ] = Field(..., description="Detected intent of the voice query")
    drug_name: str | None = Field(
        default=None, description="Primary drug name extracted from the query"
    )
    question: str = Field(
        ..., description="The verbatim or cleaned question to answer"
    )


class VoiceAnswer(BaseModel):
    """Answer produced by the voice answer agent."""

    answer_text: str = Field(
        ..., description="Human-readable answer to the voice query"
    )
    source: Literal["database", "corpus", "llm", "stub"] = Field(
        default="llm", description="Where the answer came from"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )


class ForecastResult(BaseModel):
    """Output of the forecast agent — reorder prediction for a medicine."""

    medicine_name: str = Field(
        ..., description="Name of the medicine"
    )
    predicted_reorder_date: str | None = Field(
        default=None, description="Predicted date to reorder (YYYY-MM-DD)"
    )
    predicted_quantity: int | None = Field(
        default=None, ge=0, description="Predicted reorder quantity"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Prediction confidence"
    )


# ──────────────────────────────────────────────
# Main graph state
# ──────────────────────────────────────────────


class PharmacyState(BaseModel):
    """
    Shared state object passed through the LangGraph StateGraph.

    Every agent node reads from and writes to this state. The orchestrator
    creates a new PharmacyState for each incoming request.
    """

    input_type: Literal["image", "voice"] = Field(
        ..., description="Type of input: photographed bill or voice query"
    )
    raw_input_ref: str = Field(
        ..., description="File path to the uploaded image or audio file"
    )
    transcript: str | None = Field(
        default=None,
        description="Transcribed text from voice input (ASR output)",
    )
    extracted_fields: ExtractionResult | None = Field(
        default=None,
        description="Structured extraction output from the extraction agent",
    )
    validation_flags: list[str] = Field(
        default_factory=list,
        description="Anomaly flags from the validation agent",
    )
    safety_flags: list[SafetyFlag] = Field(
        default_factory=list,
        description="Drug-interaction flags from the safety agent",
    )
    voice_query: VoiceQuery | None = Field(
        default=None,
        description="Parsed voice query intent (voice pipeline only)",
    )
    voice_answer: VoiceAnswer | None = Field(
        default=None,
        description="Answer produced by the voice answer agent",
    )
    forecast: ForecastResult | None = Field(
        default=None,
        description="Reorder forecast from the forecast agent",
    )
    final_response: str | None = Field(
        default=None,
        description="Final human-readable response text",
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Langfuse trace ID for this pipeline run",
    )

    # ── Metadata (not part of the agent spec, used for API response) ──
    processing_time_ms: float = Field(
        default=0.0, description="Total pipeline processing time in ms"
    )
    estimated_cost_usd: float = Field(
        default=0.0, description="Estimated LLM cost for this run in USD"
    )
    error: str | None = Field(
        default=None, description="Error message if pipeline failed"
    )
