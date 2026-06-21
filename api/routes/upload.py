"""
RxPilot — Bill upload and extraction endpoints.

POST /v1/upload  — Upload a bill image, run extraction pipeline
GET  /v1/bills   — List recent bills
GET  /v1/bills/{id} — Get a single bill with its extracted items
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents.state import PharmacyState
from api.database import get_bill_by_id, get_recent_bills, insert_bill
from graph import compiled_graph
from observability.tracing import flush_langfuse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["bills"])

# ── Response models ──


class ExtractedItemResponse(BaseModel):
    """A single extracted line item."""
    medicine_name: str
    batch_number: str | None = None
    expiry_date: str | None = None
    manufacture_date: str | None = None
    quantity: int | None = None
    unit: str | None = None
    supplier_name: str | None = None
    price: float | None = None
    currency: str = "INR"


class UploadResponse(BaseModel):
    """Response from the bill upload endpoint."""
    bill_id: str
    status: str
    items: list[ExtractedItemResponse] = Field(default_factory=list)
    items_count: int = 0
    trace_id: str
    processing_time_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    error: str | None = None
    image_url: str = ""


class BillSummary(BaseModel):
    """Summary of a bill for list view."""
    id: str
    image_path: str
    status: str
    trace_id: str | None = None
    processing_time_ms: float | None = None
    estimated_cost_usd: float | None = None
    items_count: int = 0
    created_at: str


# ── Endpoints ──


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload_bill(file: UploadFile = File(...)):
    """
    Upload a pharmacy bill image and extract structured data.

    Accepts JPEG, PNG, GIF, or WebP images up to 10MB.
    Runs the LangGraph extraction pipeline and stores results in Postgres.
    """
    # Validate file type
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)}MB",
        )

    # Save to upload directory
    upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    file_path = upload_dir / filename

    with open(file_path, "wb") as f:
        f.write(contents)

    logger.info("Uploaded bill image: %s (%d bytes)", filename, len(contents))

    # Create initial state for the pipeline
    trace_id = str(uuid.uuid4())
    initial_state = PharmacyState(
        input_type="image",
        raw_input_ref=str(file_path),
        trace_id=trace_id,
    )

    # Run the LangGraph pipeline
    start_time = time.perf_counter()
    try:
        result = await compiled_graph.ainvoke(initial_state.model_dump())
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse result
        extracted_items = []
        items_data = []
        error_msg = result.get("error")
        status = "failed" if error_msg else "completed"

        if result.get("extracted_fields") and result["extracted_fields"].get("items"):
            for item in result["extracted_fields"]["items"]:
                extracted_items.append(ExtractedItemResponse(**item))
                items_data.append(item)

        # Store in database
        try:
            bill_id = insert_bill(
                image_path=str(file_path),
                extracted_data={"items": items_data} if items_data else None,
                trace_id=trace_id,
                processing_time_ms=result.get("processing_time_ms", elapsed_ms),
                estimated_cost_usd=result.get("estimated_cost_usd", 0.0),
                status=status,
                error_message=error_msg,
            )
        except Exception as e:
            logger.error("Failed to store bill in database: %s", e)
            bill_id = file_id  # Use file ID as fallback

        # Flush Langfuse traces
        flush_langfuse()

        return UploadResponse(
            bill_id=bill_id,
            status=status,
            items=extracted_items,
            items_count=len(extracted_items),
            trace_id=trace_id,
            processing_time_ms=round(result.get("processing_time_ms", elapsed_ms), 2),
            estimated_cost_usd=round(result.get("estimated_cost_usd", 0.0), 6),
            error=error_msg,
            image_url=f"/uploads/{filename}",
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error("Pipeline failed: %s", e, exc_info=True)

        # Store the failure
        try:
            bill_id = insert_bill(
                image_path=str(file_path),
                trace_id=trace_id,
                status="failed",
                error_message=str(e),
                processing_time_ms=elapsed_ms,
            )
        except Exception:
            bill_id = file_id

        raise HTTPException(
            status_code=500,
            detail=f"Extraction pipeline failed: {str(e)}",
        )


@router.get("/bills")
async def list_bills(limit: int = 20):
    """List recent bills with summary info."""
    bills = get_recent_bills(limit=limit)
    summaries = []
    for bill in bills:
        extracted = bill.get("extracted_data")
        items_count = 0
        if extracted and isinstance(extracted, dict):
            items_count = len(extracted.get("items", []))
        elif extracted and isinstance(extracted, str):
            import json
            try:
                parsed = json.loads(extracted)
                items_count = len(parsed.get("items", []))
            except (json.JSONDecodeError, AttributeError):
                pass

        summaries.append(BillSummary(
            id=str(bill["id"]),
            image_path=bill["image_path"],
            status=bill.get("status", "unknown"),
            trace_id=bill.get("trace_id"),
            processing_time_ms=bill.get("processing_time_ms"),
            estimated_cost_usd=bill.get("estimated_cost_usd"),
            items_count=items_count,
            created_at=str(bill.get("created_at", "")),
        ))

    return {"bills": summaries, "total": len(summaries)}


@router.get("/bills/{bill_id}")
async def get_bill(bill_id: str):
    """Get a single bill with its extracted items."""
    bill = get_bill_by_id(bill_id)
    if bill is None:
        raise HTTPException(status_code=404, detail="Bill not found")
    return bill
