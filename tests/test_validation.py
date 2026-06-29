"""
RxPilot — Tests for the Validation Agent (Phase 2).

Tests each validation check in isolation using mocked database queries.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agents.state import ExtractedBillFields, ExtractionResult, PharmacyState


# ── Fixtures ──


def _make_state(**kwargs) -> PharmacyState:
    """Helper to create a minimal PharmacyState for testing."""
    defaults = dict(input_type="image", raw_input_ref="/tmp/test.jpg")
    defaults.update(kwargs)
    return PharmacyState(**defaults)


def _make_item(**kwargs) -> ExtractedBillFields:
    """Helper to create an ExtractedBillFields item."""
    defaults = dict(medicine_name="Paracetamol 500mg")
    defaults.update(kwargs)
    return ExtractedBillFields(**defaults)


def _state_with_items(items: list[ExtractedBillFields]) -> PharmacyState:
    """Create a PharmacyState with extracted items."""
    extraction = ExtractionResult(items=items, raw_llm_output="[]")
    return _make_state(extracted_fields=extraction)


# ── Tests: No items ──


@pytest.mark.asyncio
async def test_validation_no_items_returns_empty_flags():
    """State with no extracted items should produce no flags."""
    from agents.validation_agent import run_validation
    state = _make_state()
    result = await run_validation(state)
    assert result.validation_flags == []


@pytest.mark.asyncio
async def test_validation_empty_items_returns_empty_flags():
    """Empty items list should produce no flags."""
    from agents.validation_agent import run_validation
    state = _state_with_items([])
    result = await run_validation(state)
    assert result.validation_flags == []


# ── Tests: Expired Medicine ──


@pytest.mark.asyncio
async def test_expired_medicine_flagged():
    """Medicine with past expiry date should be flagged as expired."""
    from agents.validation_agent import run_validation

    past_date = (date.today() - timedelta(days=30)).isoformat()
    item = _make_item(
        medicine_name="Amoxicillin 500mg",
        expiry_date=past_date,
        batch_number="AMX-001",
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert any("expired" in f for f in result.validation_flags)
    assert any("Amoxicillin 500mg" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_future_expiry_not_flagged():
    """Medicine with future expiry date should NOT be flagged."""
    from agents.validation_agent import run_validation

    future_date = (date.today() + timedelta(days=365)).isoformat()
    item = _make_item(
        medicine_name="Omeprazole 20mg",
        expiry_date=future_date,
        batch_number="OMP-001",
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert not any("expired" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_null_expiry_not_flagged():
    """Medicine with no expiry date should not trigger expired flag."""
    from agents.validation_agent import run_validation

    item = _make_item(medicine_name="Vitamin D3", expiry_date=None)
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert not any("expired" in f for f in result.validation_flags)


# ── Tests: Date Inconsistency ──


@pytest.mark.asyncio
async def test_date_inconsistency_expiry_before_manufacture():
    """Expiry before manufacture date should be flagged."""
    from agents.validation_agent import run_validation

    item = _make_item(
        medicine_name="Metformin 500mg",
        manufacture_date="2024-06-01",
        expiry_date="2024-03-01",  # BEFORE manufacture
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert any("date_inconsistency" in f for f in result.validation_flags)
    assert any("Metformin 500mg" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_date_inconsistency_expiry_equals_manufacture():
    """Expiry on same day as manufacture date should be flagged."""
    from agents.validation_agent import run_validation

    item = _make_item(
        medicine_name="Lisinopril 10mg",
        manufacture_date="2024-06-01",
        expiry_date="2024-06-01",  # SAME as manufacture
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert any("date_inconsistency" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_valid_dates_not_flagged():
    """Valid expiry after manufacture should not be flagged."""
    from agents.validation_agent import run_validation

    item = _make_item(
        medicine_name="Atorvastatin 20mg",
        manufacture_date="2024-01-01",
        expiry_date="2026-12-31",
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert not any("date_inconsistency" in f for f in result.validation_flags)


# ── Tests: Missing Critical Fields ──


@pytest.mark.asyncio
async def test_missing_batch_and_expiry_flagged():
    """Item with neither batch_number nor expiry_date should be flagged."""
    from agents.validation_agent import run_validation

    item = _make_item(
        medicine_name="Unknown Tablet",
        batch_number=None,
        expiry_date=None,
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert any("missing_fields" in f for f in result.validation_flags)
    assert any("Unknown Tablet" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_has_batch_not_flagged_for_missing():
    """Item with batch_number but no expiry should NOT be flagged for missing fields."""
    from agents.validation_agent import run_validation

    item = _make_item(
        medicine_name="Test Drug",
        batch_number="BATCH-001",
        expiry_date=None,
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert not any("missing_fields" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_has_expiry_not_flagged_for_missing():
    """Item with expiry_date but no batch should NOT be flagged for missing fields."""
    from agents.validation_agent import run_validation

    future_date = (date.today() + timedelta(days=365)).isoformat()
    item = _make_item(
        medicine_name="Test Drug",
        batch_number=None,
        expiry_date=future_date,
    )
    state = _state_with_items([item])

    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    assert not any("missing_fields" in f for f in result.validation_flags)


# ── Tests: Duplicate Batch (DB-dependent) ──


@pytest.mark.asyncio
async def test_duplicate_batch_detected():
    """Batch number already in DB for same medicine should be flagged."""
    from agents.validation_agent import run_validation

    future = (date.today() + timedelta(days=180)).isoformat()
    item = _make_item(
        medicine_name="Warfarin 5mg",
        batch_number="WAR-DUPE-001",
        expiry_date=future,
    )
    state = _state_with_items([item])

    # Mock connection that reports 1 existing row
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = {"cnt": 1}
    mock_conn.cursor.return_value = mock_cur

    with patch("agents.validation_agent._get_connection", return_value=mock_conn):
        result = await run_validation(state)

    assert any("duplicate_batch" in f for f in result.validation_flags)
    assert any("WAR-DUPE-001" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_no_duplicate_batch_not_flagged():
    """Batch number NOT in DB should not produce duplicate flag."""
    from agents.validation_agent import run_validation

    future = (date.today() + timedelta(days=180)).isoformat()
    item = _make_item(
        medicine_name="Warfarin 5mg",
        batch_number="WAR-NEW-999",
        expiry_date=future,
    )
    state = _state_with_items([item])

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = {"cnt": 0}
    mock_conn.cursor.return_value = mock_cur

    with patch("agents.validation_agent._get_connection", return_value=mock_conn):
        result = await run_validation(state)

    assert not any("duplicate_batch" in f for f in result.validation_flags)


# ── Tests: Price Anomaly (DB-dependent) ──


@pytest.mark.asyncio
async def test_price_anomaly_detected():
    """Price deviating >50% from historical average should be flagged."""
    from agents.validation_agent import run_validation

    future = (date.today() + timedelta(days=180)).isoformat()
    item = _make_item(
        medicine_name="Metformin 500mg",
        batch_number="MET-001",
        expiry_date=future,
        price=500.0,  # Way above average
    )
    state = _state_with_items([item])

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    # Historical average is 30.0 INR; 500.0 is >50% deviation
    mock_cur.fetchone.return_value = {"avg_price": 30.0, "cnt": 5}
    mock_conn.cursor.return_value = mock_cur

    with patch("agents.validation_agent._get_connection", return_value=mock_conn):
        result = await run_validation(state)

    assert any("price_anomaly" in f for f in result.validation_flags)


@pytest.mark.asyncio
async def test_normal_price_not_anomaly():
    """Price within 50% of historical average should NOT be flagged."""
    from agents.validation_agent import run_validation

    future = (date.today() + timedelta(days=180)).isoformat()
    item = _make_item(
        medicine_name="Metformin 500mg",
        batch_number="MET-002",
        expiry_date=future,
        price=32.0,  # Close to average of 30.0
    )
    state = _state_with_items([item])

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = {"avg_price": 30.0, "cnt": 5}
    mock_conn.cursor.return_value = mock_cur

    with patch("agents.validation_agent._get_connection", return_value=mock_conn):
        result = await run_validation(state)

    assert not any("price_anomaly" in f for f in result.validation_flags)


# ── Tests: Clean bill ──


@pytest.mark.asyncio
async def test_clean_bill_no_flags():
    """A valid bill with no issues should produce zero flags."""
    from agents.validation_agent import run_validation

    future = (date.today() + timedelta(days=365)).isoformat()
    items = [
        _make_item(
            medicine_name="Paracetamol 500mg",
            batch_number="PCM-CLEAN-001",
            manufacture_date="2024-01-01",
            expiry_date=future,
            price=32.50,
        ),
        _make_item(
            medicine_name="Amoxicillin 250mg",
            batch_number="AMX-CLEAN-002",
            manufacture_date="2024-02-01",
            expiry_date=future,
            price=85.00,
        ),
    ]
    state = _state_with_items(items)

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    # No duplicates, reasonable prices
    mock_cur.fetchone.side_effect = [
        {"cnt": 0},        # Paracetamol batch check
        {"cnt": 0},        # Amoxicillin batch check
        {"avg_price": 32.0, "cnt": 3},  # Paracetamol price check
        {"avg_price": 85.0, "cnt": 3},  # Amoxicillin price check
    ]
    mock_conn.cursor.return_value = mock_cur

    with patch("agents.validation_agent._get_connection", return_value=mock_conn):
        result = await run_validation(state)

    assert result.validation_flags == []


# ── Tests: DB unavailable ──


@pytest.mark.asyncio
async def test_db_unavailable_graceful_degradation():
    """When DB is unavailable, should still run non-DB checks without crashing."""
    from agents.validation_agent import run_validation

    past = (date.today() - timedelta(days=30)).isoformat()
    # Item 1: expired (triggers 'expired' flag)
    item_expired = _make_item(
        medicine_name="Expired Drug",
        expiry_date=past,
        batch_number="EXP-001",
    )
    # Item 2: missing both batch AND expiry (triggers 'missing_fields' flag)
    item_missing = _make_item(
        medicine_name="Incomplete Drug",
        expiry_date=None,
        batch_number=None,
    )
    state = _state_with_items([item_expired, item_missing])

    # DB unavailable
    with patch("agents.validation_agent._get_connection", return_value=None):
        result = await run_validation(state)

    # Should still catch non-DB issues
    assert any("expired" in f for f in result.validation_flags), \
        f"Expected 'expired' flag, got: {result.validation_flags}"
    assert any("missing_fields" in f for f in result.validation_flags), \
        f"Expected 'missing_fields' flag, got: {result.validation_flags}"
