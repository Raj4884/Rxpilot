"""
RxPilot — Validation Agent.

Cross-checks extracted bill items against historical Postgres records
and flags anomalies. This agent runs AFTER extraction and BEFORE safety.

Checks performed:
  1. Duplicate batch — same batch_number already in bill_items for same medicine
  2. Expired medicine — expiry_date is in the past
  3. Price anomaly — price deviates >50% from historical average
  4. Date inconsistency — expiry_date <= manufacture_date
  5. Missing critical fields — no batch_number AND no expiry_date

All database queries use psycopg2 sync connections (matching Phase 1 pattern).
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

import psycopg2
import psycopg2.extras

from agents.state import PharmacyState

logger = logging.getLogger(__name__)


def _get_db_url() -> str:
    """Get the database connection URL."""
    return os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://rxpilot:rxpilot_dev@localhost:5432/rxpilot",
    )


def _get_connection():
    """Get a database connection (returns None if DB unavailable)."""
    try:
        conn = psycopg2.connect(_get_db_url())
        return conn
    except Exception as e:
        logger.warning("Validation agent: DB unavailable — %s", e)
        return None


def _check_duplicate_batch(
    items: list[dict[str, Any]],
    conn,
) -> list[str]:
    """Check if any batch numbers already exist in the database."""
    flags = []
    if conn is None:
        return flags

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for item in items:
            batch = item.get("batch_number")
            medicine = item.get("medicine_name", "")
            if not batch:
                continue

            cur.execute(
                """
                SELECT COUNT(*) as cnt FROM bill_items
                WHERE LOWER(batch_number) = LOWER(%s)
                  AND LOWER(medicine_name) = LOWER(%s)
                """,
                (batch, medicine),
            )
            result = cur.fetchone()
            if result and result["cnt"] > 0:
                flags.append(f"duplicate_batch:{batch}")
                logger.info(
                    "Duplicate batch detected: %s for %s", batch, medicine
                )
    except Exception as e:
        logger.warning("Duplicate batch check failed: %s", e)

    return flags


def _check_expired_medicines(
    items: list[dict[str, Any]],
) -> list[str]:
    """Check if any medicines have expired."""
    flags = []
    today = date.today()

    for item in items:
        expiry_str = item.get("expiry_date")
        medicine = item.get("medicine_name", "")
        if not expiry_str:
            continue

        try:
            expiry = date.fromisoformat(expiry_str)
            if expiry < today:
                flags.append(f"expired:{medicine}")
                logger.info(
                    "Expired medicine detected: %s (expired %s)",
                    medicine, expiry_str,
                )
        except (ValueError, TypeError):
            pass

    return flags


def _check_price_anomalies(
    items: list[dict[str, Any]],
    conn,
) -> list[str]:
    """Check if prices deviate >50% from historical average."""
    flags = []
    if conn is None:
        return flags

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for item in items:
            price = item.get("price")
            medicine = item.get("medicine_name", "")
            if price is None or price <= 0:
                continue

            cur.execute(
                """
                SELECT AVG(price) as avg_price, COUNT(*) as cnt
                FROM bill_items
                WHERE LOWER(medicine_name) = LOWER(%s)
                  AND price IS NOT NULL AND price > 0
                """,
                (medicine,),
            )
            result = cur.fetchone()
            if result and result["cnt"] and result["cnt"] >= 2:
                avg_price = float(result["avg_price"])
                if avg_price > 0:
                    deviation = abs(price - avg_price) / avg_price
                    if deviation > 0.5:
                        flags.append(f"price_anomaly:{medicine}")
                        logger.info(
                            "Price anomaly: %s — current=%.2f, avg=%.2f (%.0f%% deviation)",
                            medicine, price, avg_price, deviation * 100,
                        )
    except Exception as e:
        logger.warning("Price anomaly check failed: %s", e)

    return flags


def _check_date_inconsistencies(
    items: list[dict[str, Any]],
) -> list[str]:
    """Check if expiry_date <= manufacture_date."""
    flags = []

    for item in items:
        expiry_str = item.get("expiry_date")
        mfg_str = item.get("manufacture_date")
        medicine = item.get("medicine_name", "")

        if not expiry_str or not mfg_str:
            continue

        try:
            expiry = date.fromisoformat(expiry_str)
            mfg = date.fromisoformat(mfg_str)
            if expiry <= mfg:
                flags.append(f"date_inconsistency:{medicine}")
                logger.info(
                    "Date inconsistency: %s — expiry=%s <= manufacture=%s",
                    medicine, expiry_str, mfg_str,
                )
        except (ValueError, TypeError):
            pass

    return flags


def _check_missing_critical_fields(
    items: list[dict[str, Any]],
) -> list[str]:
    """Check if items are missing both batch_number AND expiry_date."""
    flags = []

    for item in items:
        batch = item.get("batch_number")
        expiry = item.get("expiry_date")
        medicine = item.get("medicine_name", "")

        if not batch and not expiry:
            flags.append(f"missing_fields:{medicine}")
            logger.info(
                "Missing critical fields: %s — no batch_number and no expiry_date",
                medicine,
            )

    return flags


async def run_validation(
    state: PharmacyState,
    trace: Any = None,
) -> PharmacyState:
    """
    Run the validation agent on extracted bill items.

    Checks extracted items against historical records and business rules.
    Populates state.validation_flags with any anomalies found.

    Args:
        state: PharmacyState with extracted_fields populated.
        trace: Langfuse trace object for observability.

    Returns:
        Updated PharmacyState with validation_flags populated.
    """
    if not state.extracted_fields or not state.extracted_fields.items:
        logger.info("Validation agent: no items to validate")
        return state

    items = [item.model_dump() for item in state.extracted_fields.items]
    logger.info("Validation agent starting — %d items to check", len(items))

    all_flags: list[str] = []

    # Get database connection (may be None if DB unavailable)
    conn = _get_connection()

    try:
        # Run all checks
        all_flags.extend(_check_duplicate_batch(items, conn))
        all_flags.extend(_check_expired_medicines(items))
        all_flags.extend(_check_price_anomalies(items, conn))
        all_flags.extend(_check_date_inconsistencies(items))
        all_flags.extend(_check_missing_critical_fields(items))
    finally:
        if conn:
            conn.close()

    state.validation_flags = all_flags

    logger.info(
        "Validation agent complete — %d flags: %s",
        len(all_flags),
        all_flags if all_flags else "(none)",
    )

    return state
