"""
RxPilot — Database connection management.

Manages async (asyncpg) and sync (psycopg2) connections to Postgres.
Tables are created by the init-db.sql script in Docker; this module
provides the connection pool and query helpers.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Register UUID adapter for psycopg2
psycopg2.extras.register_uuid()


def _get_sync_url() -> str:
    """Get the synchronous database URL."""
    return os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://rxpilot:rxpilot_dev@localhost:5432/rxpilot",
    )


def get_connection() -> psycopg2.extensions.connection:
    """Get a new synchronous database connection."""
    url = _get_sync_url()
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        return conn
    except Exception as e:
        logger.error("Database connection failed: %s", e)
        raise


def check_db_health() -> bool:
    """Check if the database is reachable and has the expected tables."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name IN ('bills', 'bill_items')"
        )
        result = cur.fetchone()
        conn.close()
        return result is not None and result[0] >= 2
    except Exception as e:
        logger.warning("Database health check failed: %s", e)
        return False


def _serialize_value(val: Any) -> Any:
    """Convert Python values to JSON-safe types."""
    if isinstance(val, UUID):
        return str(val)
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    return val


def insert_bill(
    image_path: str,
    extracted_data: dict[str, Any] | None = None,
    trace_id: str | None = None,
    processing_time_ms: float = 0.0,
    estimated_cost_usd: float = 0.0,
    status: str = "completed",
    error_message: str | None = None,
) -> str:
    """
    Insert a bill record and its extracted items into Postgres.
    Returns the bill UUID as a string.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Insert bill record
        cur.execute(
            """
            INSERT INTO bills (image_path, extracted_data, trace_id,
                              processing_time_ms, estimated_cost_usd,
                              status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                image_path,
                json.dumps(extracted_data) if extracted_data else None,
                trace_id,
                processing_time_ms,
                estimated_cost_usd,
                status,
                error_message,
            ),
        )
        bill_id = cur.fetchone()[0]

        # Insert line items if extraction succeeded
        if extracted_data and "items" in extracted_data:
            for item in extracted_data["items"]:
                cur.execute(
                    """
                    INSERT INTO bill_items (bill_id, medicine_name, batch_number,
                                           expiry_date, manufacture_date, quantity,
                                           unit, supplier_name, price, currency)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        bill_id,
                        item.get("medicine_name"),
                        item.get("batch_number"),
                        item.get("expiry_date"),
                        item.get("manufacture_date"),
                        item.get("quantity"),
                        item.get("unit"),
                        item.get("supplier_name"),
                        item.get("price"),
                        item.get("currency", "INR"),
                    ),
                )

        conn.commit()
        bill_id_str = str(bill_id)
        logger.info("Inserted bill %s with %d items", bill_id_str, len(extracted_data.get("items", [])) if extracted_data else 0)
        return bill_id_str

    except Exception as e:
        conn.rollback()
        logger.error("Failed to insert bill: %s", e)
        raise
    finally:
        conn.close()


def get_recent_bills(limit: int = 20) -> list[dict[str, Any]]:
    """Get the most recent bills with their items."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT b.id, b.image_path, b.extracted_data, b.trace_id,
                   b.processing_time_ms, b.estimated_cost_usd,
                   b.status, b.error_message, b.created_at
            FROM bills b
            ORDER BY b.created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {k: _serialize_value(v) for k, v in dict(row).items()}
            for row in rows
        ]
    except Exception as e:
        logger.error("Failed to fetch bills: %s", e)
        return []
    finally:
        conn.close()


def get_bill_by_id(bill_id: str) -> dict[str, Any] | None:
    """Get a single bill by ID with its items."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get the bill
        cur.execute("SELECT * FROM bills WHERE id = %s", (bill_id,))
        bill = cur.fetchone()
        if bill is None:
            return None

        bill_dict = {k: _serialize_value(v) for k, v in dict(bill).items()}

        # Get items
        cur.execute(
            "SELECT * FROM bill_items WHERE bill_id = %s ORDER BY created_at",
            (bill_id,),
        )
        items = cur.fetchall()
        bill_dict["items"] = [
            {k: _serialize_value(v) for k, v in dict(item).items()}
            for item in items
        ]

        return bill_dict

    except Exception as e:
        logger.error("Failed to fetch bill %s: %s", bill_id, e)
        return None
    finally:
        conn.close()
