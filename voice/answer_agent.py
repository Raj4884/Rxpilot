"""
RxPilot — Voice Answer Agent.

Answers pharmacy queries derived from voice transcripts.

For each intent type it uses the most appropriate data source:
  - stock_query     → queries bill_items table for recent stock data
  - expiry_query    → queries bill_items for medicines expiring soon
  - interaction_query → searches drug interaction RAG corpus
  - general_query   → uses Claude with context from the DB

Falls back gracefully if DB is unavailable or Claude API key is absent.

Usage:
    from voice.answer_agent import answer_query
    from agents.state import VoiceQuery, VoiceAnswer

    query = VoiceQuery(intent="stock_query", drug_name="Metformin", question="...")
    answer = await answer_query(query)
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from agents.state import VoiceAnswer, VoiceQuery

logger = logging.getLogger(__name__)


def _get_db_url() -> str:
    return os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://rxpilot:rxpilot_dev@localhost:5432/rxpilot",
    )


def _get_connection():
    """Return a psycopg2 connection or None if DB is unavailable."""
    try:
        import psycopg2
        return psycopg2.connect(_get_db_url())
    except Exception as e:
        logger.warning("Answer agent: DB unavailable — %s", e)
        return None


# ──────────────────────────────────────────────
# Intent handlers
# ──────────────────────────────────────────────


def _answer_stock_query(drug_name: str | None, conn) -> VoiceAnswer:
    """Query bill_items for stock levels of a given medicine."""
    if conn is None:
        return VoiceAnswer(
            answer_text=(
                f"I couldn't check the stock for "
                f"{'all medicines' if not drug_name else drug_name} "
                f"because the database is unavailable."
            ),
            source="stub",
            confidence=0.0,
        )

    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if drug_name:
            cur.execute(
                """
                SELECT medicine_name, SUM(quantity) as total_qty,
                       COUNT(*) as bill_count,
                       MAX(created_at) as last_received
                FROM bill_items
                WHERE LOWER(medicine_name) LIKE LOWER(%s)
                  AND quantity IS NOT NULL
                GROUP BY medicine_name
                ORDER BY last_received DESC
                LIMIT 5
                """,
                (f"%{drug_name}%",),
            )
        else:
            cur.execute(
                """
                SELECT medicine_name, SUM(quantity) as total_qty,
                       COUNT(*) as bill_count,
                       MAX(created_at) as last_received
                FROM bill_items
                WHERE quantity IS NOT NULL
                GROUP BY medicine_name
                ORDER BY total_qty DESC
                LIMIT 10
                """
            )

        rows = cur.fetchall()
        if not rows:
            item = drug_name or "the requested medicines"
            return VoiceAnswer(
                answer_text=f"No stock records found for {item} in the database.",
                source="database",
                confidence=0.9,
            )

        if drug_name and len(rows) == 1:
            r = rows[0]
            last = r["last_received"].strftime("%d %b %Y") if r["last_received"] else "unknown"
            answer = (
                f"{r['medicine_name']}: {r['total_qty']} units across "
                f"{r['bill_count']} bill{'s' if r['bill_count'] > 1 else ''}. "
                f"Last received: {last}."
            )
        else:
            lines = []
            for r in rows:
                lines.append(f"{r['medicine_name']}: {r['total_qty']} units")
            answer = "Current stock levels — " + "; ".join(lines) + "."

        return VoiceAnswer(answer_text=answer, source="database", confidence=0.95)

    except Exception as e:
        logger.error("Stock query failed: %s", e)
        return VoiceAnswer(
            answer_text=f"I encountered an error checking stock: {e}",
            source="database",
            confidence=0.0,
        )


def _answer_expiry_query(drug_name: str | None, conn) -> VoiceAnswer:
    """Query bill_items for medicines expiring soon."""
    if conn is None:
        return VoiceAnswer(
            answer_text="The database is unavailable for expiry checks.",
            source="stub",
            confidence=0.0,
        )

    try:
        import psycopg2.extras
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cutoff = (date.today() + timedelta(days=60)).isoformat()

        params: list[Any] = [date.today().isoformat(), cutoff]
        filter_clause = ""
        if drug_name:
            filter_clause = " AND LOWER(medicine_name) LIKE LOWER(%s)"
            params.append(f"%{drug_name}%")

        cur.execute(
            f"""
            SELECT DISTINCT medicine_name, MIN(expiry_date) as earliest_expiry
            FROM bill_items
            WHERE expiry_date IS NOT NULL
              AND expiry_date >= %s
              AND expiry_date <= %s
              {filter_clause}
            GROUP BY medicine_name
            ORDER BY earliest_expiry ASC
            LIMIT 10
            """,
            params,
        )

        rows = cur.fetchall()
        if not rows:
            scope = f"for {drug_name}" if drug_name else "in the next 60 days"
            return VoiceAnswer(
                answer_text=f"No medicines expiring {scope} were found.",
                source="database",
                confidence=0.9,
            )

        lines = [
            f"{r['medicine_name']} (expires {r['earliest_expiry']})"
            for r in rows
        ]
        count = len(lines)
        answer = (
            f"{count} medicine{'s' if count > 1 else ''} expiring within 60 days: "
            + "; ".join(lines) + "."
        )
        return VoiceAnswer(answer_text=answer, source="database", confidence=0.95)

    except Exception as e:
        logger.error("Expiry query failed: %s", e)
        return VoiceAnswer(
            answer_text=f"I encountered an error checking expiry dates: {e}",
            source="database",
            confidence=0.0,
        )


def _answer_interaction_query(drug_name: str | None, question: str) -> VoiceAnswer:
    """Search the RAG corpus for drug interactions."""
    try:
        from rag import _search_csv_fallback

        if drug_name:
            # Search for interactions involving the named drug
            # We need at least 2 drugs for a meaningful interaction check
            interactions = _search_csv_fallback([drug_name, "warfarin"])  # Common anchor
            # Filter to only those involving our drug
            interactions = [
                i for i in interactions
                if drug_name.lower() in i["drug_a"].lower()
                or drug_name.lower() in i["drug_b"].lower()
            ]
        else:
            return VoiceAnswer(
                answer_text=(
                    "Please specify a drug name to check interactions. "
                    "For example: 'Are there interactions for Warfarin?'"
                ),
                source="corpus",
                confidence=0.8,
            )

        if not interactions:
            return VoiceAnswer(
                answer_text=(
                    f"No known interactions found for {drug_name} in the corpus. "
                    "This does not mean it's interaction-free — always consult a pharmacist."
                ),
                source="corpus",
                confidence=0.7,
            )

        parts = []
        for i in interactions[:3]:
            other = i["drug_b"] if drug_name.lower() in i["drug_a"].lower() else i["drug_a"]
            parts.append(
                f"{other} ({i['severity']} severity): {i['description']}"
            )

        answer = f"Known interactions for {drug_name}: " + "; ".join(parts) + "."
        return VoiceAnswer(answer_text=answer, source="corpus", confidence=0.85)

    except Exception as e:
        logger.error("Interaction query failed: %s", e)
        return VoiceAnswer(
            answer_text=f"I couldn't retrieve interaction data: {e}",
            source="corpus",
            confidence=0.0,
        )


async def _answer_general_query(question: str) -> VoiceAnswer:
    """Use Claude to answer a general pharmacy query."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or anthropic is None:
        return VoiceAnswer(
            answer_text=(
                "I received your question but couldn't process it without the AI service. "
                f"Your question was: {question!r}"
            ),
            source="stub",
            confidence=0.3,
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-20250514"),
            max_tokens=250,
            system=(
                "You are a helpful pharmacy assistant. Answer pharmacy operations "
                "questions concisely (2-3 sentences max). "
                "This is for a PORTFOLIO DEMONSTRATION system, not real clinical advice. "
                "Always end with: 'Please verify with a licensed pharmacist.'"
            ),
            messages=[{"role": "user", "content": question}],
        )
        answer = response.content[0].text.strip()
        return VoiceAnswer(answer_text=answer, source="llm", confidence=0.85)

    except Exception as e:
        logger.error("General query LLM call failed: %s", e)
        return VoiceAnswer(
            answer_text=f"I couldn't process your question. Please try again. (Error: {e})",
            source="llm",
            confidence=0.0,
        )


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────


async def answer_query(query: VoiceQuery) -> VoiceAnswer:
    """
    Answer a parsed voice query using the appropriate data source.

    Routes to the correct handler based on intent:
    - stock_query → bill_items DB query
    - expiry_query → bill_items DB query (with date filter)
    - interaction_query → RAG corpus search
    - general_query → Claude LLM

    Args:
        query: Parsed VoiceQuery with intent and optional drug_name.

    Returns:
        VoiceAnswer with answer_text, source, and confidence.
    """
    logger.info(
        "Answering voice query — intent=%s, drug=%s",
        query.intent, query.drug_name,
    )

    conn = _get_connection()

    try:
        if query.intent == "stock_query":
            return _answer_stock_query(query.drug_name, conn)

        elif query.intent == "expiry_query":
            return _answer_expiry_query(query.drug_name, conn)

        elif query.intent == "interaction_query":
            return _answer_interaction_query(query.drug_name, query.question)

        else:  # general_query
            return await _answer_general_query(query.question)

    finally:
        if conn:
            conn.close()
