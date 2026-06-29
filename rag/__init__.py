"""
RxPilot — RAG Embedding Pipeline.

Builds and queries a drug interaction vector corpus using:
  - sentence-transformers (all-MiniLM-L6-v2, 384-dim) for embeddings
  - pgvector for cosine similarity search

The corpus is built from curated CSV files in rag/corpus/ and stored
in the drug_interactions table.

Usage:
    from rag.embeddings import search_interactions, build_corpus

    # Build corpus (run once or on data update)
    build_corpus()

    # Search for interactions between drugs
    results = search_interactions(["Warfarin", "Aspirin"])
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──
CORPUS_DIR = Path(__file__).parent / "corpus"
INTERACTIONS_CSV = CORPUS_DIR / "drug_interactions.csv"
SAFETY_INFO_CSV = CORPUS_DIR / "drug_safety_info.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
SIMILARITY_THRESHOLD = 0.3

# ── Lazy-loaded model ──
_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("Loaded sentence-transformer model: %s", MODEL_NAME)
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
    return _model


def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts (batched)."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return embeddings.tolist()


def _create_chunk_text(row: dict[str, str]) -> str:
    """Create a searchable text chunk from an interaction row."""
    parts = [
        f"Drug interaction between {row['drug_a']} and {row['drug_b']}.",
        f"Severity: {row['severity']}.",
        row["description"],
    ]
    if row.get("mechanism"):
        parts.append(f"Mechanism: {row['mechanism']}")
    if row.get("source"):
        parts.append(f"Source: {row['source']}")
    return " ".join(parts)


def load_interactions_csv() -> list[dict[str, str]]:
    """Load the drug interactions CSV file."""
    if not INTERACTIONS_CSV.exists():
        logger.warning("Drug interactions CSV not found: %s", INTERACTIONS_CSV)
        return []

    rows = []
    with open(INTERACTIONS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["chunk_text"] = _create_chunk_text(row)
            rows.append(row)

    logger.info("Loaded %d drug interaction entries from CSV", len(rows))
    return rows


def build_corpus() -> int:
    """
    Build the drug interaction vector corpus.

    Reads the CSV, generates embeddings, and inserts into the
    drug_interactions table via pgvector.

    Returns the number of entries inserted.
    """
    import psycopg2

    rows = load_interactions_csv()
    if not rows:
        logger.warning("No interaction data to build corpus from")
        return 0

    # Generate embeddings
    chunk_texts = [r["chunk_text"] for r in rows]
    logger.info("Generating embeddings for %d chunks...", len(chunk_texts))
    embeddings = embed_texts(chunk_texts)

    # Connect to database
    db_url = os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://rxpilot:rxpilot_dev@localhost:5432/rxpilot",
    )
    conn = psycopg2.connect(db_url)

    try:
        cur = conn.cursor()

        # Clear existing data
        cur.execute("DELETE FROM drug_interactions")

        # Insert new data
        for row, embedding in zip(rows, embeddings):
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            cur.execute(
                """
                INSERT INTO drug_interactions
                    (drug_a, drug_b, severity, description, mechanism, source, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    row["drug_a"],
                    row["drug_b"],
                    row["severity"],
                    row["description"],
                    row.get("mechanism", ""),
                    row["source"],
                    row["chunk_text"],
                    embedding_str,
                ),
            )

        # Log the build
        cur.execute(
            """
            INSERT INTO rag_build_log (corpus_name, entries_count, model_name, embedding_dim)
            VALUES (%s, %s, %s, %s)
            """,
            ("drug_interactions", len(rows), MODEL_NAME, EMBEDDING_DIM),
        )

        conn.commit()
        logger.info("Built drug interaction corpus: %d entries", len(rows))
        return len(rows)

    except Exception as e:
        conn.rollback()
        logger.error("Failed to build corpus: %s", e)
        raise
    finally:
        conn.close()


def search_interactions(
    drug_names: list[str],
    top_k: int = 5,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Search for drug interactions relevant to the given drug names.

    Uses two strategies:
    1. Exact text match on drug_a/drug_b columns (fast, precise)
    2. Vector similarity search for fuzzy matching (catches synonyms/variants)

    Args:
        drug_names: List of drug names to check interactions for.
        top_k: Maximum number of results to return.
        threshold: Minimum cosine similarity score.

    Returns:
        List of interaction dicts with fields: drug_a, drug_b, severity,
        description, mechanism, source, similarity_score.
    """
    import psycopg2
    import psycopg2.extras

    if len(drug_names) < 2:
        return []  # Need at least 2 drugs for an interaction

    db_url = os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://rxpilot:rxpilot_dev@localhost:5432/rxpilot",
    )

    results = []

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Strategy 1: Exact match on drug names
        drug_names_lower = [d.lower() for d in drug_names]
        cur.execute(
            """
            SELECT drug_a, drug_b, severity, description, mechanism, source,
                   1.0 AS similarity_score
            FROM drug_interactions
            WHERE LOWER(drug_a) = ANY(%s) AND LOWER(drug_b) = ANY(%s)
               OR LOWER(drug_b) = ANY(%s) AND LOWER(drug_a) = ANY(%s)
            """,
            (drug_names_lower, drug_names_lower, drug_names_lower, drug_names_lower),
        )
        exact_matches = [dict(r) for r in cur.fetchall()]
        results.extend(exact_matches)

        # Strategy 2: Vector similarity search
        # Build query from drug name pairs
        from itertools import combinations
        query_parts = []
        for a, b in combinations(drug_names, 2):
            query_parts.append(f"interaction between {a} and {b}")

        if query_parts:
            query_text = "; ".join(query_parts)
            query_embedding = embed_text(query_text)
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            cur.execute(
                """
                SELECT drug_a, drug_b, severity, description, mechanism, source,
                       1 - (embedding <=> %s::vector) AS similarity_score
                FROM drug_interactions
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, embedding_str, threshold, embedding_str, top_k),
            )
            vector_matches = [dict(r) for r in cur.fetchall()]

            # Deduplicate: prefer exact matches
            seen = {(r["drug_a"].lower(), r["drug_b"].lower()) for r in results}
            for match in vector_matches:
                key = (match["drug_a"].lower(), match["drug_b"].lower())
                rev_key = (match["drug_b"].lower(), match["drug_a"].lower())
                if key not in seen and rev_key not in seen:
                    results.append(match)
                    seen.add(key)

        conn.close()

    except Exception as e:
        logger.warning("Drug interaction search failed: %s", e)
        # Fall back to CSV-based search (no DB needed)
        results = _search_csv_fallback(drug_names)

    return results[:top_k]


def _search_csv_fallback(drug_names: list[str]) -> list[dict[str, Any]]:
    """
    Fallback search using CSV data directly (when DB is unavailable).

    Performs exact name matching on the CSV entries.
    """
    rows = load_interactions_csv()
    drug_names_lower = {d.lower().strip() for d in drug_names}
    results = []

    for row in rows:
        a = row["drug_a"].lower().strip()
        b = row["drug_b"].lower().strip()

        # Check if both drugs in this interaction are in our list
        if a in drug_names_lower and b in drug_names_lower:
            results.append({
                "drug_a": row["drug_a"],
                "drug_b": row["drug_b"],
                "severity": row["severity"],
                "description": row["description"],
                "mechanism": row.get("mechanism", ""),
                "source": row["source"],
                "similarity_score": 1.0,
            })
        # Also check if either drug matches (partial interaction detection)
        elif a in drug_names_lower or b in drug_names_lower:
            # Check if the other drug name is a substring of any drug in our list
            for drug in drug_names_lower:
                if (a in drug or drug in a) and b in drug_names_lower:
                    results.append({
                        "drug_a": row["drug_a"],
                        "drug_b": row["drug_b"],
                        "severity": row["severity"],
                        "description": row["description"],
                        "mechanism": row.get("mechanism", ""),
                        "source": row["source"],
                        "similarity_score": 0.8,
                    })
                    break

    return results
