"""
RxPilot — Drug Interaction Corpus Seeder.

Reads drug_interactions.csv, generates sentence-transformer embeddings,
and inserts into the drug_interactions PostgreSQL table via pgvector.

Usage:
    python scripts/seed-interactions.py

Requires:
    - PostgreSQL running with pgvector extension
    - DATABASE_URL_SYNC env var (or default localhost connection)
    - sentence-transformers installed: pip install sentence-transformers
"""

from __future__ import annotations

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed-interactions")


def main() -> None:
    logger.info("=" * 60)
    logger.info("RxPilot Drug Interaction Corpus Seeder")
    logger.info("=" * 60)

    # Import after path setup
    from rag import build_corpus, load_interactions_csv

    # Verify CSV exists
    rows = load_interactions_csv()
    if not rows:
        logger.error("No interaction data found. Check rag/corpus/drug_interactions.csv")
        sys.exit(1)

    logger.info("Found %d drug interaction entries", len(rows))

    # Verify DB connection
    db_url = os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://rxpilot:rxpilot_dev@localhost:5432/rxpilot",
    )
    logger.info("Connecting to database: %s", db_url.split("@")[-1])

    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        conn.close()
        logger.info("Database connection OK")
    except Exception as e:
        logger.error("Cannot connect to database: %s", e)
        logger.info("Make sure PostgreSQL is running and the database is initialized.")
        logger.info("Run: docker-compose up -d postgres")
        sys.exit(1)

    # Build corpus
    logger.info("Building corpus (generating embeddings — this may take 30-60s)...")
    try:
        count = build_corpus()
        logger.info("=" * 60)
        logger.info("Corpus built successfully: %d entries seeded", count)
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Corpus build failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
