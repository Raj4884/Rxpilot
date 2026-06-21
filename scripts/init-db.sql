-- ──────────────────────────────────────────────
-- RxPilot — Database Initialization
-- ──────────────────────────────────────────────
-- This script runs automatically on first Postgres startup
-- via docker-entrypoint-initdb.d.

-- Enable pgvector extension (for Phase 2 RAG)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Bills table ──
CREATE TABLE IF NOT EXISTS bills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_path TEXT NOT NULL,
    extracted_data JSONB,
    trace_id TEXT,
    processing_time_ms FLOAT,
    estimated_cost_usd FLOAT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | completed | failed
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Bill items (extracted line items) ──
CREATE TABLE IF NOT EXISTS bill_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bill_id UUID NOT NULL REFERENCES bills(id) ON DELETE CASCADE,
    medicine_name TEXT NOT NULL,
    batch_number TEXT,
    expiry_date DATE,
    manufacture_date DATE,
    quantity INTEGER,
    unit TEXT,
    supplier_name TEXT,
    price NUMERIC(12, 2),
    currency TEXT NOT NULL DEFAULT 'INR',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Indexes ──
CREATE INDEX IF NOT EXISTS idx_bills_created_at ON bills(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bills_status ON bills(status);
CREATE INDEX IF NOT EXISTS idx_bill_items_bill_id ON bill_items(bill_id);
CREATE INDEX IF NOT EXISTS idx_bill_items_medicine ON bill_items(medicine_name);

-- ── Phase 2 placeholder: drug interaction corpus vectors ──
-- CREATE TABLE IF NOT EXISTS drug_embeddings (
--     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
--     drug_name TEXT NOT NULL,
--     chunk_text TEXT NOT NULL,
--     source TEXT NOT NULL,
--     embedding vector(1536),
--     created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
-- );
