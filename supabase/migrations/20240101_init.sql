-- Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table to store documents
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  gstin TEXT,
  fiscal_year INT,
  company_name TEXT,
  file_name TEXT,
  is_preloaded BOOLEAN DEFAULT false,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Create a table to store chunks
CREATE TABLE document_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  context_string TEXT,
  -- Bounding boxes [ymin, xmin, ymax, xmax] scaled 0-1000
  bounding_boxes JSONB,
  -- Structural path metadata (Company > Year > Section)
  path_metadata JSONB,
  -- Redundant metadata for easy filtering 
  company_name TEXT,
  fiscal_year INT,
  file_name TEXT,
  is_preloaded BOOLEAN DEFAULT false,
  -- Dense embedding (768 dimensions for Gemini)
  embedding VECTOR(768),
  -- Sparse index
  fts TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Create an index for vector similarity search
CREATE INDEX document_chunks_embedding_idx ON document_chunks USING hnsw (embedding vector_cosine_ops);
-- Create an index for full-text search
CREATE INDEX document_chunks_fts_idx ON document_chunks USING gin (fts);

-- Create a table for Agent Lightning Telemetry
CREATE TABLE agent_lightning_telemetry (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT NOT NULL,
  query TEXT NOT NULL,
  retrieval_latency_ms INT,
  llm_latency_ms INT,
  tokens_used INT,
  optimization_metrics JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Create a function for Hybrid Search with Reciprocal Rank Fusion (RRF)
-- Explicitly requested: K=60, RRF algorithm mathematically: score(d) = sum(1 / (k + rank_d))
CREATE OR REPLACE FUNCTION match_chunks_rrf(
  query_text TEXT,
  query_embedding VECTOR(768),
  metadata_filter JSONB DEFAULT '{}'::jsonb,
  match_count INT DEFAULT 20,
  rrf_k INT DEFAULT 60,
  filter_document_id UUID DEFAULT NULL
)
RETURNS TABLE (
  chunk_id UUID,
  document_id UUID,
  content TEXT,
  bounding_boxes JSONB,
  path_metadata JSONB,
  document_title TEXT,
  score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  WITH dense_matches AS (
    SELECT
      id,
      ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS rank
    FROM document_chunks
    WHERE (path_metadata @> metadata_filter OR metadata_filter = '{}'::jsonb)
      AND (filter_document_id IS NULL OR document_chunks.document_id = filter_document_id)
    ORDER BY embedding <=> query_embedding
    LIMIT match_count
  ),
  sparse_matches AS (
    SELECT
      id,
      ROW_NUMBER() OVER (ORDER BY ts_rank_cd(fts, plainto_tsquery('english', query_text)) DESC) AS rank
    FROM document_chunks
    WHERE fts @@ plainto_tsquery('english', query_text)
      AND (path_metadata @> metadata_filter OR metadata_filter = '{}'::jsonb)
      AND (filter_document_id IS NULL OR document_chunks.document_id = filter_document_id)
    ORDER BY ts_rank_cd(fts, plainto_tsquery('english', query_text)) DESC
    LIMIT match_count
  ),
  combined_matches AS (
    SELECT
      COALESCE(d.id, s.id) AS id,
      COALESCE(1.0 / (rrf_k + d.rank), 0.0) +
      COALESCE(1.0 / (rrf_k + s.rank), 0.0) AS score
    FROM dense_matches d
    FULL OUTER JOIN sparse_matches s ON d.id = s.id
  )
  SELECT
    c.id,
    c.document_id,
    c.content,
    c.bounding_boxes,
    c.path_metadata,
    docs.title,
    cm.score
  FROM combined_matches cm
  JOIN document_chunks c ON c.id = cm.id
  JOIN documents docs ON c.document_id = docs.id
  ORDER BY cm.score DESC
  LIMIT match_count;
END;
$$;