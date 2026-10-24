# FiscalAgent: MultiModal RAG for Financial Reports

An agentic, enterprise-grade RAG system for processing complex corporate annual reports. It combines a two-pass PDF extraction pipeline (PyMuPDF + Gemma 4 vision), hybrid vector/full-text search with RRF, and a LangGraph Reflexion loop for high-accuracy financial Q&A.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js (App Router), TypeScript, Tailwind CSS |
| **Backend** | FastAPI (Python), LangGraph (Reflexion pattern) |
| **Database** | Supabase — Postgres, `pgvector` (HNSW), `tsvector` (GIN) |
| **PDF Parsing** | PyMuPDF (`fitz`) — TOC/font-heuristic boundary detection, page rasterization |
| **Extraction LLM** | Gemma 4 26B (`google-generativeai`) — 2D spatial extraction via Gemini File API |
| **Query Parsing** | Groq / `qwen/qwen3.8-27b` — structured metadata + query rewrite |
| **Reasoning LLM** | Gemma 4 31B (`langchain-google-genai`) — context grading, query rewriting, answer generation |
| **Embeddings** | `gemini-embedding-2` — 768-dim dense vectors (task-typed) |
| **Reranker** | FlashRank — lightweight local cross-encoder (top-10 of 25 retrieved) |
| **Table Encoding** | TOON (Text-Oriented Object Notation) — pipe-delimited row serialization for LLM tokens |
| **Output Validation** | NativeGuard — zero-dependency Pydantic validator (`AnswerWithEvidence`) |
| **Security** | slowapi (rate limiting), injection defense middleware (`meta-llama/llama-prompt-guard-2-86m`), token budget middleware |
| **Observability** | Langfuse (`@observe`) — traces every chat generation |
| **Evaluation** | RAGAS pipeline (`eval.py`) |
| **Checkpointing** | LangGraph `AsyncPostgresSaver` — persistent multi-turn session state in Postgres |

---

## Project Structure

```
FiscalAgent/
├── backend/
│   ├── api/
│   │   ├── chat.py          # SSE streaming endpoint (/api/v1/chat/stream)
│   │   └── documents.py     # Document management endpoints
│   ├── middleware/
│   │   ├── rate_limit.py    # slowapi rate limiting
│   │   ├── token_budget.py  # Max token budget per request
│   │   └── injection_defense.py  # Prompt injection sanitization
│   ├── models/
│   │   ├── output.py        # AnswerWithEvidence + Citation Pydantic schemas
│   │   └── validation.py    # NativeGuard, DocumentMetadata, GSTIN validator
│   ├── services/
│   │   ├── graph.py         # LangGraph Reflexion agent (7-node workflow)
│   │   └── ingestion.py     # Chunking, context generation (Groq), batch embedding
│   ├── utils/
│   │   ├── logger.py        # Rotating file + console logger
│   │   ├── parsers.py       # TOON table encoder
│   │   └── coordinates.py   # Bounding box utilities
│   ├── scraper.py           # Two-pass ingestion pipeline entry point
│   ├── eval.py              # RAGAS evaluation pipeline
│   └── main.py              # FastAPI app, lifespan, CORS, routing
├── frontend/                # Next.js streaming chat UI
├── supabase/
│   └── migrations/
│       └── 20240101_init.sql  # pgvector, GIN FTS, RRF function, telemetry table
└── logs/                    # Rotating backend log files
```

---

## Ingestion Pipeline (Two-Pass Dynamic Batching)

`scraper.py` orchestrates a two-pass architecture using **PyMuPDF** for local parsing and **Gemma 4 26B** for multimodal extraction.

### Pass 1 — Logical Boundary Detection (PyMuPDF)
- Opens the PDF with `fitz.open()`.
- If a **Table of Contents (TOC)** exists, section boundaries are derived directly from it.
- If no TOC is found, **font-size heuristics** are applied: pages are sampled to compute an average font size, and spans `×1.6` larger that appear in the top 25% of the page are treated as section headings.
- Boundaries are merged and validated to guarantee 100% page coverage.

### Pass 2 — Rasterization & 2D Spatial Extraction (Gemma 4 26B)
- Each logical section is **rasterized** at 2× resolution (`fitz.Matrix(2, 2)`) into temporary PNG files.
- Images are uploaded in batches to the **Gemini File API** asynchronously (5 concurrent sections via `asyncio.Semaphore`).
- **Gemma 4 26B** receives the images and returns structured `StructuralElement` objects:
  - `text` — extracted content
  - `bounding_box` — `[ymin, xmin, ymax, xmax]` normalized 0–1000
  - `type` — `"paragraph"` | `"table_row"` | `"heading"`
- Response is validated via `ExtractionOutput` Pydantic schema using `response_schema`.
- Temporary PNGs and remote Gemini files are cleaned up after extraction.

### Chunking & Embedding (`ingestion.py`)
1. **Layout-Aware Grouping** — Elements are grouped by structural type. A heading starts a new chunk when the current group exceeds 2 elements; groups are capped at 15. Table splits reinject the last header row.
2. **TOON Encoding** — `table_row` elements are serialized via `encode_to_toon_table()` into pipe-delimited TOON format; paragraphs and headings are prefixed with their type labels.
3. **Contextual Retrieval (Groq)** — Groq `qwen/qwen3.8-27b` generates a 1–2 sentence contextual summary per chunk, concurrent under a rate-limit semaphore with exponential backoff (20s, 40s).
4. **Batch Embedding** — `gemini-embedding-2` embeds `"Context: <summary>\n\n<chunk>"` strings in batches of 100, producing 768-dim vectors.
5. **Persistence** — Document record and all chunks (`content`, `context_string`, `embedding`, `bounding_boxes`, `path_metadata`) are batch-inserted into Supabase.

---

## Query & Retrieval Pipeline (LangGraph Reflexion Agent)

The `rag_agent` in `graph.py` is a compiled `StateGraph` with 7 nodes:

```
parse_query → retrieve → rerank → grade_context
                  ↑                      |
                  └── reflect_and_rewrite ┘  (if irrelevant, max 3 retries)
                                         |
                              generate_answer / fallback_node → END
```

| Node | Model | Role |
|---|---|---|
| `parse_query` | Groq `qwen/qwen3.8-27b` | Extracts `optimized_query` + `metadata_filters` (company, year, section, gstin) via structured output |
| `retrieve` | `gemini-embedding-2` + Supabase RPC | Embeds the optimized query, calls `match_chunks_rrf` to fetch 25 candidates |
| `rerank` | FlashRank (local) | Cross-encodes all 25 candidates, keeps top 10, formats context string |
| `grade_context` | Gemma 4 31B | Binary relevance grade: is the context sufficient to answer the query? |
| `reflect_and_rewrite` | Gemma 4 31B | Rewrites query with different keywords; loops back to `retrieve` |
| `generate_answer` | Gemma 4 31B | Generates `AnswerWithEvidence` (answer + reasoning + bounding-box citations) |
| `fallback_node` | — | Returns a graceful "cannot find answer" after 3 failed retries |

### Hybrid Search (RRF)
The Supabase `match_chunks_rrf` SQL function fuses two ranked lists:
- **Dense retrieval** — cosine similarity via HNSW index on `embedding vector(768)`
- **Sparse retrieval** — BM25-style full-text search via GIN index on `fts tsvector`

Using **Reciprocal Rank Fusion**: `score(d) = Σ 1 / (k + rank_d)` with `k=60`.

---

## Output Schema

```python
class Citation(BaseModel):
    exact_quote: str         # Verbatim quote from source
    bounding_box: List[int]  # [ymin, xmin, ymax, xmax] (0–1000 scale)
    page_number: int | None

class AnswerWithEvidence(BaseModel):
    answer: str
    reasoning: str
    citations: List[Citation]
```

All answers pass through **NativeGuard** (`models/validation.py`) — a zero-dependency Pydantic validator that also enforces GSTIN format + Luhn-mod-36 checksum for Indian tax identifiers.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/chat/stream` | SSE streaming chat (LangGraph `astream_events`) |
| `GET/POST` | `/api/v1/documents/*` | Document upload and management |
| `GET` | `/health` | Health check |

The `/chat/stream` endpoint applies:
1. **Token budget validation** — rejects oversized inputs
2. **Injection defense** — dual-layer sanitization (regex filter + Groq `meta-llama/llama-prompt-guard-2-86m` semantic scanning) before graph execution
3. **Session locking** — enforces `active_document_id` consistency across turns via the Postgres checkpointer

---

## Setup Instructions

### 1. Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=...
GROQ_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
DATABASE_URL=postgresql://...   # For LangGraph Postgres checkpointer
PDF_INPUT_PATH=path/to/report.pdf
```

### 2. Backend
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Database
Run the migration against your Supabase instance to create tables, HNSW/GIN indexes, the `match_chunks_rrf` RRF function, and the telemetry table:
```bash
# Via Supabase dashboard SQL editor or psql:
psql $DATABASE_URL -f supabase/migrations/20240101_init.sql
```

---

## Running the Ingestion Pipeline
```bash
# Set PDF_INPUT_PATH in .env, then:
python backend/scraper.py
```

## Running Evaluation
```bash
python backend/eval.py
```
