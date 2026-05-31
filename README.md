# FiscalAgent: MultiModal RAG for Financial Reports

This is an agentic, enterprise-grade RAG system designed to process complex Indian corporate annual reports (e.g., Reliance, Tata). It uses advanced spatial extraction, hybrid search, and the Reflexion pattern for high-accuracy financial analytics.

## Tech Stack
- **Frontend:** Next.js 16 (App Router), TypeScript, Tailwind CSS.
- **Backend:** FastAPI (Python), LangGraph (Reflexion pattern).
- **Database:** Supabase (Postgres, pgvector, tsvector).
- **LLM:** Gemini 1.5 Pro/Flash (Primary), Gemini Embeddings (768 dimensions).
- **Security:** slowapi (Rate limiting), llm-guard (Injection defense), Pydantic (Strict metadata validation).
- **Observability:** Langfuse.
- **Evaluation:** RAGAS.

## Project Structure
- `backend/`: FastAPI application, security middleware, and LangGraph services.
- `frontend/`: Next.js 16 application with streaming chat UI.
- `supabase/`: Database migrations including Hybrid Search RRF function.
- `data/`: Local storage for PDF documents.

## Key Features
- **Multimodal 2D Extraction:** Gemini native spatial extraction for text, tables, and bounding boxes.
- **Reflexion Orchestration:** LangGraph-based retrieval-grading-rewriting loop.
- **Defense-in-Depth:** Rate limiting, token budgeting, and dual-layer prompt injection defense.
- **Factual Enforcement:** Structured `AnswerWithEvidence` output validated via Guardrails AI (logic scaffolded).
- **Coordinate Conversion:** Backend utility to map Gemini normalized boxes to frontend UI coordinates.

## Setup Instructions

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Update .env with your GEMINI_API_KEY and DATABASE_URL
uvicorn main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Database Setup
Run the SQL migrations in `supabase/migrations/20240101_init.sql` against your local Supabase/Postgres instance.

## Ingestion Pipeline
To trigger the nightly ingestion simulation:
```bash
python backend/scraper.py
```

## Evaluation
To run the RAGAS evaluation pipeline:
```bash
python backend/eval.py
```
