# FiscalAgent: MultiModal RAG for Financial Reports

This is an agentic, enterprise-grade RAG system designed to process complex corporate annual reports. It uses advanced spatial extraction, hybrid search, and the Reflexion pattern for high-accuracy financial analytics.

## Tech Stack
- **Frontend:** Next.js 16 (App Router), TypeScript, Tailwind CSS, Supabase Auth.
- **Backend:** FastAPI (Python 3.13), LangGraph (Reflexion pattern).
- **Database:** Supabase (Postgres, pgvector, tsvector).
- **LLMs:** 
  - **Gemini 1.5 Pro:** Primary Reasoning & Multimodal Extraction.
  - **Groq (Llama 3.3):** High-speed Question Parsing & Contextual Retrieval.
  - **Gemini Embeddings:** 768-dimensional dense vectors.
- **Reranker:** FlashRank (Lightweight local cross-encoder).
- **Encoding:** TOON (Token-Oriented Object Notation) for ultra-efficient table serialization.
- **Security:** slowapi (Rate limiting), NativeGuard (Pydantic-based output security), Supabase Auth.
- **Observability:** Langfuse / Agent Lightning Telemetry.
- **Evaluation:** RAGAS (evaluated via Groq).

## Project Structure
- `backend/`: FastAPI application, middleware, and LangGraph services.
- `frontend/`: Next.js 16 application with streaming chat UI and Google OAuth.
- `supabase/`: Database migrations including RRF Hybrid Search and Telemetry tables.
- `logs/`: Local rotating log files for backend monitoring.

## Key Features
- **Multimodal Layout-Aware Chunking:** Groups structural elements (headings, paragraphs, tables) using Gemini File API 2D extraction.
- **TOON Encoding:** Reduces token usage for financial tables using Token-Oriented Object Notation.
- **RRF + FlashRank:** Two-stage retrieval using Reciprocal Rank Fusion (Supabase) followed by local FlashRank reranking.
- **Reflexion Orchestration:** LangGraph workflow with nodes for Question Parsing (Groq), Grading, and Rewriting.
- **SSE Token Streaming:** True token-by-token streaming from backend to React.
- **NativeGuard Security:** Zero-dependency Pydantic validation for structured financial outputs.
- **Robust Monitoring:** Centralized logging with rotating file handlers and console output for production visibility.

## Setup Instructions

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# Activate venv
pip install -r requirements.txt
# Update .env in root with GEMINI_API_KEY, GROQ_API_KEY, and SUPABASE credentials
uvicorn main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Database Setup
Run the SQL migrations in `supabase/migrations/20240101_init.sql` against your Supabase instance to enable pgvector, telemetry tables, and RRF functions.

## Ingestion Pipeline
To trigger the layout-aware ingestion pipeline:
```bash
python backend/scraper.py
```

## Evaluation
To run the production-ready RAGAS evaluation pipeline:
```bash
python backend/eval.py
```
