import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg
from dotenv import load_dotenv

from backend.api.chat import router as chat_router
from backend.security.rate_limit import setup_rate_limiting

load_dotenv()

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the LangGraph Postgres Checkpointer
    async with psycopg.AsyncConnection.connect(DB_URI) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        # As explicitly required in Phase 1: call the async .setup() method
        await checkpointer.setup()
        print("LangGraph Postgres Checkpointer setup complete.")
        
        # We attach it to the app state so routes can access the configured checkpointer
        app.state.checkpointer = checkpointer
        
        yield
        print("Shutting down...")

app = FastAPI(title="MultiModal RAG API", lifespan=lifespan)

# Phase 3 Security: slowapi Rate Limiting
setup_rate_limiting(app)

# Phase 3 Security: CORS Middleware for localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
