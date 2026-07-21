import os
import sys

# Add the project root to the Python path to resolve backend module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg
from dotenv import load_dotenv

from backend.api.chat import router as chat_router
from backend.middleware.rate_limit import setup_rate_limiting
from backend.utils.logger import logger

load_dotenv()

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the LangGraph Postgres Checkpointer
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # As explicitly required in Phase 1: call the async .setup() method
            await checkpointer.setup()
            logger.info("LangGraph Postgres Checkpointer setup complete.")
            
            # We attach it to the app state so routes can access the configured checkpointer
            app.state.checkpointer = checkpointer
            
            yield
            logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Failed to initialize checkpointer: {e}")
        raise

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
