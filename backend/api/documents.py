import os
import shutil
import uuid
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from supabase import create_client, Client
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from backend.utils.logger import logger
from backend.services.ingestion import process_document_text

router = APIRouter(prefix="/documents", tags=["documents"])

supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
supabase_key = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_SECRET_KEY")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
)
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

class DocumentResponse(BaseModel):
    id: str
    title: str
    file_name: str
    company_name: str | None = None
    fiscal_year: int | None = None
    is_preloaded: bool

@router.get("", response_model=dict)
async def get_documents():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        response = supabase.table("documents").select("*").order("created_at", desc=True).execute()
        return {"documents": response.data}
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save temp file
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    os.makedirs("/tmp", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Step 1: Create Document Entry in Supabase
        doc_res = supabase.table("documents").insert({
            "title": file.filename,
            "file_name": file.filename,
            "is_preloaded": False
        }).execute()
        
        doc_id = doc_res.data[0]["id"]
        
        # We need to extract the PDF text (assuming ingestion.py's process_document_text handles it or we use PyMuPDF here)
        import fitz # PyMuPDF
        doc = fitz.open(temp_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        
        path_metadata = {
            "document_id": doc_id,
            "file_name": file.filename
        }
        
        # Process and get chunks (embeddings, context strings, etc.)
        # Note: This might take a while, in a real production app this should be a background task (e.g. Celery)
        chunks = process_document_text(file.filename, full_text, path_metadata)
        
        # Step 2: Insert Chunks into Supabase
        for chunk in chunks:
            supabase.table("document_chunks").insert({
                "document_id": doc_id,
                "content": chunk.text,
                "context_string": chunk.context_string,
                "embedding": chunk.embedding,
                "path_metadata": path_metadata,
                "file_name": file.filename,
                "is_preloaded": False,
                "bounding_boxes": chunk.bounding_boxes
            }).execute()

        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
