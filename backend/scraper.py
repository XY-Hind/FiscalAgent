import os
import sys

# Add the project root to the Python path to resolve backend module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import asyncio
import tempfile
import fitz  # PyMuPDF
from typing import List, Tuple
from pydantic import BaseModel, Field
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

from backend.services.ingestion import StructuralElement, process_structural_elements
from backend.utils.logger import logger

load_dotenv()

# Configure raw Gemini SDK
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
supabase_key = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_SECRET_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
)
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

class ExtractionOutput(BaseModel):
    elements: List[StructuralElement] = Field(description="List of extracted structural elements with bounding boxes and types.")

def get_logical_boundaries(pdf_path: str) -> List[Tuple[int, int]]:
    """
    Pass 1: Local Logical Boundary Detection (PyMuPDF)
    Detects section boundaries using TOC or font-size heuristics.
    Ensures 100% page coverage.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    boundaries = []
    total_pages = doc.page_count
    
    if toc:
        logger.info("TOC detected, using for logical boundaries.")
        current_page = 0
        for i in range(len(toc)):
            # toc[i][2] is 1-based page number
            section_start = toc[i][2] - 1
            
            # 1. Handle gap before this section (e.g. cover pages)
            if section_start > current_page:
                boundaries.append((current_page, section_start - 1))
            
            # 2. Determine section end
            if i + 1 < len(toc):
                section_end = toc[i+1][2] - 2
            else:
                section_end = total_pages - 1
            
            # 3. Add section if valid
            if section_start <= section_end:
                # Merge with previous if overlapping
                if boundaries and boundaries[-1][1] >= section_start:
                    boundaries[-1] = (boundaries[-1][0], max(boundaries[-1][1], section_end))
                else:
                    boundaries.append((section_start, section_end))
                current_page = section_end + 1
        
        # 4. Handle remaining pages
        if current_page < total_pages:
            boundaries.append((current_page, total_pages - 1))
    else:
        logger.info("No TOC detected, using font-size heuristics.")
        current_start = 0
        threshold = 30 # Default high threshold if sampling fails
        
        # Sample pages to find base font size
        sample_pages = min(10, total_pages)
        sizes = []
        for i in range(sample_pages):
            for block in doc[i].get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            sizes.append(span["size"])
        
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            threshold = avg_size * 1.6  # Heuristic for headings
        
        for page_num in range(total_pages):
            page = doc[page_num]
            found_heading = False
            # Only look for headings in the top 25% of the page
            page_height = page.rect.height
            
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        # Check vertical position
                        if line["bbox"][1] > page_height * 0.25:
                            continue 
                            
                        for span in line["spans"]:
                            if span["size"] > threshold and page_num > current_start:
                                boundaries.append((current_start, page_num - 1))
                                current_start = page_num
                                found_heading = True
                                break
                        if found_heading: break
                if found_heading: break
        
        boundaries.append((current_start, total_pages - 1))

    doc.close()
    
    # Final cleanup: filter invalid and merge contiguous
    valid_boundaries = []
    for start, end in boundaries:
        if end >= start:
            if valid_boundaries and valid_boundaries[-1][1] >= start - 1:
                valid_boundaries[-1] = (valid_boundaries[-1][0], max(valid_boundaries[-1][1], end))
            else:
                valid_boundaries.append((start, end))
                
    logger.info(f"Final logical boundaries: {valid_boundaries}")
    return valid_boundaries

def slice_and_rasterize_pdf(pdf_path: str, start_page: int, end_page: int) -> List[str]:
    """
    Pass 2: Dynamic PDF Slicing and Rasterization
    Slices the PDF by logical boundaries and rasterizes the pages into high-resolution images.
    """
    images = []
    doc = fitz.open(pdf_path)
    for i in range(start_page, end_page + 1):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High resolution
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pix.save(temp_file.name)
        images.append(temp_file.name)
        temp_file.close()
    doc.close()
    return images

async def upload_multiple_to_gemini_async(paths: List[str], mime_type: str = "image/png"):
    """
    Uploads a batch of images to Gemini File API asynchronously.
    """
    loop = asyncio.get_running_loop()
    files = []
    for path in paths:
        file = await loop.run_in_executor(None, lambda p=path: genai.upload_file(p, mime_type=mime_type))
        logger.info(f"Uploaded image '{file.display_name}' as: {file.uri}")
        
        while file.state.name == "PROCESSING":
            await asyncio.sleep(1)
            file = await loop.run_in_executor(None, lambda f=file.name: genai.get_file(f))
            
        if file.state.name == "FAILED":
            raise Exception(f"Image processing failed: {file.state.name}")
        
        files.append(file)
    return files

async def process_section_async(pdf_path: str, start_page: int, end_page: int, semaphore: asyncio.Semaphore) -> List[StructuralElement]:
    """
    Pass 3: Asynchronous Gemma 4 26B Processing
    Passes image batches to the Gemma 4 26B API for 2D spatial extraction.
    """
    async with semaphore:
        temp_images = []
        gemini_files = []
        try:
            # Slice and rasterize PDF
            temp_images = slice_and_rasterize_pdf(pdf_path, start_page, end_page)
            
            # Upload image batches to Gemini
            gemini_files = await upload_multiple_to_gemini_async(temp_images)
            
            # Initialize model (Gemma 4 26B)
            model = genai.GenerativeModel("gemma-4-26b")
            
            prompt = """
            Extract all text and tables from the provided document pages. 
            You must perform 2D spatial extraction. For every distinct block of text, paragraph, or table row, 
            you must provide its bounding box coordinates in the exact format: [ymin, xmin, ymax, xmax] 
            where coordinates are normalized and scaled from 0 to 1000.
            
            Classify each block as 'paragraph', 'table_row', or 'heading'.
            """
            
            content_payload = [prompt] + gemini_files
            
            response = await model.generate_content_async(
                content_payload,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionOutput
                )
            )
            
            output = ExtractionOutput.model_validate_json(response.text)
            return output.elements
        except Exception as e:
            logger.error(f"Failed processing section {start_page}-{end_page}: {e}")
            return []
        finally:
            # Cleanup local temporary images
            for temp_img in temp_images:
                if os.path.exists(temp_img):
                    os.remove(temp_img)
            # Cleanup remote Gemini files
            for g_file in gemini_files:
                try:
                    genai.delete_file(g_file.name)
                except:
                    pass

async def run_batch_extraction(pdf_path: str) -> List[StructuralElement]:
    """
    Orchestrates the two-pass batching architecture.
    """
    # 1. Detect boundaries
    boundaries = get_logical_boundaries(pdf_path)
    
    # 2. Process sections asynchronously with concurrency limit
    semaphore = asyncio.Semaphore(5)
    tasks = [process_section_async(pdf_path, start, end, semaphore) for start, end in boundaries]
    
    results = await asyncio.gather(*tasks)
    
    # Merge results sequentially to maintain original document flow
    all_elements = []
    for section_elements in results:
        all_elements.extend(section_elements)
    
    return all_elements

def ingest_reports():
    """
    Production ingestion pipeline with Two-Pass Dynamic Batching.
    """
    input_pdf_path = os.getenv("PDF_INPUT_PATH", "placeholder_path") 
    doc_title = os.path.basename(input_pdf_path) if input_pdf_path != "placeholder_path" else "Financial Report"
    
    if input_pdf_path == "placeholder_path" or not os.path.exists(input_pdf_path):
        logger.warning(f"Skipping ingestion: Path '{input_pdf_path}' not found.")
        return

    logger.info(f"Starting Two-Pass Dynamic Batching extraction on {input_pdf_path}...")
    
    # Run async extraction
    elements = asyncio.run(run_batch_extraction(input_pdf_path))
    
    logger.info(f"Extracted total of {len(elements)} structural elements.")
    
    if not elements:
        logger.error("No elements extracted. Ingestion pipeline aborted.")
        return

    # TODO: Extract real path metadata (Company, Year, etc.) dynamically from headings
    path_metadata = {"company": "Unknown", "year": 2024} 
    chunk_results = process_structural_elements(doc_title, elements, path_metadata)
    logger.info(f"Generated {len(chunk_results)} semantically grouped chunks.")
    
    if not supabase:
        logger.error("Supabase client not initialized. Cannot persist data.")
        return

    try:
        # 1. Create document record
        doc_resp = supabase.table("documents").insert({
            "title": doc_title,
            "metadata": path_metadata
        }).execute()
        
        if not doc_resp.data:
            raise Exception("Failed to create document record.")
            
        doc_id = doc_resp.data[0]["id"]
        logger.info(f"Created document record with ID: {doc_id}")
        
        # 2. Prepare chunks for batch insert
        chunks_to_insert = []
        for chunk in chunk_results:
            chunks_to_insert.append({
                "document_id": doc_id,
                "content": chunk.text,
                "context_string": chunk.context_string,
                "embedding": chunk.embedding,
                "bounding_boxes": chunk.bounding_boxes,
                "path_metadata": chunk.path_metadata
            })
            
        # 3. Batch insert chunks
        if chunks_to_insert:
            chunk_resp = supabase.table("document_chunks").insert(chunks_to_insert).execute()
            logger.info(f"Successfully batch inserted {len(chunk_resp.data)} chunks into Supabase.")
        
    except Exception as e:
        logger.error(f"Failed to persist data to Supabase: {e}")
        
    logger.info("Ingestion pipeline execution finished.")

if __name__ == "__main__":
    ingest_reports()
