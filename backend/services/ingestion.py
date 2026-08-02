import os
import json
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from backend.utils.logger import logger
from backend.utils.parsers import encode_to_toon_table

class ChunkResult(BaseModel):
    text: str
    context_string: str
    embedding: List[float]
    path_metadata: Dict[str, Any] = Field(default_factory=dict)
    bounding_boxes: List[list] | None = None

class StructuralElement(BaseModel):
    text: str
    bounding_box: List[int]
    type: str # "paragraph" | "table_row" | "heading"

def get_gemini_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document",
        output_dimensionality=768
    )

# 1. Global LLM Initialization to reuse the connection pool
try:
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM globally: {e}")
    groq_llm = None

# 2. Smart Rate Limiter for the Free Tier
MAX_CONCURRENT_REQUESTS = 3
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def _async_generate_single_context(llm, document_title: str, raw_chunk: str, max_retries: int = 3) -> str:
    """Asynchronously generates a context string with exponential backoff for 429 limits."""
    if not llm:
        return ""
        
    for attempt in range(max_retries):
        # The semaphore prevents bursting the API by holding chunks in a queue
        async with api_semaphore:
            try:
                # Tiny baseline delay to prevent sudden spikes
                await asyncio.sleep(0.5) 
                
                prompt = (
                    f"You are a financial analyst. Below is a chunk of text from the document '{document_title}'.\n"
                    f"Provide a concise 1-2 sentence summary that situates this chunk within the broader document context "
                    f"to improve search accuracy. Do not repeat the text. Focus on entities, dates, and core financial themes.\n\n"
                    f"Chunk Content:\n{raw_chunk}\n\n"
                    f"Context Summary:"
                )
                
                response = await llm.ainvoke(prompt)
                return response.content.strip() if response.content else ""
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate_limit" in error_msg or "quota" in error_msg:
                    if attempt < max_retries - 1:
                        # Wait 20 seconds, then 40 seconds to let the Groq RPM limit reset
                        wait_time = (attempt + 1) * 20
                        logger.warning(f"Groq Free Tier RPM hit. Yielding for {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error("Groq API limit reached. Max retries exhausted.")
                        return "Context generation skipped due to API limit."
                
                logger.error(f"Context generation failed: {e}")
                return ""

async def _async_batch_generate_context(document_title: str, raw_chunks: List[str]) -> List[str]:
    """Fires requests concurrently under semaphore protection."""
    tasks = [_async_generate_single_context(groq_llm, document_title, chunk) for chunk in raw_chunks]
    return await asyncio.gather(*tasks)

def process_structural_elements(document_title: str, elements: List[StructuralElement], path_metadata: Dict[str, Any]) -> List[ChunkResult]:
    """
    Robust layout-aware grouping logic. 
    Groups structural elements based on type, spatial proximity, and semantic flow.
    Applies Header Injection for tables and Sliding Window Overlap for paragraphs.
    Batches API requests asynchronously with backoff, then batches embeddings.
    """
    logger.info(f"Processing {len(elements)} structural elements for '{document_title}'")
    embeddings_model = get_gemini_embeddings_model()
    results = []
    
    grouped_chunks = []
    current_group = []
    
    # Track the active table header to inject if a table gets split
    current_table_header = None
    
    for i, el in enumerate(elements):
        if el.type == "table_row":
            if i == 0 or elements[i-1].type != "table_row":
                current_table_header = el
        else:
            current_table_header = None

        if el.type == "heading" and len(current_group) > 2:
            grouped_chunks.append(list(current_group))
            current_group = []
            
        current_group.append(el)
        
        if len(current_group) >= 15:
            grouped_chunks.append(list(current_group))
            
            # Smart split logic
            if el.type == "table_row" and current_table_header:
                current_group = [current_table_header]
            else:
                current_group = list(current_group[-3:])
            
    if current_group:
        grouped_chunks.append(current_group)
        
    logger.info(f"Grouped elements into {len(grouped_chunks)} chunks.")

    # Prepare all text blocks
    encoded_texts = []
    chunk_bboxes_list = []
    
    for chunk in grouped_chunks:
        table_rows = [e for e in chunk if e.type == "table_row"]
        others = [e for e in chunk if e.type != "table_row"]
        
        toon_parts = []
        for item in others:
            if item.type == "heading":
                toon_parts.append(f"heading: {item.text}")
            else:
                toon_parts.append(f"paragraph: {item.text}")
                
        if table_rows:
            rows_data = [{"content": tr.text} for tr in table_rows]
            toon_parts.append(encode_to_toon_table("extracted_table", rows_data))
            
        encoded_texts.append("\n".join(toon_parts))
        chunk_bboxes_list.append([e.bounding_box for e in chunk])

    # Run asynchronous context generation concurrently
    logger.info("Generating contextual summaries concurrently via Groq...")
    # Since upload_document is synchronous, it runs in a FastAPI threadpool without an active event loop.
    # Therefore, we can safely use asyncio.run to execute the async context generation block.
    context_strings = asyncio.run(_async_batch_generate_context(document_title, encoded_texts))

    # Recombine the concurrent results
    pending_chunks = []
    enriched_texts = []
    
    for idx, encoded_text in enumerate(encoded_texts):
        context_str = context_strings[idx]
        enriched_text = f"Context: {context_str}\n\n{encoded_text}"
        
        pending_chunks.append({
            "text": encoded_text,
            "context_string": context_str,
            "bounding_boxes": chunk_bboxes_list[idx]
        })
        enriched_texts.append(enriched_text)

    logger.info("Batch generating embeddings using Gemini Embedding 2 (gemini-embedding-2)...")
    batch_size = 100
    
    for i in range(0, len(enriched_texts), batch_size):
        batch_texts = enriched_texts[i:i + batch_size]
        try:
            vectors = embeddings_model.embed_documents(batch_texts)
            
            for j, vector in enumerate(vectors):
                chunk_data = pending_chunks[i + j]
                results.append(
                    ChunkResult(
                        text=chunk_data["text"],
                        context_string=chunk_data["context_string"],
                        embedding=vector,
                        path_metadata=path_metadata,
                        bounding_boxes=chunk_data["bounding_boxes"]
                    )
                )
            logger.info(f"Embedded batch {i // batch_size + 1}")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                logger.error(f"Gemini API limit reached during embedding generation for batch {i // batch_size + 1}.")
                break
            logger.error(f"Failed to generate embeddings for batch {i // batch_size + 1}: {e}")

    logger.info(f"Successfully processed {len(results)} chunks for '{document_title}'")
    return results

def process_document_text(document_title: str, full_text: str, path_metadata: Dict[str, Any]) -> List[ChunkResult]:
    """
    Fallback entry point. Converts raw text to basic structural elements if needed.
    """
    lines = full_text.split('\n')
    elements = []
    for line in lines:
        if not line.strip(): continue
        
        el_type = "paragraph"
        if len(line.strip()) < 60 and line.strip().isupper():
            el_type = "heading"
            
        elements.append(StructuralElement(
            text=line.strip(),
            bounding_box=[0,0,0,0], 
            type=el_type
        ))
        
    return process_structural_elements(document_title, elements, path_metadata)