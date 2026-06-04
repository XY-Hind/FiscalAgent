import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from backend.utils.logger import logger

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
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document"
    )

def generate_context_string(document_title: str, raw_chunk: str) -> str:
    """
    Uses Groq API (Llama-3.1-8B-Instant) for fast contextual retrieval summary.
    """
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        prompt = (
            f"You are a financial analyst. Below is a chunk of text from the document '{document_title}'.\n"
            f"Provide a concise 1-2 sentence summary that situates this chunk within the broader document context "
            f"to improve search accuracy. Do not repeat the text. Focus on entities, dates, and core financial themes.\n\n"
            f"Chunk Content:\n{raw_chunk}\n\n"
            f"Context Summary:"
        )
        
        response = llm.invoke(prompt)
        return response.content.strip() if response.content else ""
    except Exception as e:
        if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
            logger.error("Groq API limit reached during context generation.")
            # We don't raise here to allow ingestion to continue if possible, 
            # but in a production batch script, you might want to stop.
            return "Context generation skipped due to API limit."
        logger.error(f"Context generation failed: {e}")
        return ""

def process_structural_elements(document_title: str, elements: List[StructuralElement], path_metadata: Dict[str, Any]) -> List[ChunkResult]:
    """
    Robust layout-aware grouping logic. 
    Groups structural elements based on type, spatial proximity, and semantic flow.
    Batches embeddings into arrays of 100 for high throughput using Gemini Embedding 2.
    """
    logger.info(f"Processing {len(elements)} structural elements for '{document_title}'")
    embeddings_model = get_gemini_embeddings_model()
    results = []
    
    grouped_chunks = []
    current_group = []
    
    for i, el in enumerate(elements):
        if el.type == "heading" and len(current_group) > 2:
            grouped_chunks.append(list(current_group))
            current_group = []
            
        current_group.append(el)
        
        if len(current_group) >= 15:
            grouped_chunks.append(list(current_group))
            current_group = []
            
    if current_group:
        grouped_chunks.append(current_group)
        
    logger.info(f"Grouped elements into {len(grouped_chunks)} chunks.")

    pending_chunks = []
    enriched_texts = []
    
    for idx, chunk in enumerate(grouped_chunks):
        try:
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
                
            encoded_text = "\n".join(toon_parts)
            context_str = generate_context_string(document_title, encoded_text)
            
            enriched_text = f"Context: {context_str}\n\n{encoded_text}"
            chunk_bboxes = [e.bounding_box for e in chunk]
            
            pending_chunks.append({
                "text": encoded_text,
                "context_string": context_str,
                "bounding_boxes": chunk_bboxes
            })
            enriched_texts.append(enriched_text)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Prepared {idx + 1}/{len(grouped_chunks)} chunks...")
        except Exception as e:
            logger.error(f"Failed to prepare chunk {idx} for '{document_title}': {e}")
            continue

    logger.info("Batch generating embeddings using Gemini Embedding 2 (text-embedding-004)...")
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
                # Stop processing further chunks in this document to respect the limit
                break
            logger.error(f"Failed to generate embeddings for batch {i // batch_size + 1}: {e}")

    logger.info(f"Successfully processed {len(results)} chunks for '{document_title}'")
    return results

def process_document_text(document_title: str, full_text: str, path_metadata: Dict[str, Any]) -> List[ChunkResult]:
    """
    Fallback entry point. Converts raw text to basic structural elements if needed.
    """
    # Simple line-based layout simulation for fallback
    lines = full_text.split('\n')
    elements = []
    for line in lines:
        if not line.strip(): continue
        
        # Heuristic for headings (e.g. short lines, all caps)
        el_type = "paragraph"
        if len(line.strip()) < 60 and line.strip().isupper():
            el_type = "heading"
            
        elements.append(StructuralElement(
            text=line.strip(),
            bounding_box=[0,0,0,0], # Unknown spatial info in raw text
            type=el_type
        ))
        
    return process_structural_elements(document_title, elements, path_metadata)
