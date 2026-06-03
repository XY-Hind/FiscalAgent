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
    Uses Groq API for fast contextual retrieval summary.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", # Or latest Llama-3 on Groq
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Best practice: Show, don't just describe. Contextual Retrieval pattern.
    prompt = (
        f"You are a financial analyst. Below is a chunk of text from the document '{document_title}'.\n"
        f"Provide a concise 1-2 sentence summary that situates this chunk within the broader document context "
        f"to improve search accuracy. Do not repeat the text. Focus on entities, dates, and core financial themes.\n\n"
        f"Chunk Content:\n{raw_chunk}\n\n"
        f"Context Summary:"
    )
    
    response = llm.invoke(prompt)
    return response.content.strip() if response.content else ""

def encode_to_toon_table(table_name: str, rows: List[Dict[str, Any]]) -> str:
    """
    Encodes a list of dictionaries into TOON (Token-Oriented Object Notation) table format.
    Syntax: key[count]{field1,field2}:
    """
    if not rows:
        return ""
    
    count = len(rows)
    fields = list(rows[0].keys())
    header = ",".join(fields)
    
    # Using comma as default, but TOON allows tab for better token efficiency if needed
    toon_lines = [f"{table_name}[{count}]{{{header}}}:"]
    for row in rows:
        values = [str(row.get(f, "")).replace(",", "\\,") for f in fields]
        toon_lines.append(",".join(values))
        
    return "\n".join(toon_lines)

def process_structural_elements(document_title: str, elements: List[StructuralElement], path_metadata: Dict[str, Any]) -> List[ChunkResult]:
    """
    Robust layout-aware grouping logic. 
    Groups structural elements based on type, spatial proximity, and semantic flow.
    """
    logger.info(f"Processing {len(elements)} structural elements for '{document_title}'")
    embeddings_model = get_gemini_embeddings_model()
    results = []
    
    grouped_chunks = []
    current_group = []
    
    for i, el in enumerate(elements):
        # 1. Start a new group for headings if the current group is substantial
        if el.type == "heading" and len(current_group) > 2:
            grouped_chunks.append(list(current_group))
            current_group = []
            
        current_group.append(el)
        
        # 2. Heuristic: cap the number of elements per chunk to manage token limits
        if len(current_group) >= 15:
            grouped_chunks.append(list(current_group))
            current_group = []
            
    if current_group:
        grouped_chunks.append(current_group)
        
    logger.info(f"Grouped elements into {len(grouped_chunks)} chunks.")

    for idx, chunk in enumerate(grouped_chunks):
        try:
            # Separate tables from paragraphs for TOON encoding
            table_rows = [e for e in chunk if e.type == "table_row"]
            others = [e for e in chunk if e.type != "table_row"]
            
            toon_parts = []
            
            # Encode non-table elements as TOON objects/fields
            for item in others:
                if item.type == "heading":
                    toon_parts.append(f"heading: {item.text}")
                else:
                    toon_parts.append(f"paragraph: {item.text}")
                    
            # Encode table rows as a single TOON table
            if table_rows:
                rows_data = [{"content": tr.text} for tr in table_rows]
                toon_parts.append(encode_to_toon_table("extracted_table", rows_data))
                
            encoded_text = "\n".join(toon_parts)
            
            # Generate contextual retrieval string via Groq
            context_str = generate_context_string(document_title, encoded_text)
            
            # Final text for embedding: Context + TOON Data
            enriched_text = f"Context: {context_str}\n\n{encoded_text}"
            
            # Generate 768-dim embeddings via Gemini
            vector = embeddings_model.embed_query(enriched_text)
            
            # Aggregate bounding boxes for this chunk
            chunk_bboxes = [e.bounding_box for e in chunk]
            
            results.append(
                ChunkResult(
                    text=encoded_text,
                    context_string=context_str,
                    embedding=vector,
                    path_metadata=path_metadata,
                    bounding_boxes=chunk_bboxes
                )
            )
            if (idx + 1) % 5 == 0:
                logger.info(f"Processed {idx + 1}/{len(grouped_chunks)} chunks...")
        except Exception as e:
            logger.error(f"Failed to process chunk {idx} for '{document_title}': {e}")
            continue
            
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
