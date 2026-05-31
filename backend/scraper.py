import os
import time
import base64
import json
from apscheduler.schedulers.blocking import BlockingScheduler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def run_gemini_multimodal_extraction(pdf_path: str):
    """
    Simulates sending a PDF document page-by-page to Gemini Multimodal for 2D extraction.
    Instructs Gemini to return bounding box coordinates [ymin, xmin, ymax, xmax] (scaled 0-1000).
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0
    )
    
    # In a real implementation, you would chunk the PDF into images or use Gemini's File API for PDFs.
    # Here we mock the prompt assuming we pass a page image or PDF.
    
    prompt = """
    Extract all text and tables from the provided document. 
    You must perform 2D spatial extraction. For every distinct block of text, paragraph, or table row, 
    you must provide its bounding box coordinates in the exact format: [ymin, xmin, ymax, xmax] where coordinates are normalized and scaled from 0 to 1000.
    
    Format your response as a JSON list of objects:
    [
      {
        "text": "Extracted text content...",
        "bounding_box": [ymin, xmin, ymax, xmax],
        "type": "paragraph" | "table_row" | "heading"
      }
    ]
    Ensure strictly valid JSON output without any markdown wrapping (no ```json).
    """
    
    # Mocking execution for local test purposes. In real scenario:
    # message = HumanMessage(content=[
    #    {"type": "text", "text": prompt},
    #    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    # ])
    # response = llm.invoke([message])
    
    print(f"Executing Gemini Multimodal extraction on: {pdf_path}")
    # Return mock structured data for demonstration
    mock_response = [
        {
            "text": "Tata Motors Annual Report FY 24-25",
            "bounding_box": [50, 100, 100, 900],
            "type": "heading"
        },
        {
            "text": "GSTIN: 27AAACT5555C1Z1",
            "bounding_box": [120, 100, 150, 400],
            "type": "paragraph"
        }
    ]
    return mock_response

def ingest_latest_reports():
    """
    Mock ingestion task. Normally this would scrape SEBI/RBI filings.
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting nightly ingestion pipeline...")
    
    # 1. Scrape (Mocked)
    mock_pdf = "data/tata_motors_2024.pdf"
    
    # 2. Extract Multimodal content
    extracted_data = run_gemini_multimodal_extraction(mock_pdf)
    print(f"Extracted {len(extracted_data)} blocks with bounding boxes.")
    
    # 3. Validation & Supabase Storage (To be connected to services/ingestion.py and Supabase)
    print("Ingestion pipeline finished.")


if __name__ == "__main__":
    print("Starting APScheduler for Nightly Ingestion...")
    scheduler = BlockingScheduler()
    # Run every night at 2 AM
    scheduler.add_job(ingest_latest_reports, 'cron', hour=2, minute=0)
    
    # Run once immediately for local testing
    ingest_latest_reports()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
