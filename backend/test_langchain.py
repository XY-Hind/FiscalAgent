import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document",
        output_dimensionality=768
    )
    res = embeddings.embed_documents(["hello world"])
    print(len(res[0]))
except Exception as e:
    print(f"Failed with output_dimensionality: {e}")

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document",
        client_options={"output_dimensionality": 768}
    )
    res = embeddings.embed_documents(["hello world"])
    print(len(res[0]))
except Exception as e:
    print(f"Failed with client_options: {e}")
