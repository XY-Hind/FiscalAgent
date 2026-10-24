import os
import operator
from typing import Annotated, Sequence, TypedDict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from supabase import create_client, Client
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

load_dotenv()

from backend.models.output import AnswerWithEvidence
from backend.models.validation import output_guard
from backend.utils.logger import logger

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
supabase_key = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_SECRET_KEY")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
    or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
)
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

# Initialize FlashRank (lightweight local reranker)
ranker = Ranker()

# Standard Safety Settings for Google GenAI Models
DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# -----------------
# 1. State Definition
# -----------------
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    metadata_filters: dict
    active_document_id: str | None
    retrieved_docs: List[dict]
    retrieved_context: str
    context_is_relevant: bool
    final_output: AnswerWithEvidence | None
    reflection_attempts: int
    retry_count: int

# -----------------
# 2. Nodes
# -----------------

class QueryFilters(BaseModel):
    optimized_query: str = Field(description="The rewritten search query optimized for embedding and keyword search.")
    metadata_filters: dict = Field(description="Extracted metadata filters like company, year, section, gstin. Keys must match path_metadata keys.")

def parse_query(state: GraphState):
    """
    Uses Groq API (qwen/qwen3.8-27b) to parse the user question into optimized keywords and strict metadata filters.
    """
    logger.info("Parsing user query for metadata extraction...")
    try:
        llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        user_message = state["messages"][-1].content
        
        llm_with_tool = llm.with_structured_output(QueryFilters)
        prompt = (
            f"You are a financial query parser. Analyze this user query and extract an optimized search query "
            f"and any explicit metadata filters (company, year, section, gstin).\n\n"
            f"Query: {user_message}"
        )
        
        parsed = llm_with_tool.invoke(prompt)
        optimized_query = parsed.optimized_query
        filters = parsed.metadata_filters
        logger.info(f"Extracted filters: {filters}")
    except Exception as e:
        if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
            logger.error("Groq API limit reached during query parsing.")
            # Standard error message for UI to catch
            optimized_query = "ERROR_API_LIMIT_REACHED: Groq"
            filters = {}
        else:
            logger.error(f"Query parsing failed: {e}")
            optimized_query = state["messages"][-1].content
            filters = {}
    
    return {
        "query": optimized_query, 
        "metadata_filters": filters,
        "reflection_attempts": state.get("reflection_attempts", 0)
    }

def retrieve(state: GraphState):
    """
    Retrieves documents from Supabase using the RRF RPC function.
    """
    query = state["query"]
    filters = state["metadata_filters"]
    
    if "ERROR_API_LIMIT_REACHED" in query:
        return {"retrieved_docs": [], "retrieved_context": query}

    logger.info(f"Retrieving chunks for query: '{query}' with filters: {filters}")
    
    if not supabase:
        logger.error("Supabase client not initialized.")
        return {"retrieved_docs": [], "retrieved_context": "Supabase client not initialized."}

    from backend.services.ingestion import get_gemini_embeddings_model
    embeddings_model = get_gemini_embeddings_model()
    
    try:
        query_embedding = embeddings_model.embed_query(query)
        
        response = supabase.rpc("match_chunks_rrf", {
            "query_text": query,
            "query_embedding": query_embedding,
            "metadata_filter": filters,
            "match_count": 25,
            "rrf_k": 60,
            "filter_document_id": state.get("active_document_id")
        }).execute()
        
        docs = response.data or []
        logger.info(f"Retrieved {len(docs)} raw chunks from Supabase.")
        return {"retrieved_docs": docs}
            
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            logger.error("Gemini API limit reached during retrieval (embedding).")
            return {"retrieved_docs": [], "retrieved_context": "ERROR_API_LIMIT_REACHED: Gemini"}
        logger.error(f"Supabase retrieval failed: {e}")
        return {"retrieved_docs": [], "retrieved_context": f"Error during retrieval: {str(e)}"}

def rerank(state: GraphState):
    """
    Reranks the retrieved documents using FlashRank for higher precision.
    """
    query = state["query"]
    docs = state["retrieved_docs"]
    context = state.get("retrieved_context", "")
    
    if "ERROR_API_LIMIT_REACHED" in context:
        return {"retrieved_context": context}
    
    if not docs:
        logger.warning("No documents to rerank.")
        return {"retrieved_context": "No relevant documents found."}
    
    logger.info(f"Reranking {len(docs)} documents using FlashRank...")
    
    try:
        passages = []
        for doc in docs:
            passages.append({
                "id": doc.get("chunk_id"),
                "text": doc.get("content"),
                "meta": {
                    "title": doc.get("document_title"),
                    "path": doc.get("path_metadata")
                }
            })
        
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)
        
        top_results = reranked_results[:10]
        logger.info(f"Reranking complete. Top score: {top_results[0].get('score') if top_results else 'N/A'}")
        
        context_pieces = []
        for res in top_results:
            meta = res.get("meta", {})
            context_pieces.append(
                f"Document: {meta.get('title', 'Unknown')}\n"
                f"Path: {meta.get('path', {})}\n"
                f"Content:\n{res.get('text')}"
            )
        
        retrieved_context = "\n\n---\n\n".join(context_pieces)
        return {"retrieved_context": retrieved_context}
    except Exception as e:
        logger.error(f"FlashRank reranking failed: {e}")
        return {"retrieved_context": "\n\n---\n\n".join([d.get("content") for d in docs[:10]])}

class Grade(BaseModel):
    is_relevant: bool = Field(description="True if the context contains the answer, False otherwise")

def grade_context(state: GraphState):
    """
    Explicitly grade the context for relevance using Gemma 4 31B.
    """
    retry_count = state.get("retry_count", 0) + 1

    if "ERROR_API_LIMIT_REACHED" in state["retrieved_context"]:
        return {"context_is_relevant": True, "retry_count": retry_count} # Bypass to generate_answer to show error

    logger.info("Grading retrieved context relevance...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemma-4-31b", 
            temperature=0, 
            safety_settings=DEFAULT_SAFETY_SETTINGS
        )
        llm_with_tool = llm.with_structured_output(Grade)
        
        prompt = f"Does the following context answer the user query?\nQuery: {state['query']}\nContext: {state['retrieved_context']}"
        result = llm_with_tool.invoke(prompt)
        is_relevant = result.is_relevant
        logger.info(f"Context relevance grade: {is_relevant}")
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            logger.error("Gemini API limit reached during context grading.")
            return {"retrieved_context": "ERROR_API_LIMIT_REACHED: Gemini", "context_is_relevant": True, "retry_count": retry_count}
        logger.error(f"Context grading failed: {e}")
        is_relevant = True
    
    return {"context_is_relevant": is_relevant, "retry_count": retry_count}

def reflect_and_rewrite(state: GraphState):
    """
    If context is poor, reflect on the failure and rewrite the query using Gemma 4 31B.
    """
    logger.info(f"Context irrelevant. Rewriting query (Attempt {state['reflection_attempts'] + 1})...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemma-4-31b", 
            temperature=0, 
            safety_settings=DEFAULT_SAFETY_SETTINGS
        )
        prompt = f"The previous query '{state['query']}' did not retrieve relevant context. Rewrite it to be more specific or use different keywords."
        result = llm.invoke(prompt)
        new_query = result.content.strip()
        logger.info(f"New optimized query: '{new_query}'")
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            logger.error("Gemini API limit reached during query rewrite.")
            new_query = "ERROR_API_LIMIT_REACHED: Gemini"
        else:
            new_query = state["query"]
    
    return {
        "query": new_query,
        "reflection_attempts": state["reflection_attempts"] + 1
    }

def generate_answer(state: GraphState):
    """
    Generate structured output based on context using Gemma 4 31B, validated.
    """
    context = state["retrieved_context"]
    
    # Handle API Limit propagation
    if "ERROR_API_LIMIT_REACHED" in context:
        provider = context.split(": ")[1]
        final_output = AnswerWithEvidence(
            answer=f"⚠️ {provider} API limit reached. Please wait a moment and try again.", 
            reasoning=f"The ingestion/generation pipeline hit the rate limit for {provider}.", 
            citations=[]
        )
        return {"final_output": final_output}

    logger.info("Generating final answer with evidence...")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemma-4-31b", 
            temperature=0, 
            streaming=True, 
            safety_settings=DEFAULT_SAFETY_SETTINGS
        )
        llm_with_schema = llm.with_structured_output(AnswerWithEvidence)
        
        prompt = f"Answer the query based ONLY on the context. Provide exact quotes and bounding boxes if present in the context.\nQuery: {state['query']}\nContext: {context}"
        
        result = llm_with_schema.invoke(prompt)
        
        guard_result = output_guard.validate(result.model_dump())
        if guard_result.validation_passed:
            final_output = AnswerWithEvidence(**guard_result.validated_output)
            logger.info("Final answer successfully validated.")
        else:
            logger.warning("Final answer failed validation. Returning raw output.")
            final_output = result
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            logger.error("Gemini API limit reached during answer generation.")
            final_output = AnswerWithEvidence(
                answer="⚠️ Gemini API limit reached. Please wait a moment and try again.", 
                reasoning="Rate limit hit during final synthesis.", 
                citations=[]
            )
        else:
            logger.error(f"Answer generation failed: {e}")
            final_output = AnswerWithEvidence(answer=f"Error generating answer: {str(e)}", reasoning="", citations=[])
    
    return {"final_output": final_output}

# -----------------
# 3. Edges & Graph
# -----------------

def decide_next_step(state: GraphState):
    if state["context_is_relevant"]:
        return "generate_answer"
    if state.get("retry_count", 0) >= 3:
        return "fallback_node"
    return "reflect_and_rewrite"

def fallback_node(state: GraphState):
    """
    Fallback node when maximum retries are reached.
    """
    logger.warning("Max retry count reached. Returning fallback response.")
    final_output = AnswerWithEvidence(
        answer="I cannot find the exact answer in the provided documents.",
        reasoning="Max retry count reached and no relevant context found.",
        citations=[]
    )
    return {"final_output": final_output}

workflow = StateGraph(GraphState)

workflow.add_node("parse_query", parse_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("grade_context", grade_context)
workflow.add_node("reflect_and_rewrite", reflect_and_rewrite)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("fallback_node", fallback_node)

workflow.set_entry_point("parse_query")
workflow.add_edge("parse_query", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade_context")
workflow.add_conditional_edges("grade_context", decide_next_step)
workflow.add_edge("reflect_and_rewrite", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("fallback_node", END)

# Compile graph
rag_agent = workflow.compile()