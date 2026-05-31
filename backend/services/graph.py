import os
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from backend.models.output import AnswerWithEvidence
from backend.services.ingestion import get_gemini_embeddings_model

# -----------------
# 1. State Definition
# -----------------
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    metadata_filters: dict
    retrieved_context: str
    context_is_relevant: bool
    final_output: AnswerWithEvidence | None
    reflection_attempts: int

# -----------------
# 2. Nodes
# -----------------

def parse_query(state: GraphState):
    """
    Parses the raw user question into optimized keywords and strict metadata filters.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    user_message = state["messages"][-1].content
    
    # In a real app, we'd use LLM with structured output to extract filters (e.g. {"gstin": "...", "year": 2024})
    # Mocking extraction for demonstration
    optimized_query = user_message
    filters = {"year": 2024} # Mock
    
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
    # Mock retrieval from Supabase RRF
    retrieved_docs = "Mock context: Tata Motors revenue increased by 20% in FY 24-25. Exact Quote: 'Revenue increased by 20%'"
    return {"retrieved_context": retrieved_docs}

class Grade(BaseModel):
    is_relevant: bool = Field(description="True if the context contains the answer, False otherwise")

def grade_context(state: GraphState):
    """
    Explicitly grade the context for relevance.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    llm_with_tool = llm.with_structured_output(Grade)
    
    prompt = f"Does the following context answer the user query?\nQuery: {state['query']}\nContext: {state['retrieved_context']}"
    result = llm_with_tool.invoke(prompt)
    
    return {"context_is_relevant": result.is_relevant}

def reflect_and_rewrite(state: GraphState):
    """
    If context is poor, reflect on the failure and rewrite the query.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    prompt = f"The previous query '{state['query']}' did not retrieve relevant context. Rewrite it to be more specific or use different keywords."
    result = llm.invoke(prompt)
    
    return {
        "query": result.content.strip(),
        "reflection_attempts": state["reflection_attempts"] + 1
    }

def generate_answer(state: GraphState):
    """
    Generate structured output (AnswerWithEvidence) based on context.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    llm_with_schema = llm.with_structured_output(AnswerWithEvidence)
    
    prompt = f"Answer the query based ONLY on the context. Provide exact quotes and bounding boxes if present in the context.\nQuery: {state['query']}\nContext: {state['retrieved_context']}"
    
    result = llm_with_schema.invoke(prompt)
    
    # In a real app, this is where Guardrails AI would intercept the structured output
    # to verify exact string quotes against `state['retrieved_context']`.
    
    return {"final_output": result}

# -----------------
# 3. Edges & Graph
# -----------------

def decide_next_step(state: GraphState):
    if state["context_is_relevant"]:
        return "generate_answer"
    if state["reflection_attempts"] >= 3:
        return "generate_answer" # Fallback, try best effort
    return "reflect_and_rewrite"

workflow = StateGraph(GraphState)

workflow.add_node("parse_query", parse_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_context", grade_context)
workflow.add_node("reflect_and_rewrite", reflect_and_rewrite)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("parse_query")
workflow.add_edge("parse_query", "retrieve")
workflow.add_edge("retrieve", "grade_context")
workflow.add_conditional_edges("grade_context", decide_next_step)
workflow.add_edge("reflect_and_rewrite", "retrieve")
workflow.add_edge("generate_answer", END)

# Compile graph
rag_agent = workflow.compile()
