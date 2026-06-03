import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langfuse import observe

from backend.middleware.token_budget import token_budgeter
from backend.middleware.injection_defense import injection_defense
from backend.services.graph import rag_agent
from backend.utils.logger import logger

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@observe()
async def generate_chat_stream(request: ChatRequest, checkpointer) -> AsyncGenerator[str, None]:
    """
    Generator for Server-Sent Events (SSE) streaming of the LangGraph output.
    Uses LangGraph's native astream_events for true token streaming.
    """
    logger.info(f"Received chat request for session: {request.session_id}")
    
    # 1. Defense-in-Depth validation
    try:
        token_budgeter.validate_budget(request.message)
        secured_prompt = injection_defense.secure_prompt(request.message)
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return
    
    # 2. Prepare Graph State
    config = {"configurable": {"thread_id": request.session_id}, "checkpointer": checkpointer}
    input_state = {
        "messages": [HumanMessage(content=secured_prompt)]
    }
    
    # 3. Stream LangGraph execution
    try:
        logger.info(f"Starting LangGraph stream for session: {request.session_id}")
        # True token streaming without simulated delays
        async for event in rag_agent.astream_events(input_state, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
            elif kind == "on_chain_end" and event["name"] == "generate_answer":
                output = event["data"].get("output")
                if output and "final_output" in output:
                    final_answer = output["final_output"]
                    if final_answer:
                        yield f"data: {json.dumps({'type': 'final', 'data': final_answer.model_dump()})}\n\n"
                    
        logger.info(f"LangGraph stream complete for session: {request.session_id}")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error during LangGraph streaming for session {request.session_id}: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest):
    """
    FastAPI SSE Endpoint for the React frontend.
    """
    checkpointer = request.app.state.checkpointer
    if not checkpointer:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    
    return StreamingResponse(
        generate_chat_stream(payload, checkpointer), 
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"}
    )
