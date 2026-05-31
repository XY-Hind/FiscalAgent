import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langfuse.decorators import observe

from backend.security.token_budget import token_budgeter
from backend.security.injection_defense import injection_defense
from backend.services.graph import rag_agent

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@observe()
async def generate_chat_stream(request: ChatRequest, checkpointer) -> AsyncGenerator[str, None]:
    """
    Generator for Server-Sent Events (SSE) streaming of the LangGraph output.
    """
    # 1. Defense-in-Depth validation
    token_budgeter.validate_budget(request.message)
    secured_prompt = injection_defense.secure_prompt(request.message)
    
    # 2. Prepare Graph State
    config = {"configurable": {"thread_id": request.session_id}, "checkpointer": checkpointer}
    input_state = {
        "messages": [HumanMessage(content=secured_prompt)]
    }
    
    # 3. Stream LangGraph execution
    # For a real streaming scenario with structured output, we would stream the `generate_answer` 
    # node's token chunks. Since `rag_agent.astream` yields state updates, we mock token-by-token.
    try:
        async for event in rag_agent.astream(input_state, config=config, stream_mode="updates"):
            # Serialize state updates as SSE
            data = json.dumps(str(event))
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.01) # Small delay for demonstration
            
            # If we reach the final output, we extract it.
            if "generate_answer" in event:
                final_answer = event["generate_answer"].get("final_output")
                if final_answer:
                    # Stream the final validated AnswerWithEvidence structure
                    yield f"data: {final_answer.model_dump_json()}\n\n"
                    
        yield "data: [DONE]\n\n"
    except Exception as e:
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
