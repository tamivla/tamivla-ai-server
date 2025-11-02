# src/api/routes/chat.py
"""
OpenAI-compatible chat endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger

from services.llm_service import llm_service

router = APIRouter(tags=["chat"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str  # Required in OpenAI format
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@router.post("/completions", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    """
    Create chat completion compatible with OpenAI API
    
    - **model**: Model name (required)
    - **messages**: Conversation history
    - **temperature**: Creativity level
    - **max_tokens**: Maximum tokens to generate
    """
    try:
        logger.info(f"OpenAI chat completion request for model: {request.model}")
        
        # Convert to service format
        messages_dict = [msg.dict() for msg in request.messages]
        
        result = await llm_service.chat_completion(
            messages=messages_dict,
            model_name=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 500
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Transform to OpenAI format
        import time
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=result["model"],
            choices=result["choices"],
            usage=result["usage"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI chat completion error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")