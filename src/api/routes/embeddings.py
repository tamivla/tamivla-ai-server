# src/api/routes/embeddings.py
"""
OpenAI-compatible embeddings endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

from services.embedding_service import embedding_service

router = APIRouter(tags=["embeddings"])

class EmbeddingRequest(BaseModel):
    input: List[str]  # OpenAI format
    model: str  # Required in OpenAI format
    user: Optional[str] = None

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings compatible with OpenAI API
    
    - **input**: List of texts to embed
    - **model**: Model name (required)
    - **user**: Optional user ID
    """
    try:
        logger.info(f"OpenAI embeddings request for {len(request.input)} texts")
        
        result = await embedding_service.get_embeddings(
            texts=request.input,
            model_name=request.model
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 🔴 ИСПРАВЛЕНИЕ: embedding_service теперь возвращает OpenAI формат
        # Проверяем есть ли данные в поле "data" (OpenAI) или "embeddings" (старый)
        if "data" in result:
            # Уже в OpenAI формате - возвращаем как есть
            return EmbeddingResponse(
                data=result["data"],
                model=result["model"],
                usage=result.get("usage", {"prompt_tokens": 0, "total_tokens": 0})
            )
        elif "embeddings" in result:
            # Старый формат - конвертируем в OpenAI
            embeddings_data = []
            for i, embedding in enumerate(result["embeddings"]):
                embeddings_data.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                })
            
            return EmbeddingResponse(
                data=embeddings_data,
                model=result["model"],
                usage={
                    "prompt_tokens": result.get("texts_processed", len(request.input)),
                    "total_tokens": result.get("texts_processed", len(request.input))
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Invalid response format from embedding service")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI embeddings error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")