"""
Эндпоинты для работы с эмбеддингами
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger

from services.embedding_service import embedding_service

# Создаем router - ЭТОЙ СТРОКИ НЕ БЫЛО!
router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# Pydantic модели для запросов и ответов
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    model: str
    embeddings: List[List[float]]
    dimensions: int
    count: int
    texts_processed: int
    error: Optional[str] = None

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    model: Optional[str] = None

class SimilarityResponse(BaseModel):
    similarity: float
    text1: str
    text2: str
    model: str
    error: Optional[str] = None

@router.post("/", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    Получение векторных представлений для списка текстов
    
    - **texts**: Список текстов для векторизации
    - **model**: Опциональное имя модели (по умолчанию: all-MiniLM-L6-v2)
    """
    try:
        logger.info(f"Получен запрос на эмбеддинги для {len(request.texts)} текстов")
        
        result = await embedding_service.get_embeddings(
            texts=request.texts,
            model_name=request.model
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return EmbeddingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка в эндпоинте эмбеддингов: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    Вычисление семантической схожести между двумя текстами
    
    - **text1**: Первый текст для сравнения
    - **text2**: Второй текст для сравнения  
    - **model**: Опциональное имя модели
    """
    try:
        logger.info(f"Запрос на вычисление схожести: '{request.text1[:50]}...' vs '{request.text2[:50]}...'")
        
        result = await embedding_service.get_similarity(
            text1=request.text1,
            text2=request.text2,
            model_name=request.model
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return SimilarityResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка в эндпоинте схожести: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@router.get("/models/available")
async def get_available_models():
    """
    Получение списка доступных моделей для эмбеддингов
    """
    return {
        "models": [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Легкая модель для общего использования (384 размерности)",
                "max_tokens": 256,
                "dimensions": 384
            },
            {
                "name": "sentence-transformers/all-mpnet-base-v2", 
                "description": "Более качественная модель (768 размерности)",
                "max_tokens": 384,
                "dimensions": 768
            },
            {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "Мультиязычная модель (384 размерности)",
                "max_tokens": 128,
                "dimensions": 384
            }
        ]
    }