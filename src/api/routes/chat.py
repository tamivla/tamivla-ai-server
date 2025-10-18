"""
Эндпоинты для работы с языковыми моделями
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger

from services.llm_service import llm_service

# Создаем router - ЭТОЙ СТРОКИ НЕ БЫЛО!
router = APIRouter(prefix="/chat", tags=["chat"])

# Pydantic модели для запросов и ответов
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500

class ChatResponse(BaseModel):
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    error: Optional[str] = None

class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500

class CompletionResponse(BaseModel):
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    error: Optional[str] = None

@router.post("/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Чат-комплишн аналогично OpenAI API
    
    - **messages**: Список сообщений в формате [{"role": "user", "content": "текст"}]
    - **model**: Опциональное имя модели
    - **temperature**: Креативность ответа (0.0-1.0)
    - **max_tokens**: Максимальное количество токенов в ответе
    """
    try:
        logger.info(f"Получен запрос на чат-комплишн, сообщений: {len(request.messages)}")
        
        # Конвертируем Pydantic модели в обычные dict
        messages_dict = [msg.dict() for msg in request.messages]
        
        result = await llm_service.chat_completion(
            messages=messages_dict,
            model_name=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка в эндпоинте чата: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@router.post("/generate", response_model=CompletionResponse)
async def generate_text(request: CompletionRequest):
    """
    Генерация текста по промпту
    
    - **prompt**: Текст-промпт для генерации
    - **model**: Опциональное имя модели
    - **temperature**: Креативность ответа
    - **max_tokens**: Максимальное количество токенов
    """
    try:
        logger.info(f"Запрос на генерацию текста: '{request.prompt[:100]}...'")
        
        result = await llm_service.generate_text(
            prompt=request.prompt,
            model_name=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return CompletionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка в эндпоинте генерации: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@router.get("/models/available")
async def get_available_models():
    """
    Получение списка доступных языковых моделей
    """
    return {
        "models": [
            {
                "name": "microsoft/DialoGPT-medium",
                "description": "Диалоговая модель от Microsoft",
                "max_tokens": 1024,
                "type": "chat"
            },
            {
                "name": "gpt2", 
                "description": "Базовая модель генерации текста",
                "max_tokens": 1024,
                "type": "completion"
            },
            {
                "name": "facebook/blenderbot-400M-distill",
                "description": "Модель для диалогов",
                "max_tokens": 512,
                "type": "chat"
            }
        ]
    }