# src/services/llm_service.py
"""
Сервис языковых моделей для Tamivla AI Server
Только эмбеддинги - LLM временно отключен
"""

from typing import List, Dict, Any, Optional
from loguru import logger

class LLMService:
    """Сервис для работы с языковыми моделями - временно только заглушка"""
    
    def __init__(self):
        logger.info("🤖 LLM Service: временно отключен, работаем только с эмбеддингами")
        
    async def chat_completion(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Заглушка для чат-комплишн"""
        return {
            "error": "LLM service temporarily disabled - embeddings only",
            "choices": []
        }
    
    async def generate_text(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Заглушка для генерации текста"""
        return {
            "error": "LLM service temporarily disabled - embeddings only", 
            "choices": []
        }
    
    async def health_check(self) -> bool:
        """Всегда здоров"""
        return True
    
    async def close(self):
        """Ничего не закрываем"""
        pass

# Глобальный экземпляр сервиса LLM
llm_service = LLMService()