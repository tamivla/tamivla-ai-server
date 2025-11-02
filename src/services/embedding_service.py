# src/services/embedding_service.py
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from services.model_manager import model_manager
from services.batch_processor import batch_processor

class EmbeddingService:
    """Сервис для работы с текстовыми эмбеддингами"""
    
    def __init__(self):
        self.default_model = "models--intfloat--multilingual-e5-large-instruct"
        
    async def get_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение векторных представлений для списка текстов
        """
        try:
            if not texts:
                return {
                    "object": "list",
                    "data": [],
                    "model": model_name or self.default_model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0}
                }
                
            model_to_use = model_name or self.default_model
            
            # Загружаем модель если не загружена
            if model_to_use not in model_manager.loaded_models:
                logger.info(f"Loading model: {model_to_use}")
                success = model_manager.load_model(model_to_use, "embedding")
                if not success:
                    return {
                        "object": "list",
                        "data": [],
                        "model": model_to_use,
                        "error": f"Failed to load model: {model_to_use}"
                    }
            
            # Получаем модель
            model = model_manager.loaded_models[model_to_use]['model']
            
            # 🔥 ИСПОЛЬЗУЕМ VOLUME-BASED БАТЧИНГ
            logger.info(f"🔄 Начало volume-based батчинга для {len(texts)} текстов")
            batches = batch_processor.form_batches(texts)
            logger.info(f"📦 Сформировано батчей: {len(batches)}")
            
            all_embeddings = []
            
            for i, batch in enumerate(batches):
                if batch:
                    logger.info(f"🔨 Обработка батча {i+1}/{len(batches)} размером {len(batch)} текстов")
                    batch_embeddings = model.encode(batch).tolist()
                    all_embeddings.extend(batch_embeddings)
            
            logger.info(f"✅ Volume-based батчинг завершен")
            
            # 🔴 OPENAI-СОВМЕСТИМЫЙ ФОРМАТ ОТВЕТА
            response_data = []
            for i, embedding in enumerate(all_embeddings):
                response_data.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                })
            
            total_tokens = sum(len(text) for text in texts)
            
            return {
                "object": "list",
                "data": response_data,
                "model": model_to_use,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                },
                "batches_used": len(batches)  # 🔥 НАШЕ КАСТОМНОЕ ПОЛЕ
            }
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return {
                "object": "list",
                "data": [],
                "model": model_name or self.default_model,
                "error": f"Embedding error: {str(e)}"
            }

# Глобальный экземпляр
embedding_service = EmbeddingService()