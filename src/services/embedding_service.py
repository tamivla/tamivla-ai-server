"""
Сервис эмбеддингов для Tamivla AI Server
Векторизация текстов с использованием sentence-transformers
"""

from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
from .model_manager import model_manager

class EmbeddingService:
    """Сервис для работы с текстовыми эмбеддингами"""
    
    def __init__(self):
        self.default_model = "sentence-transformers/all-MiniLM-L6-v2"
        
    async def get_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение векторных представлений для списка текстов
        
        Args:
            texts: Список текстов для векторизации
            model_name: Имя модели (если None, используется модель по умолчанию)
            
        Returns:
            Словарь с эмбеддингами и метаинформацией
        """
        try:
            if not texts:
                return {
                    "error": "Пустой список текстов",
                    "embeddings": []
                }
                
            model_to_use = model_name or self.default_model
            
            # Проверяем загружена ли модель
            if not model_manager.get_model_info(model_to_use):
                logger.info(f"Модель {model_to_use} не загружена, пытаемся загрузить...")
                if not model_manager.load_embedding_model(model_to_use):
                    return {
                        "error": f"Не удалось загрузить модель {model_to_use}",
                        "embeddings": []
                    }
            
            # Здесь будет реальная логика векторизации
            # Пока возвращаем заглушки
            embeddings = self._generate_dummy_embeddings(texts)
            
            return {
                "model": model_to_use,
                "embeddings": embeddings,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "count": len(embeddings),
                "texts_processed": len(texts)
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов: {e}")
            return {
                "error": f"Внутренняя ошибка сервера: {str(e)}",
                "embeddings": []
            }
    
    def _generate_dummy_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Генерация заглушечных эмбеддингов для тестирования
        В реальной реализации будет заменено на sentence-transformers
        """
        # Простая заглушка - случайные векторы фиксированной размерности
        embedding_size = 384  # Размерность all-MiniLM-L6-v2
        embeddings = []
        
        for text in texts:
            # Генерируем псевдо-случайный вектор на основе текста
            # для детерминированности в тестовом режиме
            seed = hash(text) % 10000
            np.random.seed(seed)
            embedding = np.random.randn(embedding_size).tolist()
            embeddings.append(embedding)
            
        return embeddings
    
    async def get_similarity(self, text1: str, text2: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Вычисление семантической схожести между двумя текстами
        """
        try:
            # Получаем эмбеддинги для обоих текстов
            result = await self.get_embeddings([text1, text2], model_name)
            
            if "error" in result:
                return {"error": result["error"]}
                
            embeddings = result["embeddings"]
            
            if len(embeddings) != 2:
                return {"error": "Не удалось получить эмбеддинги для сравнения"}
            
            # Вычисляем косинусное сходство
            vec1 = np.array(embeddings[0])
            vec2 = np.array(embeddings[1])
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            return {
                "similarity": float(similarity),
                "text1": text1,
                "text2": text2,
                "model": result["model"]
            }
            
        except Exception as e:
            logger.error(f"Ошибка вычисления схожести: {e}")
            return {"error": f"Ошибка вычисления схожести: {str(e)}"}

# Глобальный экземпляр сервиса эмбеддингов
embedding_service = EmbeddingService()