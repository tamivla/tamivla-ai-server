"""
Сервис языковых моделей для Tamivla AI Server
Чат-интерфейсы и генерация текста
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from .model_manager import model_manager

class LLMService:
    """Сервис для работы с языковыми моделями"""
    
    def __init__(self):
        self.default_model = "microsoft/DialoGPT-medium"
        
    async def chat_completion(self, messages: List[Dict[str, str]], model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Чат-комплишн по аналогии с OpenAI API
        
        Args:
            messages: Список сообщений в формате [{"role": "user", "content": "текст"}]
            model_name: Имя модели
            **kwargs: Дополнительные параметры (temperature, max_tokens и т.д.)
            
        Returns:
            Ответ модели в стандартизированном формате
        """
        try:
            if not messages:
                return {
                    "error": "Пустой список сообщений",
                    "choices": []
                }
                
            model_to_use = model_name or self.default_model
            
            # Проверяем загружена ли модель
            if not model_manager.get_model_info(model_to_use):
                logger.info(f"Модель {model_to_use} не загружена, пытаемся загрузить...")
                if not model_manager.load_llm_model(model_to_use):
                    return {
                        "error": f"Не удалось загрузить модель {model_to_use}",
                        "choices": []
                    }
            
            # Здесь будет реальная логика работы с моделью
            # Пока возвращаем заглушку
            response_text = self._generate_dummy_response(messages)
            
            return {
                "model": model_to_use,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "length",
                        "index": 0
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(messages)),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(str(messages)) + len(response_text.split())
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка чат-комплишн: {e}")
            return {
                "error": f"Внутренняя ошибка сервера: {str(e)}",
                "choices": []
            }
    
    def _generate_dummy_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Генерация заглушечного ответа для тестирования
        В реальной реализации будет заменено на transformers
        """
        last_message = messages[-1]["content"] if messages else ""
        
        # Простые ответы на основе ключевых слов
        if "привет" in last_message.lower():
            return "Привет! Я Tamivla AI Assistant. Чем могу помочь?"
        elif "как дела" in last_message.lower():
            return "У меня все отлично! Готов помочь вам с любыми вопросами."
        elif "погода" in last_message.lower():
            return "К сожалению, у меня нет доступа к данным о погоде. Но я могу помочь с другими вопросами!"
        else:
            return f"Я получил ваше сообщение: '{last_message}'. В текущей тестовой версии я использую заглушечные ответы. Реальная модель будет подключена позже."
    
    async def generate_text(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Генерация текста по промпту
        
        Args:
            prompt: Текст-промпт для генерации
            model_name: Имя модели
            **kwargs: Дополнительные параметры
            
        Returns:
            Сгенерированный текст
        """
        try:
            # Используем тот же механизм что и для чата
            messages = [{"role": "user", "content": prompt}]
            return await self.chat_completion(messages, model_name, **kwargs)
            
        except Exception as e:
            logger.error(f"Ошибка генерации текста: {e}")
            return {
                "error": f"Ошибка генерации текста: {str(e)}",
                "generated_text": ""
            }

# Глобальный экземпляр сервиса LLM
llm_service = LLMService()