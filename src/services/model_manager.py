"""
Менеджер моделей для Tamivla AI Server
Динамическая загрузка и выгрузка AI-моделей
"""

import os
import gc
from typing import Dict, Any, Optional
from loguru import logger
from pathlib import Path

class ModelManager:
    """Управление жизненным циклом AI-моделей"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.models_cache = Path(os.environ.get('HF_HOME', 'storage/models'))
        
    def load_embedding_model(self, model_name: str, **kwargs) -> bool:
        """Загрузка модели для эмбеддингов"""
        try:
            if model_name in self.loaded_models:
                logger.info(f"Модель {model_name} уже загружена")
                return True
                
            logger.info(f"Загрузка модели эмбеддингов: {model_name}")
            
            # Здесь будет реализация загрузки через sentence-transformers
            # Пока заглушка
            self.loaded_models[model_name] = {
                'type': 'embedding',
                'status': 'loaded',
                'model': None  # Будет заменено на реальную модель
            }
            
            logger.success(f"Модель {model_name} успешно загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            return False
    
    def load_llm_model(self, model_name: str, **kwargs) -> bool:
        """Загрузка языковой модели"""
        try:
            if model_name in self.loaded_models:
                logger.info(f"Модель {model_name} уже загружена")
                return True
                
            logger.info(f"Загрузка языковой модели: {model_name}")
            
            # Здесь будет реализация загрузки через transformers
            # Пока заглушка
            self.loaded_models[model_name] = {
                'type': 'llm', 
                'status': 'loaded',
                'model': None  # Будет заменено на реальную модель
            }
            
            logger.success(f"Языковая модель {model_name} успешно загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки языковой модели {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Выгрузка модели из памяти"""
        try:
            if model_name not in self.loaded_models:
                logger.warning(f"Модель {model_name} не найдена в загруженных")
                return False
                
            logger.info(f"Выгрузка модели: {model_name}")
            
            # Освобождаем ресурсы
            model_info = self.loaded_models.pop(model_name)
            if model_info.get('model'):
                del model_info['model']
                
            # Принудительная сборка мусора
            gc.collect()
            
            logger.success(f"Модель {model_name} успешно выгружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка выгрузки модели {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Получение информации о модели"""
        return self.loaded_models.get(model_name)
    
    def list_loaded_models(self) -> Dict[str, str]:
        """Список загруженных моделей"""
        return {name: info['type'] for name, info in self.loaded_models.items()}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Статистика по моделям"""
        return {
            'total_loaded': len(self.loaded_models),
            'embedding_models': sum(1 for info in self.loaded_models.values() 
                                  if info['type'] == 'embedding'),
            'llm_models': sum(1 for info in self.loaded_models.values() 
                            if info['type'] == 'llm'),
            'models': self.list_loaded_models()
        }

# Глобальный экземпляр менеджера моделей
model_manager = ModelManager()