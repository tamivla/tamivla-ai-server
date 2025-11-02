# src/services/model_manager.py
"""
Менеджер моделей для Tamivla AI Server
Управление загрузкой и выгрузкой AI-моделей ТОЛЬКО из локального кеша
"""

import os
import gc
import torch
from typing import Dict, Any, Optional
from loguru import logger
from pathlib import Path

class ModelManager:
    """Управление жизненным циклом AI-моделей"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.models_cache = Path(os.environ.get('HF_HOME', 'storage/models'))
        
    def load_model(self, model_name: str, model_type: str, **kwargs) -> bool:
        try:
            if model_name in self.loaded_models:
                logger.info(f"Модель {model_name} уже загружена")
                return True
                
            logger.info(f"Загрузка {model_type} модели: {model_name}")
            
            # ЖЕСТКАЯ ПРОВЕРКА: модель ДОЛЖНА существовать локально
            local_path = self._get_local_model_path(model_name)
            if not local_path:
                logger.error(f"🚫 ЗАПРЕЩЕНО: Модель {model_name} не найдена в локальном кеше")
                return False
            
            # ЗАПРЕТ на автозагрузку через переменные окружения
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            # ЗАГРУЖАЕМ ИСКЛЮЧИТЕЛЬНО ИЗ ЛОКАЛЬНОГО ПУТИ!
            if model_type == 'embedding':
                from sentence_transformers import SentenceTransformer
                try:
                    # ЗАГРУЖАЕМ ПРЯМО ИЗ ПУТИ!
                    model = SentenceTransformer(
                        str(local_path),  # ← ВОТ ОНО! ЛОКАЛЬНЫЙ ПУТЬ!
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                except Exception as e:
                    logger.error(f"Ошибка загрузки SentenceTransformer: {e}")
                    return False
                    
            elif model_type == 'llm':
                from transformers import pipeline
                model = pipeline(
                    "text-generation",
                    model=str(local_path),  # ← ВОТ ОНО! ЛОКАЛЬНЫЙ ПУТЬ!
                    tokenizer=str(local_path),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
            
            self.loaded_models[model_name] = {
                'type': model_type,
                'status': 'loaded',
                'model': model,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'local_path': str(local_path)
            }
            
            logger.success(f"Модель {model_name} успешно загружена из локального кеша")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            return False
    
    def _convert_cache_name_to_model_name(self, cache_name: str) -> str:
        """
        Преобразует имя из кеша в имя для загрузки ИСКЛЮЧИТЕЛЬНО по стандарту HF
        """
        # ТОЛЬКО стандартный формат HF!
        if cache_name.startswith('models--'):
            return cache_name.replace('models--', '').replace('--', '/')
        
        # Если не стандартный формат - ОШИБКА!
        raise ValueError(f"Модель {cache_name} в нестандартном формате HF")
    
    def _get_local_model_path(self, model_name: str) -> Optional[Path]:
        """Получаем локальный путь к модели по СТАНДАРТНОМУ ФОРМАТУ HF"""
        
        # ТОЛЬКО стандартный формат: models--author--model-name
        if not model_name.startswith('models--'):
            # Пробуем конвертировать красивое имя в стандартный формат
            if '/' in model_name:
                model_name = f"models--{model_name.replace('/', '--')}"
            else:
                logger.error(f"Модель {model_name} в нестандартном формате")
                return None
        
        путь = self.models_cache / model_name
        if путь.exists():
            logger.info(f"Найден стандартный путь: {путь}")
            return путь
        
        logger.error(f"Модель {model_name} не найдена в стандартном формате")
        return None
    
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.success(f"Модель {model_name} успешно выгружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка выгрузки модели {model_name}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Получение загруженной модели"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]['model']
        return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Получение информации о модели"""
        return self.loaded_models.get(model_name)
    
    def list_loaded_models(self) -> Dict[str, str]:
        """Список загруженных моделей"""
        return {name: info['type'] for name, info in self.loaded_models.items()}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Базовая статистика по моделям"""
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