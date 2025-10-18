"""
Эндпоинты для управления моделями через веб-интерфейс
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger

from services.model_discovery import model_discovery
from services.model_manager import model_manager

router = APIRouter(prefix="/models", tags=["models"])

# Pydantic модели для запросов
class DownloadModelRequest(BaseModel):
    model_id: str
    quantize: Optional[bool] = False
    quantization_bits: Optional[int] = 8

class ModelActionRequest(BaseModel):
    model_name: str

# Эндпоинты для веб-интерфейса
@router.get("/discovery")
async def discover_models():
    """
    Сканирование локального кеша моделей
    
    Возвращает:
    - Список всех моделей в кеше
    - Размеры, типы, метаданные
    - Информацию о файлах
    """
    try:
        logger.info("Запрос на сканирование моделей в кеше")
        return model_discovery.scan_models_cache()
        
    except Exception as e:
        logger.error(f"Ошибка сканирования моделей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сканирования: {str(e)}")

@router.get("/resources")
async def get_system_resources():
    """
    Получение информации о системных ресурсах
    
    Возвращает:
    - CPU, RAM использование
    - GPU информацию (память, загрузка)
    - Совместимость с моделями
    """
    try:
        logger.info("Запрос информации о системных ресурсах")
        return model_discovery.get_system_resources()
        
    except Exception as e:
        logger.error(f"Ошибка получения ресурсов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения ресурсов: {str(e)}")

@router.post("/download")
async def download_model(request: DownloadModelRequest):
    """
    Загрузка модели из HuggingFace Hub
    
    - **model_id**: Идентификатор модели на HF (например: "microsoft/DialoGPT-medium")
    - **quantize**: Автоматическое квантование при загрузке
    - **quantization_bits**: Уровень квантования (4, 8, 16)
    """
    try:
        logger.info(f"Запрос на загрузку модели: {request.model_id}")
        
        # Здесь будет реализация загрузки через huggingface_hub
        # Пока возвращаем заглушку
        
        return {
            "status": "success",
            "message": f"Модель {request.model_id} поставлена в очередь на загрузку",
            "model_id": request.model_id,
            "quantize": request.quantize,
            "quantization_bits": request.quantization_bits,
            "download_id": f"dl_{hash(request.model_id) % 10000}"  # Временный ID для отслеживания
        }
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {request.model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {str(e)}")

@router.post("/load")
async def load_model_to_memory(request: ModelActionRequest):
    """
    Загрузка модели в оперативную память/VRAM
    
    - **model_name**: Имя модели из локального кеша
    """
    try:
        logger.info(f"Запрос на загрузку модели в память: {request.model_name}")
        
        # Проверяем существует ли модель в кеше
        cache_info = model_discovery.scan_models_cache()
        model_exists = any(model['name'] == request.model_name for model in cache_info.get('models', []))
        
        if not model_exists:
            raise HTTPException(status_code=404, detail=f"Модель {request.model_name} не найдена в локальном кеше")
        
        # Пытаемся загрузить модель
        # Здесь будет логика определения типа модели и выбора соответствующего загрузчика
        success = model_manager.load_embedding_model(request.model_name)  # Временная заглушка
        
        if success:
            return {
                "status": "success",
                "message": f"Модель {request.model_name} загружена в память",
                "model_name": request.model_name
            }
        else:
            raise HTTPException(status_code=500, detail=f"Не удалось загрузить модель {request.model_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки модели в память: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки в память: {str(e)}")

@router.post("/unload")
async def unload_model_from_memory(request: ModelActionRequest):
    """
    Выгрузка модели из памяти
    
    - **model_name**: Имя загруженной модели
    """
    try:
        logger.info(f"Запрос на выгрузку модели из памяти: {request.model_name}")
        
        success = model_manager.unload_model(request.model_name)
        
        if success:
            return {
                "status": "success", 
                "message": f"Модель {request.model_name} выгружена из памяти",
                "model_name": request.model_name
            }
        else:
            return {
                "status": "warning",
                "message": f"Модель {request.model_name} не была загружена или уже выгружена",
                "model_name": request.model_name
            }
            
    except Exception as e:
        logger.error(f"Ошибка выгрузки модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка выгрузки: {str(e)}")

@router.get("/loaded")
async def get_loaded_models():
    """
    Получение списка моделей загруженных в память
    """
    try:
        loaded_models = model_manager.list_loaded_models()
        model_stats = model_manager.get_model_stats()
        
        return {
            "loaded_models": loaded_models,
            "stats": model_stats
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения списка загруженных моделей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка: {str(e)}")

@router.delete("/cache/{model_name}")
async def delete_model_from_cache(model_name: str):
    """
    Удаление модели из локального кеша
    
    - **model_name**: Имя модели для удаления
    """
    try:
        logger.warning(f"Запрос на удаление модели из кеша: {model_name}")
        
        # Здесь будет реализация удаления папки с моделью
        # Пока возвращаем заглушку
        
        return {
            "status": "success",
            "message": f"Модель {model_name} будет удалена из кеша",
            "model_name": model_name,
            "warning": "Функция удаления в разработке"
        }
        
    except Exception as e:
        logger.error(f"Ошибка удаления модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка удаления: {str(e)}")

@router.get("/recommendations")
async def get_model_recommendations():
    """
    Получение рекомендаций по моделям на основе системных ресурсов
    """
    try:
        resources = model_discovery.get_system_resources()
        
        recommendations = {
            "based_on_resources": {
                "max_model_size_gb": 2.0,  # Примерная логика расчета
                "recommended_quantization": "8-bit",
                "suitable_for_llm": True,
                "suitable_for_embeddings": True
            },
            "popular_models": [
                {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "type": "embeddings",
                    "size_gb": 0.09,
                    "recommended": True
                },
                {
                    "name": "microsoft/DialoGPT-medium", 
                    "type": "chat",
                    "size_gb": 0.8,
                    "recommended": True
                }
            ]
        }
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Ошибка получения рекомендаций: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка рекомендаций: {str(e)}")