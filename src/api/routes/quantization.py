"""
Эндпоинты для управления квантованием моделей
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from loguru import logger

from services.quantization_service import quantization_service

router = APIRouter(prefix="/quantization", tags=["quantization"])

# Pydantic модели для запросов
class QuantizationAnalysisRequest(BaseModel):
    model_name: str
    target_device: Optional[str] = "cuda:0"

class QuantizationSuggestionResponse(BaseModel):
    model_name: str
    estimated_size_gb: float
    can_load: bool
    best_recommendation: Dict[str, Any]
    alternative_recommendations: List[Dict[str, Any]]
    suggestions: List[str]
    gpu_info: Dict[str, Any]

class BatchQuantizationAnalysisRequest(BaseModel):
    model_names: List[str]

@router.post("/analyze", response_model=QuantizationSuggestionResponse)
async def analyze_model_quantization(request: QuantizationAnalysisRequest):
    """
    Анализ возможности загрузки модели и рекомендации по квантованию
    
    - **model_name**: Имя модели для анализа
    - **target_device**: Целевое GPU устройство (по умолчанию: cuda:0)
    """
    try:
        logger.info(f"Анализ квантования для модели: {request.model_name}")
        
        # Получаем анализ квантования
        analysis = quantization_service.generate_quantization_suggestions(request.model_name)
        quantization_analysis = analysis["quantization_analysis"]
        
        # Формируем альтернативные рекомендации
        alternative_recommendations = [
            rec for rec in quantization_analysis["recommendations"] 
            if rec["can_fit"] and rec != quantization_analysis["best_recommendation"]
        ][:3]  # Ограничиваем 3 альтернативами
        
        return QuantizationSuggestionResponse(
            model_name=analysis["model_name"],
            estimated_size_gb=analysis["estimated_size_gb"],
            can_load=quantization_analysis["can_load"],
            best_recommendation=quantization_analysis["best_recommendation"],
            alternative_recommendations=alternative_recommendations,
            suggestions=analysis["suggestions"],
            gpu_info=quantization_service.get_gpu_memory_info()
        )
        
    except Exception as e:
        logger.error(f"Ошибка анализа квантования для {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

@router.post("/analyze-batch")
async def analyze_batch_quantization(request: BatchQuantizationAnalysisRequest):
    """
    Пакетный анализ нескольких моделей
    """
    try:
        logger.info(f"Пакетный анализ квантования для {len(request.model_names)} моделей")
        
        results = {}
        for model_name in request.model_names:
            try:
                analysis = quantization_service.generate_quantization_suggestions(model_name)
                results[model_name] = {
                    "estimated_size_gb": analysis["estimated_size_gb"],
                    "can_load": analysis["quantization_analysis"]["can_load"],
                    "best_recommendation": analysis["quantization_analysis"]["best_recommendation"],
                    "suggestions": analysis["suggestions"][:3]  # Первые 3 предложения
                }
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        return {
            "total_models": len(request.model_names),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Ошибка пакетного анализа: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка пакетного анализа: {str(e)}")

@router.get("/gpu-info")
async def get_detailed_gpu_info():
    """
    Получение детальной информации о GPU
    """
    try:
        gpu_info = quantization_service.get_gpu_memory_info()
        return gpu_info
    except Exception as e:
        logger.error(f"Ошибка получения информации о GPU: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации о GPU: {str(e)}")

@router.get("/recommendations/popular-models")
async def get_popular_models_recommendations():
    """
    Рекомендации по квантованию для популярных моделей
    """
    popular_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/DialoGPT-medium", 
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "intfloat/multilingual-e5-large-instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/flan-t5-large"
    ]
    
    try:
        results = {}
        for model in popular_models:
            try:
                analysis = quantization_service.generate_quantization_suggestions(model)
                results[model] = {
                    "estimated_size_gb": analysis["estimated_size_gb"],
                    "can_load": analysis["quantization_analysis"]["can_load"],
                    "best_recommendation": analysis["quantization_analysis"]["best_recommendation"]
                }
            except Exception as e:
                results[model] = {"error": str(e)}
        
        return {
            "popular_models": results,
            "gpu_info": quantization_service.get_gpu_memory_info()
        }
        
    except Exception as e:
        logger.error(f"Ошибка генерации рекомендаций: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации рекомендаций: {str(e)}")

@router.get("/model/{model_name}/quantization-options")
async def get_model_quantization_options(model_name: str):
    """
    Получение всех вариантов квантования для конкретной модели
    """
    try:
        model_size = quantization_service.get_model_size_estimation(model_name)
        quantization_analysis = quantization_service.calculate_optimal_quantization(model_size)
        
        return {
            "model_name": model_name,
            "estimated_size_gb": model_size,
            "quantization_options": quantization_analysis["recommendations"],
            "gpu_constraints": {
                "free_vram_gb": quantization_analysis["free_vram_gb"],
                "total_vram_gb": quantization_analysis["total_vram_gb"]
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения вариантов квантования для {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения вариантов: {str(e)}")

@router.post("/model/{model_name}/quantize")
async def quantize_model(model_name: str, quantization_level: str = "8bit"):
    """
    Запуск процесса квантования модели
    
    - **model_name**: Имя модели для квантования
    - **quantization_level**: Уровень квантования (4bit, 8bit, fp16, etc.)
    """
    try:
        logger.info(f"Запрос на квантование модели {model_name} с уровнем {quantization_level}")
        
        # Проверяем существует ли уже квантованная версия
        if quantization_service.is_model_quantized(model_name, quantization_level):
            return {
                "status": "already_exists",
                "message": f"Квантованная версия {model_name} ({quantization_level}) уже существует",
                "model_name": model_name,
                "quantization_level": quantization_level
            }
        
        # Здесь будет реализация процесса квантования
        # Пока возвращаем заглушку
        
        return {
            "status": "queued",
            "message": f"Модель {model_name} поставлена в очередь на квантование {quantization_level}",
            "model_name": model_name,
            "quantization_level": quantization_level,
            "estimated_size_gb": quantization_service.get_model_size_estimation(model_name),
            "quantized_path": str(quantization_service.get_quantized_model_path(model_name, quantization_level))
        }
        
    except Exception as e:
        logger.error(f"Ошибка квантования модели {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка квантования: {str(e)}")

@router.get("/quantized-models")
async def list_quantized_models():
    """
    Список уже квантованных моделей
    """
    try:
        quantized_models = []
        quantized_dir = quantization_service.quantized_models_cache
        
        if quantized_dir.exists():
            for model_dir in quantized_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    # Извлекаем оригинальное имя модели и уровень квантования
                    if '--' in model_name:
                        parts = model_name.split('--')
                        if len(parts) >= 2:
                            original_name = '--'.join(parts[:-1]).replace('--', '/')
                            quant_level = parts[-1]
                            
                            quantized_models.append({
                                "original_name": original_name,
                                "quantized_name": model_name,
                                "quantization_level": quant_level,
                                "path": str(model_dir),
                                "size_mb": quantization_service.get_directory_size_mb(model_dir) if hasattr(quantization_service, 'get_directory_size_mb') else 0
                            })
        
        return {
            "quantized_models": quantized_models,
            "total_quantized": len(quantized_models)
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения списка квантованных моделей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка: {str(e)}")