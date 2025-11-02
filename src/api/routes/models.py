# src/api/routes/models.py
"""
OpenAI-совместимые models endpoints + кастомные эндпоинты управления
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from loguru import logger

from services.model_manager import model_manager
from services.model_discovery import model_discovery

router = APIRouter(tags=["models"])

class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tamivla"

class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[ModelResponse]

class LoadModelRequest(BaseModel):
    model_name: str
    model_type: str  # "embedding" or "llm"

class DownloadModelRequest(BaseModel):
    model_id: str  # Format: "author/model-name"

# === OPENAI-СОВМЕСТИМЫЕ ЭНДПОИНТЫ ===
@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List available models (OpenAI-compatible)
    Returns ONLY models that are locally available in STANDARD HF format
    """
    try:
        # Get ONLY locally cached models
        cache_info = model_discovery.scan_models_cache()
        models_data = []
        
        for model in cache_info.get("models", []):
            # ФИЛЬТРУЕМ: оставляем ТОЛЬКО стандартные форматы HF
            if model["name"].startswith('models--'):
                models_data.append(ModelResponse(
                    id=model["name"],  # Use the actual cached name
                    created=1700000000,
                    owned_by="tamivla"
                ))
        
        return ModelsListResponse(data=models_data)
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# === КАСТОМНЫЕ ЭНДПОИНТЫ ДЛЯ УПРАВЛЕНИЯ ===
@router.post("/models/load")
async def load_model(request: LoadModelRequest):
    """Load model into memory - ONLY if exists locally"""
    try:
        # Check if model exists locally first
        cache_info = model_discovery.scan_models_cache()
        available_models = [model["name"] for model in cache_info.get("models", [])]
        
        if request.model_name not in available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found in local cache"
            )
        
        success = model_manager.load_model(
            model_name=request.model_name,
            model_type=request.model_type
        )
        
        if success:
            return {
                "status": "success", 
                "message": f"Model {request.model_name} loaded",
                "model_name": request.model_name
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load model {request.model_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/models/unload")
async def unload_model(model_name: str):
    """Unload model from memory"""
    try:
        success = model_manager.unload_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} unloaded", 
                "model_name": model_name
            }
        else:
            return {
                "status": "warning", 
                "message": f"Model {model_name} was not loaded",
                "model_name": model_name
            }
            
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/models/loaded")
async def get_loaded_models():
    """Get currently loaded models"""
    return {
        "loaded_models": model_manager.list_loaded_models(),
        "stats": model_manager.get_model_stats()
    }

@router.post("/models/download")
async def download_model(request: DownloadModelRequest):
    """Download model from HuggingFace Hub"""
    try:
        logger.info(f"Downloading model: {request.model_id}")
        
        success = model_discovery.download_model(request.model_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Model {request.model_id} downloaded successfully",
                "model_id": request.model_id
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to download model {request.model_id}"
            )
            
    except Exception as e:
        logger.error(f"Error downloading model {request.model_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# === НОВЫЕ КАСТОМНЫЕ ЭНДПОИНТЫ ДЛЯ КЕША ===
@router.get("/cache/info")
async def get_cache_info():
    """Получение детальной информации о кеше моделей"""
    try:
        cache_info = model_discovery.scan_models_cache()
        return cache_info
    except Exception as e:
        logger.error(f"Ошибка получения информации о кеше: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/cache/analyze")
async def analyze_cache():
    """Анализ кеша на наличие битых моделей"""
    try:
        result = model_discovery.analyze_model_cache()
        return result
    except Exception as e:
        logger.error(f"Ошибка анализа кеша: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/cache/{model_name}")
async def delete_model_from_cache(model_name: str):
    """Удаление модели из локального кеша"""
    try:
        # Используем метод delete_model из model_discovery
        success = model_discovery.delete_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Модель {model_name} удалена из кеша",
                "model_name": model_name
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Модель {model_name} не найдена в кеше"
            )
            
    except Exception as e:
        logger.error(f"Ошибка удаления модели {model_name}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

@router.get("/debug/delete-test/{model_name}")
async def debug_delete_test(model_name: str):
    """Тест удаления модели"""
    try:
        local_path = model_discovery._get_local_model_path(model_name)
        return {
            "model_name": model_name,
            "local_path": str(local_path) if local_path else None,
            "exists": local_path.exists() if local_path else False
        }
    except Exception as e:
        return {"error": str(e)}