# src/main.py
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import path_fix

from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

# Только необходимые сервисы
from services.model_manager import model_manager
from services.model_discovery import model_discovery

# Эндпоинты
from api.routes.embeddings import router as embeddings_router
from api.routes.chat import router as chat_router
from api.routes.models import router as models_router

# Настройка путей
BASE_DIR = Path(__file__).parent.parent
MODELS_CACHE = BASE_DIR / "storage" / "models"
LOGS_DIR = BASE_DIR / "storage" / "logs"

# Создаем папки если их нет
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_CACHE.mkdir(parents=True, exist_ok=True)

# Настройка логгера
logger.remove()
logger.add(
    LOGS_DIR / "server.log",
    rotation="10 MB",
    retention=5,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Tamivla AI Server (OpenAI-совместимый) запускается...")
    logger.info(f"📁 Модели: {MODELS_CACHE}")
    
    # Устанавливаем переменные окружения для моделей
    os.environ['HF_HOME'] = str(MODELS_CACHE)
    os.environ['TRANSFORMERS_CACHE'] = str(MODELS_CACHE)
    
    # Сканируем модели при запуске
    discovery_result = model_discovery.scan_models_cache()
    logger.info(f"📊 Найдено моделей в кеше: {discovery_result.get('total_models', 0)}")
    
    # 🔴 ПРЕДЗАГРУЗКА ОСНОВНЫХ МОДЕЛЕЙ через model_manager
    logger.info("🔄 Предзагрузка основных моделей...")
    model_manager.preload_essential_models()
    
    yield
    
    # Shutdown
    logger.info("🛑 Tamivla AI Server останавливается...")
    for model_name in list(model_manager.loaded_models.keys()):
        model_manager.unload_model(model_name)

# Создаем приложение FastAPI
app = FastAPI(
    title="Tamivla AI Server",
    description="OpenAI-совместимый API сервер для AI моделей",
    version="1.0.0",
    lifespan=lifespan
)

# OpenAI-совместимые роутеры
app.include_router(embeddings_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1/chat")
app.include_router(models_router, prefix="/v1")

# КАСТОМНЫЕ эндпоинты для управления моделями (БЕЗ префикса /v1)
app.include_router(models_router)

# Базовые эндпоинты
@app.get("/")
async def root():
    return {
        "message": "Tamivla AI Server работает!",
        "version": "1.0.0",
        "openai_endpoints": {
            "embeddings": "/v1/embeddings",
            "chat": "/v1/chat/completions", 
            "models": "/v1/models",
            "docs": "/docs"
        },
        "custom_endpoints": {
            "load_model": "/models/load",
            "unload_model": "/models/unload", 
            "loaded_models": "/models/loaded"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Tamivla AI Server"}

def start_server():
    try:
        logger.info("Запуск Tamivla AI Server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_config=None
        )
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        sys.exit(1)

# Временный эндпоинт для диагностики
@app.get("/debug/routes")
async def debug_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": getattr(route, "path", None),
            "name": getattr(route, "name", None),
            "methods": getattr(route, "methods", None)
        })
    return {"routes": routes}

if __name__ == "__main__":
    start_server()