import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# Фикс путей для службы
import path_fix

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

# Импорты сервисов
from services.model_manager import model_manager
from services.model_discovery import model_discovery
from services.quantization_service import quantization_service

# Импорты эндпоинтов
from api.routes.embeddings import router as embeddings_router
from api.routes.chat import router as chat_router
from api.routes.models import router as models_router
from api.routes.model_dashboard import router as dashboard_router
from api.routes.test_dashboard import router as test_router
from api.routes.quantization import router as quantization_router
from api.routes.quantization_dashboard import router as quantization_dashboard_router

# Настройка путей
BASE_DIR = Path(__file__).parent.parent
MODELS_CACHE = BASE_DIR / "storage" / "models"
LOGS_DIR = BASE_DIR / "storage" / "logs"
STATIC_DIR = BASE_DIR / "templates"

# Создаем папки если их нет
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_CACHE.mkdir(parents=True, exist_ok=True)

# Настройка логгера
logger.remove()  # Убираем стандартный handler
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
    logger.info("🚀 Tamivla AI Server запускается...")
    logger.info(f"📁 Модели: {MODELS_CACHE}")
    logger.info(f"📁 Логи: {LOGS_DIR}")
    logger.info(f"📁 Статические файлы: {STATIC_DIR}")
    
    # Устанавливаем переменные окружения для моделей
    os.environ['HF_HOME'] = str(MODELS_CACHE)
    os.environ['TRANSFORMERS_CACHE'] = str(MODELS_CACHE)
    
    # Инициализация сервисов
    logger.info("🔄 Инициализация менеджера моделей...")
    logger.info("🔍 Сканирование доступных моделей...")
    
    # Сканируем модели при запуске
    discovery_result = model_discovery.scan_models_cache()
    logger.info(f"📊 Найдено моделей в кеше: {discovery_result.get('total_models', 0)}")
    
    # Проверяем системные ресурсы и GPU
    resources = model_discovery.get_system_resources()
    gpu_info = quantization_service.get_gpu_memory_info()
    
    if 'error' not in resources:
        logger.info(f"💻 Доступно RAM: {resources['memory']['available_gb']:.1f} GB")
        if resources['gpu']:
            for gpu_id, gpu_info in resources['gpu'].items():
                logger.info(f"🎮 {gpu_id}: {gpu_info['name']} ({gpu_info['memory_total_gb']:.1f} GB)")
    
    # Логируем информацию о GPU для квантования - ИСПРАВЛЕННАЯ ВЕРСИЯ
    if gpu_info and gpu_info.get('available'):
        for gpu_id, gpu_details in gpu_info.get('gpus', {}).items():
            logger.info(f"⚡ {gpu_id}: {gpu_details.get('free_gb', 0):.1f} GB свободно из {gpu_details.get('total_gb', 0):.1f} GB")
    else:
        error_msg = gpu_info.get('error', 'Unknown error') if gpu_info else 'GPU info not available'
        logger.warning(f"❌ GPU не доступен для квантования: {error_msg}")
    
    yield  # Сервер работает
    
    # Shutdown
    logger.info("🛑 Tamivla AI Server останавливается...")
    # Очистка моделей при остановке
    for model_name in list(model_manager.loaded_models.keys()):
        model_manager.unload_model(model_name)

# Создаем приложение FastAPI
app = FastAPI(
    title="Tamivla AI Server",
    description="Высокопроизводительный сервер для AI моделей от Tamivla Industrial Group",
    version="1.0.0",
    lifespan=lifespan
)

# Подключаем статические файлы (CSS, JS, изображения)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Подключаем все роутеры
app.include_router(embeddings_router)
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(dashboard_router)
app.include_router(test_router)
app.include_router(quantization_router)
app.include_router(quantization_dashboard_router)

# Базовые эндпоинты
@app.get("/")
async def root():
    return {
        "message": "Tamivla AI Server работает!",
        "version": "1.0.0",
        "manufacturer": "Tamivla Industrial Group",
        "endpoints": {
            "embeddings": "/embeddings",
            "chat": "/chat", 
            "models": "/models",
            "dashboard": "/dashboard",
            "quantization": "/quantization",
            "quantization-dashboard": "/quantization-dashboard",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "Tamivla AI Server",
        "models_path": str(MODELS_CACHE),
        "logs_path": str(LOGS_DIR)
    }

@app.get("/system/status")
async def system_status():
    """Статус системы и загруженные модели"""
    # Получаем информацию о ресурсах
    resources = model_discovery.get_system_resources()
    gpu_info = quantization_service.get_gpu_memory_info()
    
    return {
        "status": "running",
        "service": "Tamivla AI Server",
        "models_cache": str(MODELS_CACHE),
        "loaded_models": model_manager.list_loaded_models(),
        "model_stats": model_manager.get_model_stats(),
        "system_resources": resources,
        "gpu_detailed_info": gpu_info
    }

# Функция для запуска сервера (вызывается извне)
def start_server():
    try:
        logger.info("Starting Tamivla AI Server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_config=None
        )
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
        sys.exit(1)

# Запуск напрямую (для ручного запуска)
if __name__ == "__main__":
    start_server()