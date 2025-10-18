"""
Сервис обнаружения и анализа моделей в кеше
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

class ModelDiscoveryService:
    """Обнаружение и анализ моделей в локальном кеше"""
    
    def __init__(self):
        self.models_cache = Path(os.environ.get('HF_HOME', 'storage/models'))
        
    def scan_models_cache(self) -> Dict[str, Any]:
        """
        Сканирование папки с моделями и сбор информации
        """
        try:
            logger.info(f"Сканирование кеша моделей: {self.models_cache}")
            
            if not self.models_cache.exists():
                return {"error": "Папка с моделями не существует", "models": []}
            
            models_info = []
            total_folders = 0
            
            # Сканируем подпапки (каждая модель в отдельной папке)
            for model_dir in self.models_cache.iterdir():
                if model_dir.is_dir():
                    total_folders += 1
                    # Фильтруем только реальные модели (игнорируем служебные папки)
                    if self._is_real_model_directory(model_dir):
                        model_info = self.analyze_model_directory(model_dir)
                        if model_info:
                            models_info.append(model_info)
            
            return {
                "cache_path": str(self.models_cache),
                "total_folders": total_folders,  # Всего папок (включая служебные)
                "total_models": len(models_info),  # Только реальные модели
                "models": models_info
            }
            
        except Exception as e:
            logger.error(f"Ошибка сканирования моделей: {e}")
            return {"error": str(e), "models": []}
    
    def _is_real_model_directory(self, model_dir: Path) -> bool:
        """
        Проверяет является ли папка реальной моделью (а не служебной)
        """
        dir_name = model_dir.name
        
        # Игнорируем служебные папки
        ignore_patterns = [
            '.locks',
            'hub',
            'quantized',
            '__pycache__',
            '.git',
            '.vscode'
        ]
        
        if any(pattern in dir_name for pattern in ignore_patterns):
            return False
            
        # Папки с моделями обычно содержат 'models--' в названии
        # или имеют специфичную структуру файлов
        if 'models--' in dir_name:
            return True
            
        # Дополнительная проверка: есть ли внутри файлы модели
        model_files = list(model_dir.rglob('*.safetensors')) + list(model_dir.rglob('*.bin'))
        config_files = list(model_dir.rglob('config.json'))
        
        return len(model_files) > 0 or len(config_files) > 0
    
    def analyze_model_directory(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Анализ отдельной папки с моделью
        """
        try:
            model_name = model_dir.name
            
            # Базовая информация
            info = {
                "name": model_name,
                "display_name": self._get_display_name(model_name),
                "path": str(model_dir),
                "size_mb": self.get_directory_size_mb(model_dir),
                "files": [],
                "type": "unknown",
                "is_real_model": True
            }
            
            # Анализ файлов в папке
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    file_info = {
                        "name": file_path.name,
                        "size_mb": file_path.stat().st_size / (1024 * 1024),
                        "relative_path": str(file_path.relative_to(model_dir))
                    }
                    info["files"].append(file_info)
                    
                    # Определяем тип модели по файлам
                    if file_path.suffix in ['.bin', '.safetensors']:
                        if "pytorch" in file_path.name.lower() or "pytorch" in str(file_path.parent):
                            info["type"] = "pytorch"
                        elif "tf_model" in file_path.name:
                            info["type"] = "tensorflow"
                    
                    # Пытаемся найти config файл для дополнительной информации
                    if file_path.name == "config.json":
                        config_info = self.parse_config_file(file_path)
                        info.update(config_info)
            
            # Сортируем файлы по размеру
            info["files"].sort(key=lambda x: x["size_mb"], reverse=True)
            
            return info
            
        except Exception as e:
            logger.warning(f"Ошибка анализа папки {model_dir}: {e}")
            return None
    
    def _get_display_name(self, model_dir_name: str) -> str:
        """
        Преобразует имя папки в читаемое имя модели
        """
        if 'models--' in model_dir_name:
            # Формат: models--author--model-name → author/model-name
            return model_dir_name.replace('models--', '').replace('--', '/')
        else:
            return model_dir_name
    
    def get_directory_size_mb(self, directory: Path) -> float:
        """Вычисление размера папки в MB"""
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    
    def parse_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Парсинг config.json для получения метаданных модели"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            info = {}
            
            # Извлекаем полезную информацию из конфига
            if "architectures" in config:
                info["architecture"] = config["architectures"][0] if config["architectures"] else "unknown"
            
            if "model_type" in config:
                info["model_type"] = config["model_type"]
            
            if "vocab_size" in config:
                info["vocab_size"] = config["vocab_size"]
                
            if "hidden_size" in config:
                info["hidden_size"] = config["hidden_size"]
            
            return info
            
        except Exception as e:
            logger.warning(f"Ошибка парсинга config.json {config_path}: {e}")
            return {}
    
    def get_system_resources(self) -> Dict[str, Any]:
        """
        Получение информации о системных ресурсах
        """
        try:
            import psutil
            import torch
            
            # Информация о CPU и RAM
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            resources = {
                "cpu": {
                    "percent": cpu_percent,
                    "cores": psutil.cpu_count(logical=False),
                    "threads": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_percent": memory.percent
                },
                "gpu": {}
            }
            
            # Информация о GPU если доступно
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_props = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3)
                    }
                    resources["gpu"][f"cuda:{i}"] = gpu_props
            
            return resources
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о ресурсах: {e}")
            return {"error": str(e)}

# Глобальный экземпляр сервиса
model_discovery = ModelDiscoveryService()