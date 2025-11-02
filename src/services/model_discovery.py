# src\services\model_discovery.py
"""
Сервис обнаружения и анализа моделей в кеше
Поддержка GGUF формата для LLM и HF формата для embeddings
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from huggingface_hub import snapshot_download

class ModelDiscoveryService:
    """Обнаружение и анализ моделей в локальном кеше"""
    
    def __init__(self):
        self.models_cache = Path(os.environ.get('HF_HOME', 'storage/models'))
        
    def scan_models_cache(self) -> Dict[str, Any]:
        """
        Сканирование папки с моделями с поддержкой GGUF и HF форматов
        """
        try:
            logger.info(f"🔍 Сканирование кеша моделей: {self.models_cache}")
            
            if not self.models_cache.exists():
                return {"error": "Папка с моделями не существует", "models": []}
            
            models_info = []
            
            # Сканируем ВСЕ элементы в кеше
            for item in self.models_cache.iterdir():
                if item.is_file() and item.suffix.lower() == '.gguf':
                    # Найден GGUF файл - анализируем как LLM модель
                    model_info = self._analyze_gguf_file(item)
                    if model_info and self._is_usable_model(model_info):
                        models_info.append(model_info)
                        logger.info(f"✅ Найден GGUF: {item.name}")
                        
                elif item.is_dir() and self._is_valid_model_directory(item):
                    # Найдена папка в HF формате
                    model_info = self.analyze_model_directory(item)
                    if model_info and self._is_usable_model(model_info):
                        models_info.append(model_info)
                        logger.info(f"✅ Найден HF: {item.name} ({model_info.get('type', 'unknown')})")
            
            # СОРТИРУЕМ модели по типу и качеству
            models_info.sort(key=lambda x: (
                0 if x.get('type') == 'embedding' else 1,  # Сначала embedding модели
                x['name']  # Затем по имени
            ))
            
            result = {
                "cache_path": str(self.models_cache),
                "total_models": len(models_info),
                "models": models_info
            }
            
            logger.success(f"📊 Сканирование завершено: {len(models_info)} моделей")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка сканирования моделей: {e}")
            return {"error": str(e), "models": []}
    
    def _analyze_gguf_file(self, gguf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Анализирует GGUF файл и возвращает информацию о модели
        """
        try:
            file_size_mb = gguf_path.stat().st_size / (1024 * 1024)
            file_name = gguf_path.name
            
            # Определяем тип модели по имени файла
            model_type = "llm"
            lower_name = file_name.lower()
            
            if any(keyword in lower_name for keyword in ['embedding', 'embed', 'encoder']):
                model_type = "embedding"
            
            return {
                "name": file_name,
                "display_name": file_name,  # GGUF файлы имеют понятные имена
                "path": str(gguf_path),
                "size_mb": round(file_size_mb, 2),
                "type": model_type,
                "format": "gguf",
                "is_gguf": True,
                "is_hf": False,
                "files": [{
                    "name": file_name,
                    "size_mb": round(file_size_mb, 2),
                    "relative_path": file_name
                }],
                "is_usable": True
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа GGUF файла {gguf_path}: {e}")
            return None
    
    def _is_valid_model_directory(self, model_dir: Path) -> bool:
        """
        Проверяет является ли папка РЕАЛЬНОЙ рабочей моделью в СТАНДАРТНОМ ФОРМАТЕ HF
        """
        dir_name = model_dir.name
        
        # ВСЁ ПРОСТО: ТОЛЬКО стандартный формат HF!
        if not dir_name.startswith('models--'):
            return False
        
        # ВСЁ ПРОСТО: Проверяем что это реальная модель (есть конфиг)
        has_config = (model_dir / "config.json").exists()
        
        return has_config
    
    def _is_usable_model(self, model_info: Dict) -> bool:
        """
        Проверяет можно ли использовать модель
        """
        # Для GGUF файлов - всегда используемы если файл существует
        if model_info.get('is_gguf', False):
            return Path(model_info['path']).exists()
            
        # Для HF моделей - старая логика
        if model_info.get('size_mb', 0) < 1:  # Слишком маленькая
            return False
            
        if not model_info.get('files'):  # Нет файлов
            return False
            
        # Минимум 1 файл модели и 1 конфиг
        model_files = [f for f in model_info['files'] if any(ext in f['name'] for ext in ['.bin', '.safetensors', '.pt'])]
        config_files = [f for f in model_info['files'] if 'config.json' in f['name']]
        
        return len(model_files) > 0 and len(config_files) > 0
    
    def analyze_model_directory(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """
        ДЕТАЛЬНЫЙ анализ папки с моделью (HF формат)
        """
        try:
            model_name = model_dir.name
            
            # ОПРЕДЕЛЯЕМ ТИП МОДЕЛИ по структуре
            model_type = self._detect_model_type(model_dir)
            
            info = {
                "name": model_name,
                "display_name": self._get_display_name(model_name),
                "path": str(model_dir),
                "size_mb": self.get_directory_size_mb(model_dir),
                "type": model_type,
                "format": "hf",
                "is_gguf": False,
                "is_hf": True,
                "files": [],
                "is_usable": True
            }
            
            # Анализ файлов
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    file_info = {
                        "name": file_path.name,
                        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                        "relative_path": str(file_path.relative_to(model_dir))
                    }
                    info["files"].append(file_info)
            
            # Парсим конфиг для дополнительной информации
            config_info = self.parse_config_file(model_dir)
            info.update(config_info)
            
            return info
            
        except Exception as e:
            logger.warning(f"Ошибка анализа папки {model_dir}: {e}")
            return None

    def _detect_model_type(self, model_dir: Path) -> str:
        """
        Универсальное определение типа модели по стандартам Hugging Face
        Сохраняем ВСЮ старую логику для HF моделей
        """
        # === 1. ПРОВЕРКА НА EMBEDDING МОДЕЛИ ===
        if (model_dir / "config_sentence_transformers.json").exists():
            return "embedding"
        
        if (model_dir / "modules.json").exists():
            return "embedding"
            
        # === 2. ПРОВЕРКА ПО ОСНОВНОМУ CONFIG.JSON ===
        try:
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                embedding_architectures = [
                    "SentenceTransformer", "Transformer", "EmbeddingModel",
                    "XLMRobertaModel", "MPNetModel", "DistilBertModel"
                ]
                
                llm_architectures = [
                    "Qwen2ForCausalLM", "LlamaForCausalLM", "GPT2LMHeadModel",
                    "MistralForCausalLM", "PhiForCausalLM", "BloomForCausalLM"
                ]
                
                architectures = config.get("architectures", [])
                model_type = config.get("model_type", "")
                
                if any(arch in str(architectures) for arch in embedding_architectures):
                    return "embedding"
                    
                if any(arch in str(architectures) for arch in llm_architectures):
                    return "llm"
                
                if any(tipo in model_type for tipo in ["sentence_transformers", "embedding"]):
                    return "embedding"
                elif any(tipo in model_type for tipo in ["text-generation", "causal-lm"]):
                    return "llm"
                    
        except Exception as e:
            logger.warning(f"Ошибка чтения config.json для {model_dir}: {e}")
        
        # === 3. ПРОВЕРКА ПО ФАЙЛАМ ТОКЕНАЙЗЕРА ===
        tokenizer_files = list(model_dir.glob("tokenizer*")) + list(model_dir.glob("*vocab*"))
        if tokenizer_files:
            return "llm"
        
        # === 4. РЕЗЕРВНЫЙ ВАРИАНТ: ПО ИМЕНИ ПАПКИ ===
        dir_name = model_dir.name.lower()
        
        embedding_keywords = ['e5', 'embedding', 'sentence', 'transformers', 'mpnet', 'minilm']
        llm_keywords = ['qwen', 'chat', 'instruct', 'gpt', 'llama', 'mistral', 'phi']
        
        if any(keyword in dir_name for keyword in embedding_keywords):
            return "embedding"
        elif any(keyword in dir_name for keyword in llm_keywords):
            return "llm"
        
        logger.warning(f"Не удалось определить тип модели: {model_dir.name}")
        return "unknown"
    
    def _get_display_name(self, model_dir_name: str) -> str:
        """
        Преобразует имя папки в читаемое имя модели
        """
        if 'models--' in model_dir_name:
            return model_dir_name.replace('models--', '').replace('--', '/')
        else:
            return model_dir_name
    
    def get_directory_size_mb(self, directory: Path) -> float:
        """Вычисление размера папки в MB"""
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return round(total_size / (1024 * 1024), 2)
    
    def parse_config_file(self, model_dir: Path) -> Dict[str, Any]:
        """Парсинг config.json для получения метаданных модели"""
        try:
            config_path = next(model_dir.rglob("config.json"))
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            info = {}
            
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
            logger.warning(f"Ошибка парсинга config.json: {e}")
            return {}

    def analyze_model_cache(self) -> Dict[str, Any]:
        """
        Анализ кеша на наличие битых и неиспользуемых моделей
        """
        try:
            cache_info = self.scan_models_cache()
            broken_models = []
            
            for model in cache_info.get("models", []):
                if not model.get("is_usable", True):
                    broken_models.append(model["name"])
            
            return {
                "total_models": cache_info["total_models"],
                "broken_models": broken_models,
                "usable_models": [m["name"] for m in cache_info["models"] if m.get("is_usable", True)]
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа кеша: {e}")
            return {"error": str(e)}

    def _get_local_model_path(self, model_name: str):
        """Получаем локальный путь к модели"""
        # Сначала проверяем GGUF файлы
        gguf_path = self.models_cache / model_name
        if gguf_path.exists() and gguf_path.is_file() and gguf_path.suffix.lower() == '.gguf':
            return gguf_path
        
        # Затем проверяем HF папки
        возможные_пути = []
        
        if 'models--' in model_name:
            возможные_пути.append(self.models_cache / model_name)
        else:
            cache_name = f"models--{model_name.replace('/', '--')}"
            возможные_пути.append(self.models_cache / cache_name)
            возможные_пути.append(self.models_cache / model_name)
        
        for путь in возможные_пути:
            if путь.exists():
                return путь
        
        return None

    def delete_model(self, model_name: str) -> bool:
        """Удаление модели из кеша"""
        try:
            local_path = self._get_local_model_path(model_name)
            if local_path and local_path.exists():
                import shutil
                if local_path.is_file():
                    local_path.unlink()  # Удаляем GGUF файл
                else:
                    shutil.rmtree(local_path)  # Удаляем HF папку
                logger.info(f"Модель {model_name} удалена из кеша")
                return True
            logger.warning(f"Модель {model_name} не найдена для удаления: {local_path}")
            return False
        except Exception as e:
            logger.error(f"Ошибка удаления модели {model_name}: {e}")
            return False

    def download_model(self, model_id: str) -> bool:
        """Скачивание модели из HuggingFace Hub"""
        try:
            logger.info(f"Скачивание модели: {model_id}")
            
            # Создаем имя папки в стандартном формате
            cache_name = f"models--{model_id.replace('/', '--')}"
            local_dir = self.models_cache / cache_name
            
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Модель {model_id} успешно скачана в {local_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка скачивания модели {model_id}: {e}")
            return False

# Глобальный экземпляр сервиса
model_discovery = ModelDiscoveryService()