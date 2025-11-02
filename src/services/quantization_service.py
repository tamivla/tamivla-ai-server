"""
Сервис автоматического квантования моделей
Умная загрузка с определением доступной VRAM и авто-квантованием
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

class QuantizationService:
    """Сервис для автоматического квантования моделей под доступные ресурсы"""
    
    def __init__(self):
        self.quantized_models_cache = Path(os.environ.get('HF_HOME', 'storage/models')) / "quantized"
        self.quantized_models_cache.mkdir(exist_ok=True)
        
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Получение детальной информации о GPU памяти
        """
        try:
            if not torch.cuda.is_available():
                return {
                    "available": False,
                    "error": "CUDA not available", 
                    "gpus": {}
                }
            
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
                total = props.total_memory / (1024**3)                  # GB
                free = total - allocated
                
                gpu_info[f"cuda:{i}"] = {
                    "name": props.name,
                    "total_gb": round(total, 2),
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "free_gb": round(free, 2),
                    "free_percent": round((free / total) * 100, 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                }
            
            result = {
                "available": True,
                "gpus": gpu_info
            }
            
            if gpu_info:
                result["primary_gpu"] = list(gpu_info.keys())[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о GPU: {e}")
            return {
                "available": False,
                "error": str(e),
                "gpus": {}
            }
    
    def calculate_optimal_quantization(self, model_size_gb: float, target_device: str = "cuda:0") -> Dict[str, Any]:
        """
        Расчет оптимального уровня квантования для модели
        
        Args:
            model_size_gb: Размер модели в GB
            target_device: Целевое GPU устройство
            
        Returns:
            Словарь с рекомендациями по квантованию
        """
        gpu_info = self.get_gpu_memory_info()
        
        if not gpu_info["available"]:
            return {
                "recommended": "cpu",
                "reason": "GPU not available",
                "can_load": False,
                "estimated_vram_usage_gb": model_size_gb
            }
        
        target_gpu = gpu_info["gpus"].get(target_device)
        if not target_gpu:
            return {
                "recommended": "cpu", 
                "reason": "Target GPU not found",
                "can_load": False,
                "estimated_vram_usage_gb": model_size_gb
            }
        
        free_vram = target_gpu["free_gb"]
        total_vram = target_gpu["total_gb"]
        
        # Расчет коэффициентов квантования
        quantization_levels = {
            "fp32": {"bits": 32, "reduction": 1.0, "quality": "original"},
            "fp16": {"bits": 16, "reduction": 0.5, "quality": "excellent"}, 
            "bf16": {"bits": 16, "reduction": 0.5, "quality": "excellent"},
            "8bit": {"bits": 8, "reduction": 0.25, "quality": "very good"},
            "4bit": {"bits": 4, "reduction": 0.125, "quality": "good"},
            "q4": {"bits": 4, "reduction": 0.125, "quality": "good"}
        }
        
        recommendations = []
        
        for level_name, level_info in quantization_levels.items():
            estimated_size = model_size_gb * level_info["reduction"]
            safety_margin = 1.2  # 20% запас для overhead
            required_vram = estimated_size * safety_margin
            
            can_fit = required_vram <= free_vram
            vram_usage_percent = (required_vram / total_vram) * 100
            
            recommendations.append({
                "level": level_name,
                "bits": level_info["bits"],
                "estimated_size_gb": round(estimated_size, 2),
                "required_vram_gb": round(required_vram, 2),
                "can_fit": can_fit,
                "vram_usage_percent": round(vram_usage_percent, 1),
                "quality": level_info["quality"],
                "recommended": can_fit and level_info["bits"] <= 8  # Предпочитаем 8-bit или меньше
            })
        
        # Сортируем по приоритету (сначала те что влезают, потом по качеству)
        recommendations.sort(key=lambda x: (not x["can_fit"], x["bits"]))
        
        # Выбираем лучшую рекомендацию
        best_recommendation = None
        for rec in recommendations:
            if rec["can_fit"]:
                best_recommendation = rec
                break
        
        if not best_recommendation:
            # Если ничего не влезает, предлагаем самое агрессивное квантование
            best_recommendation = recommendations[-1]
            best_recommendation["forced"] = True
            best_recommendation["warning"] = f"Модель не влезает даже с квантованием. Требуется {best_recommendation['required_vram_gb']}GB, доступно {free_vram}GB"
        
        return {
            "model_size_gb": model_size_gb,
            "target_gpu": target_gpu,
            "free_vram_gb": free_vram,
            "total_vram_gb": total_vram,
            "recommendations": recommendations,
            "best_recommendation": best_recommendation,
            "can_load": best_recommendation["can_fit"] if not best_recommendation.get("forced") else False
        }
    
    def get_model_size_estimation(self, model_name: str) -> float:
        """
        Оценка размера модели по её имени и конфигурации
        """
        # Эмпирические оценки размеров популярных моделей
        model_size_estimations = {
            "Qwen2.5-7B": 14.5,
            "Qwen2.5-14B": 28.0,
            "Qwen2-7B": 14.0,
            "Qwen2-1.5B": 3.0,
            "Llama-3-8B": 16.0,
            "Llama-3-70B": 140.0,
            "mistral-7b": 14.0,
            "mixtral-8x7b": 45.0,
            "all-MiniLM-L6-v2": 0.09,
            "all-mpnet-base-v2": 0.42,
            "paraphrase-multilingual-mpnet-base-v2": 2.1,
            "multilingual-e5-large": 2.2
        }
        
        # Ищем подходящую оценку
        for pattern, size in model_size_estimations.items():
            if pattern.lower() in model_name.lower():
                return size
        
        # Если модель не найдена в списке, используем эвристику
        if "7b" in model_name.lower() or "7B" in model_name:
            return 14.0
        elif "13b" in model_name.lower() or "13B" in model_name:
            return 26.0
        elif "70b" in model_name.lower() or "70B" in model_name:
            return 140.0
        else:
            return 2.0  # Дефолтная оценка для неизвестных моделей
    
    def generate_quantization_suggestions(self, model_name: str) -> Dict[str, Any]:
        """
        Генерация предложений по квантованию для конкретной модели
        """
        model_size = self.get_model_size_estimation(model_name)
        quantization_analysis = self.calculate_optimal_quantization(model_size)
        
        return {
            "model_name": model_name,
            "estimated_size_gb": model_size,
            "quantization_analysis": quantization_analysis,
            "suggestions": self._generate_human_readable_suggestions(quantization_analysis)
        }
    
    def _generate_human_readable_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Генерация человеко-читаемых предложений"""
        suggestions = []
        
        model_size = analysis["model_size_gb"]
        free_vram = analysis["free_vram_gb"]
        best_rec = analysis["best_recommendation"]
        
        suggestions.append(f"💾 Размер модели: {model_size} GB")
        suggestions.append(f"🎮 Доступно VRAM: {free_vram} GB")
        
        if analysis["can_load"]:
            suggestions.append(f"✅ Рекомендуется: {best_rec['level']} ({best_rec['bits']}-bit)")
            suggestions.append(f"📊 Качество: {best_rec['quality']}")
            suggestions.append(f"🔮 Займет VRAM: ~{best_rec['estimated_size_gb']} GB")
        else:
            suggestions.append(f"⚠️  Модель не влезает в доступную память")
            suggestions.append(f"💡 Можно попробовать: {best_rec['level']} ({best_rec['bits']}-bit)")
            suggestions.append(f"🔮 Потребуется: ~{best_rec['required_vram_gb']} GB")
            suggestions.append("🚨 Возможны проблемы с производительностью")
        
        # Альтернативные варианты
        alt_options = [rec for rec in analysis["recommendations"] if rec["can_fit"] and rec != best_rec]
        if alt_options:
            suggestions.append("\n🔧 Альтернативные варианты:")
            for opt in alt_options[:2]:  # Показываем только 2 лучших альтернативы
                suggestions.append(f"   • {opt['level']} ({opt['bits']}-bit) - {opt['estimated_size_gb']} GB")
        
        return suggestions
    
    def get_quantized_model_path(self, model_name: str, quantization_level: str) -> Path:
        """
        Получение пути для квантованной версии модели
        """
        safe_name = model_name.replace('/', '--')
        return self.quantized_models_cache / f"{safe_name}--{quantization_level}"
    
    def is_model_quantized(self, model_name: str, quantization_level: str) -> bool:
        """
        Проверка существует ли уже квантованная версия модели
        """
        quantized_path = self.get_quantized_model_path(model_name, quantization_level)
        return quantized_path.exists()

    def quantize_model(self, model_name: str, quantization_level: str) -> Dict[str, Any]:
        """
        Квантование модели с использованием transformers (обход проблемы bitsandbytes в Windows)
        """
        try:
            logger.info(f"🔄 Запуск квантования {model_name} в {quantization_level}")
            
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            logger.info(f"📥 Импорт библиотек выполнен успешно")
            
            # Определяем конфигурацию квантования
            if quantization_level == '4bit':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif quantization_level == '8bit':
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization_level == 'fp16':
                bnb_config = None  # Будем использовать обычную загрузку с fp16
            else:
                return {'error': f'Unsupported quantization level: {quantization_level}'}
            
            logger.info(f"🔧 Конфигурация квантования создана: {quantization_level}")
            
            # Загружаем модель с квантованием
            logger.info(f"📥 Загрузка модели {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if quantization_level == 'fp16' else None
            )
            
            logger.info(f"✅ Модель успешно загружена с квантованием")
            
            # Сохраняем квантованную модель
            quantized_path = self.get_quantized_model_path(model_name, quantization_level)
            logger.info(f"💾 Сохранение квантованной модели в {quantized_path}...")
            model.save_pretrained(quantized_path)
            
            logger.info(f"🎉 Квантование завершено успешно")
            
            return {
                'success': True,
                'quantized_path': str(quantized_path),
                'model_name': model_name,
                'quantization_level': quantization_level,
                'message': f'Модель успешно квантована в {quantization_level}'
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка квантования: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name,
                'quantization_level': quantization_level
            }

# Глобальный экземпляр сервиса
quantization_service = QuantizationService()