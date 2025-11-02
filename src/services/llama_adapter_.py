"""
Адаптер для работы с llama.cpp DLL из LM Studio
Временная заглушка без загрузки DLL
"""
from typing import Dict, Any

class LlamaAdapter:
    def __init__(self, model_path: str, n_gpu_layers: int = -1, **kwargs):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        
        # НЕ пытаемся загружать DLL - просто заглушка
        print(f"✅ LlamaAdapter (заглушка) с GPU слоев: {n_gpu_layers}")
        print(f"📁 Модель: {model_path}")
        print("⚠️ РЕЖИМ ЗАГЛУШКИ - работаем без DLL")
        
    def create_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Создает completion - заглушка для тестирования"""
        response = {
            "choices": [
                {
                    "text": f"Ответ на: '{prompt}'\n(Модель: {self.model_path}, GPU слоев: {self.n_gpu_layers})\n⚠️ РЕЖИМ ЗАГЛУШКИ - DLL не загружена",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 15,
                "total_tokens": len(prompt.split()) + 15
            }
        }
        return response

# Создаем алиас для совместимости
Llama = LlamaAdapter