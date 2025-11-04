# src/services/batch_processor.py
from typing import List, Tuple
import torch
import time
from loguru import logger

class VolumeBatchProcessor:
    """
    Volume-based батчер для эмбеддингов
    Формирует батчи based на объеме памяти, а не количестве текстов
    """
    
    def __init__(self):
        self.memory_per_char = 0.3  # Временное значение
        self.is_calibrated = False
        logger.info("🔧 Batch processor инициализирован, калибровка отложена")
    
    def _calibrate_memory_usage(self) -> float:
        """
        АВТОКАЛИБРОВКА: точное определение потребления памяти на символ
        Вызывается при ПЕРВОМ использовании батчера
        """
        try:
            from services.model_manager import model_manager
            
            # Проверяем что модель загружена
            model_name = "models--intfloat--multilingual-e5-large-instruct"
            if model_name not in model_manager.loaded_models:
                logger.warning("❌ Модель не загружена для калибровки, используем константу")
                return 0.3  # Fallback
            
            model = model_manager.loaded_models[model_name]['model']
            
            # Тестовые тексты разной длины
            test_texts = [
                "A" * 100,    # Короткий текст
                "A" * 500,    # Средний текст  
                "A" * 1000,   # Длинный текст
                "A" * 2000    # Очень длинный текст
            ]
            
            # Замеряем память ДО обработки
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Обрабатываем тестовые тексты
            start_time = time.time()
            embeddings = model.encode(test_texts)
            processing_time = time.time() - start_time
            
            # Замеряем память ПОСЛЕ обработки
            final_memory = torch.cuda.memory_allocated()
            memory_used = final_memory - initial_memory
            
            # Вычисляем общее количество символов
            total_chars = sum(len(text) for text in test_texts)
            
            # Вычисляем память на символ
            memory_per_char = memory_used / total_chars if total_chars > 0 else 0.3
            
            logger.info(f"🎯 Калибровка: {len(test_texts)} текстов, {total_chars} символов")
            logger.info(f"🎯 Память: {memory_used/1024**2:.2f} MB, Время: {processing_time:.3f}s")
            logger.info(f"🎯 Результат: {memory_per_char:.4f} байт/символ")
            
            # Очищаем память
            del embeddings
            torch.cuda.empty_cache()
            
            return max(0.1, min(1.0, memory_per_char))  # Ограничиваем разумные пределы
            
        except Exception as e:
            logger.error(f"❌ Ошибка калибровки: {e}, используем константу 0.3")
            return 0.3  # Fallback значение
    
    def _ensure_calibrated(self):
        """Убеждаемся что калибровка выполнена"""
        if not self.is_calibrated:
            self.memory_per_char = self._calibrate_memory_usage()
            self.is_calibrated = True
            logger.info(f"🔧 Auto-calibrated memory per char: {self.memory_per_char:.4f} bytes")
    
    def estimate_text_volume(self, text: str) -> int:
        """
        Быстрая оценка объема памяти для текста
        Возвращает условные единицы объема
        """
        return max(1, len(text))
    
    def calculate_max_volume(self) -> int:
        """
        Вычисляет максимальный объем батча based на свободной памяти GPU
        """
        # 🔴 ВЫПОЛНЯЕМ КАЛИБРОВКУ ПРИ ПЕРВОМ ИСПОЛЬЗОВАНИИ
        self._ensure_calibrated()
        
        if not torch.cuda.is_available():
            return 10000  # Fallback для CPU
            
        # Получаем информацию о памяти
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        free_memory = total - allocated
        
        # Используем 70% свободной памяти для безопасности
        safe_memory = int(free_memory * 0.7)
        
        # Конвертируем байты в условные единицы объема
        max_volume = int(safe_memory / self.memory_per_char)
        
        logger.debug(f"🎯 Free: {free_memory/1024**2:.0f}MB -> Max volume: {max_volume}")
        
        return max(1000, max_volume)  # Минимум 1000 единиц
    
    def form_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Формирует батчи based на объеме памяти
        """
        if not texts:
            return []
            
        max_volume = self.calculate_max_volume()
        batches = []
        current_batch = []
        current_volume = 0
        
        for text in texts:
            text_volume = self.estimate_text_volume(text)
            
            # Если текст один слишком большой - обрабатываем отдельно
            if text_volume > max_volume:
                logger.warning(f"📏 Текст слишком большой: {text_volume} > {max_volume}")
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_volume = 0
                batches.append([text])
                continue
            
            # Проверяем влезает ли текст в текущий батч
            if current_volume + text_volume > max_volume and current_batch:
                # Батч заполнен - сохраняем и начинаем новый
                batches.append(current_batch)
                current_batch = [text]
                current_volume = text_volume
            else:
                # Добавляем в текущий батч
                current_batch.append(text)
                current_volume += text_volume
        
        # Добавляем последний батч если он не пустой
        if current_batch:
            batches.append(current_batch)
            
        logger.info(f"📦 Сформировано батчей: {len(batches)} для {len(texts)} текстов")
        
        return batches

# Глобальный экземпляр
batch_processor = VolumeBatchProcessor()