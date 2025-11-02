# src/services/batch_processor.py
from typing import List, Tuple
import torch
from loguru import logger

class VolumeBatchProcessor:
    """
    Volume-based батчер для эмбеддингов
    Формирует батчи based на объеме памяти, а не количестве текстов
    """
    
    def __init__(self):
        self.memory_per_char = self._calibrate_memory_usage()
        logger.info(f"🔧 Memory per char: {self.memory_per_char:.2f} bytes")
    
    def _calibrate_memory_usage(self) -> float:
        """
        Калибровка: определяем сколько байт памяти занимает 1 символ текста
        """
        # Эмпирическая константа для multilingual-e5-large-instruct
        # На основе тестов: 1 символ ~ 0.3 байта в GPU памяти при батч обработке
        return 0.3
    
    def estimate_text_volume(self, text: str) -> int:
        """
        Быстрая оценка объема памяти для текста
        Возвращает условные единицы объема
        """
        return max(1, len(text))  # Минимум 1 чтобы избежать деления на 0
    
    def calculate_max_volume(self) -> int:
        """
        Вычисляет максимальный объем батча based на свободной памяти GPU
        """
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