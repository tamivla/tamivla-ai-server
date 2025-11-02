"""
Фикс путей для Python при запуске как службы + библиотеки llama.cpp
"""

import sys
import os
from pathlib import Path

def add_project_to_path():
    """Добавляет корневую папку проекта в Python path"""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Добавлен путь в Python path: {project_root}")

def add_library_path():
    """Добавляет путь к библиотекам llama.cpp из LM Studio в PATH"""
    project_root = Path(__file__).parent.parent
    lib_path = project_root / "lib"
    cuda_path = lib_path / "cuda"
    
    if lib_path.exists():
        # Добавляем в системный PATH
        path_str = str(lib_path) + ";" + str(cuda_path)
        if path_str not in os.environ["PATH"]:
            os.environ["PATH"] = path_str + ";" + os.environ["PATH"]
            print(f"✅ Добавлены пути к библиотекам в PATH: {lib_path}")
        else:
            print(f"✅ Пути к библиотекам уже в PATH: {lib_path}")
        return True
    else:
        print(f"⚠️ Папка с библиотеками не найдена: {lib_path}")
        return False

# Автоматически добавляем пути при импорте
add_project_to_path()
_LIBRARIES_LOADED = add_library_path()

# Экспортируем статус для использования в других модулях
LIBRARIES_AVAILABLE = _LIBRARIES_LOADED