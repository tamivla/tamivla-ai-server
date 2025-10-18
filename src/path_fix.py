"""
Фикс путей для Python при запуске как службы
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

# Автоматически добавляем путь при импорте
add_project_to_path()