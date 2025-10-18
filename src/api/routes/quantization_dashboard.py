"""
Веб-интерфейс для управления квантованием моделей
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()

# Настройка шаблонов
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go up to project root
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@router.get("/quantization-dashboard", response_class=HTMLResponse)
async def quantization_dashboard(request: Request):
    """
    Главная страница дашборда управления квантованием
    """
    return templates.TemplateResponse(
        "quantization_dashboard.html",
        {"request": request}
    )