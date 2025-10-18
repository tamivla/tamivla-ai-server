"""
Web interface for model management
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

router = APIRouter()

# Template configuration - FIXED PATH
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go up to project root
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@router.get("/dashboard", response_class=HTMLResponse)
async def model_dashboard(request: Request):
    """
    Main model dashboard page
    """
    return templates.TemplateResponse(
        "model_dashboard.html",
        {"request": request}
    )

@router.get("/dashboard/models", response_class=HTMLResponse)
async def models_list(request: Request):
    """
    Models list page with cards
    """
    return templates.TemplateResponse(
        "models_list.html",
        {"request": request}
    )