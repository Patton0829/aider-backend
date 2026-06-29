from fastapi import APIRouter

from app.core.config import get_settings


router = APIRouter()


@router.get("/models")
async def models():
    settings = get_settings()
    return {
        "defaultChatModel": "flash",
        "defaultEditModel": settings.aider_model,
        "models": [
            {"id": "flash", "model": settings.qwen_flash_model, "purpose": "chat"},
            {"id": "plus", "model": settings.qwen_plus_model, "purpose": "complex"},
        ],
        "workspaces": sorted(settings.workspace_names),
    }

