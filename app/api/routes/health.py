from fastapi import APIRouter

from app.core.config import get_settings


router = APIRouter()


@router.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "dashscopeConfigured": bool(settings.dashscope_api_key),
        "aiderSourceExists": settings.aider_source_root.is_dir(),
        "jobWorkspaceRoot": str(settings.job_workspace_root),
    }

