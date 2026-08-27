from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.code_conversion import CodeConversionService
from app.core.config import get_settings
from app.services.aider import AiderSessionManager
from app.services.glm import GlmService
from app.services.knowledge import KnowledgeService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.aider_sessions = AiderSessionManager()
    app.state.knowledge_service = KnowledgeService(
        settings.knowledge_base_path,
        settings.knowledge_top_k,
        storage_dir=settings.knowledge_base_dir,
        max_upload_bytes=settings.knowledge_max_upload_bytes,
    )
    app.state.knowledge_service.reload_if_changed(force=True)
    app.state.glm_service = GlmService(settings)
    app.state.code_conversion_service = CodeConversionService(
        settings, app.state.glm_service.client
    )
    await app.state.code_conversion_service.recover_incomplete_jobs()
    try:
        yield
    finally:
        await app.state.aider_sessions.close_all()
        await app.state.code_conversion_service.close()
        await app.state.glm_service.close()


settings = get_settings()
app = FastAPI(
    title="Control Center Aider Agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
