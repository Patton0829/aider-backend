from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.code_conversion import CodeConversionService
from app.core.config import get_settings
from app.services.aider import AiderSessionManager
from app.services.glm import GlmService
from app.services.knowledge import KnowledgeService
from app.services.knowledge_llm import KnowledgeLlmService
from app.services.rag_models import DashScopeRetrievalModels
from app.services.rag_pipeline import RagPipeline
from app.services.knowledge_workflow import KnowledgeWorkflow


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
    knowledge_llm = app.state.glm_service
    conversion_service = None
    retrieval_models = None
    retriever = None
    try:
        if settings.knowledge_retrieval_mode != "keyword" or settings.knowledge_rerank_enabled:
            retrieval_models = DashScopeRetrievalModels(settings)
        retriever = RagPipeline(
            app.state.knowledge_service, settings,
            embedder=retrieval_models if settings.knowledge_retrieval_mode != "keyword" else None,
            reranker=retrieval_models if settings.knowledge_rerank_enabled else None,
        )
        app.state.knowledge_retriever = retriever
        try:
            await retriever.prepare_index()
        except Exception as error:
            logging.getLogger("uvicorn.error").warning(
                "RAG index initialization failed (%s); retrieval status is unavailable",
                type(error).__name__,
            )
        if settings.knowledge_provider == "qwen":
            knowledge_llm = KnowledgeLlmService(settings)
        app.state.knowledge_llm_service = knowledge_llm
        app.state.knowledge_workflow = KnowledgeWorkflow(
            settings, app.state.knowledge_service, knowledge_llm, retriever,
        )
        conversion_service = CodeConversionService(settings, app.state.glm_service.client)
        app.state.code_conversion_service = conversion_service
        await conversion_service.recover_incomplete_jobs()
        yield
    finally:
        await app.state.aider_sessions.close_all()
        if conversion_service is not None:
            await conversion_service.close()
        if knowledge_llm is not app.state.glm_service:
            await knowledge_llm.close()
        if retrieval_models is not None:
            await retrieval_models.close()
        if retriever is not None:
            await retriever.close()
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
