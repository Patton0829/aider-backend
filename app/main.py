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
from app.services.conversation_store import ConversationStore, HistoryError


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
    history_store = None
    llms = {}
    try:
        if settings.knowledge_history_store == "mysql":
            history_store = ConversationStore(settings)
            try:
                await history_store.check()
            except HistoryError:
                logging.getLogger("uvicorn.error").warning("History storage is unavailable; temporary questions remain supported")
        app.state.conversation_store = history_store
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
        # One shared client/semaphore per provider; a browser choice never mutates global settings.
        llms = {"glm": app.state.glm_service}
        if settings.knowledge_provider == "qwen" or settings.dashscope_api_key.strip():
            knowledge_llm = KnowledgeLlmService(settings, provider="qwen")
            llms["qwen"] = knowledge_llm
        knowledge_llm = llms[settings.knowledge_provider]
        app.state.knowledge_llm_service = knowledge_llm
        app.state.knowledge_workflows = {
            provider: KnowledgeWorkflow(
                settings.model_copy(update={"knowledge_provider": provider}),
                app.state.knowledge_service, llm, retriever,
                evidence_llm=llms.get("qwen"),
            ) for provider, llm in llms.items()
        }
        app.state.knowledge_workflow = app.state.knowledge_workflows[settings.knowledge_provider]
        conversion_service = CodeConversionService(settings, app.state.glm_service.client)
        app.state.code_conversion_service = conversion_service
        await conversion_service.recover_incomplete_jobs()
        yield
    finally:
        await app.state.aider_sessions.close_all()
        if conversion_service is not None:
            await conversion_service.close()
        for provider, llm in llms.items():
            if provider != "glm":
                await llm.close()
        if retrieval_models is not None:
            await retrieval_models.close()
        if retriever is not None:
            await retriever.close()
        if history_store is not None:
            await history_store.close()
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
