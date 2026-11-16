from contextlib import aclosing
import logging

from fastapi import APIRouter, Body, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.security import authorize_http, authorize_websocket
from app.schemas.knowledge import KnowledgeEvent, KnowledgeRequest, KnowledgeSource
from app.services.knowledge_workflow import KnowledgeWorkflow
from app.services.conversation_session import ConversationSession
from app.services.context_budget import ContextBudgetExceeded
from app.services.conversation_store import HistoryError
from app.services.knowledge import (
    KnowledgeDocumentConflictError,
    KnowledgeDocumentNotFoundError,
    KnowledgeService,
    KnowledgeValidationError,
)


router = APIRouter()


def knowledge_payload(request: Request) -> dict:
    settings = get_settings()
    payload = {
        **request.app.state.knowledge_service.info(),
        "model": settings.knowledge_model,
        "provider": settings.knowledge_provider,
        "generationModels": [
            {"provider": "qwen", "model": settings.knowledge_qwen_model,
             "label": f"云端千问 · {settings.knowledge_qwen_model}",
             "enabled": "qwen" in getattr(request.app.state, "knowledge_workflows", {})},
            {"provider": "glm", "model": settings.glm_model, "label": "GLM",
             "enabled": "glm" in getattr(request.app.state, "knowledge_workflows", {})},
        ],
        "workflow": "langgraph_single",
        "evidenceCheck": settings.knowledge_evidence_check,
        "conversationPersistence": "connection",
        "memory": {"enabled": settings.knowledge_memory_enabled,
                   "recentTurns": settings.knowledge_history_turns,
                   "contextWindowTokens": settings.knowledge_context_window_tokens,
                   "maxInputTokens": settings.knowledge_max_input_tokens,
                   "tokenEstimator": "utf8_bytes_v1",
                   "historyRecall": "connection_only"},
    }
    history_store = getattr(request.app.state, "conversation_store", None)
    if history_store is not None:
        payload["conversationPersistence"] = "mysql"
        payload["historyStore"] = history_store.info()
        payload["memory"]["historyRecall"] = "owned_mysql_lexical"
    retriever = getattr(request.app.state, "knowledge_retriever", None)
    if retriever is not None:
        payload.update(retriever.info())
        payload["ready"] = payload["ready"] and payload["retrievalReady"]
        payload["loadError"] = payload["loadError"] or payload["retrievalError"]
        if payload["chunkCount"] and not payload["retrievalReady"] and not payload["loadError"]:
            payload["loadError"] = "向量索引待更新，下一次检索时自动构建"
    return payload


async def refresh_retrieval(request: Request):
    retriever = getattr(request.app.state, "knowledge_retriever", None)
    if retriever is not None:
        try:
            await retriever.prepare_index()
        except Exception:
            # Document writes succeeded; report index failure explicitly in the payload.
            # A later retrieval can retry the missing vectors without re-embedding cached ones.
            pass


async def send_event(
    websocket: WebSocket,
    event_type: str,
    content: str = "",
    sources: list[KnowledgeSource] | None = None,
    request_id: str | None = None,
    data: dict | None = None,
    conversation_id: str | None = None,
):
    event = KnowledgeEvent(
        type=event_type,
        content=content,
        sources=sources or [],
        request_id=request_id,
        data=data,
        conversation_id=conversation_id,
    )
    await websocket.send_json(event.model_dump(by_alias=True, exclude_none=True))


@router.get("/api/knowledge/info")
async def knowledge_info(request: Request):
    return knowledge_payload(request)


@router.get("/api/knowledge/indexes")
async def knowledge_indexes(request: Request):
    authorize_http(request)
    retriever = request.app.state.knowledge_retriever
    return {"activeVersion": retriever.info().get("indexVersion"), "versions": retriever.index_versions()}


@router.get("/api/knowledge/cache")
async def knowledge_cache_status(request: Request):
    authorize_http(request)
    return await request.app.state.knowledge_retriever.query_cache.ping()


@router.post("/api/knowledge/cache/probe")
async def knowledge_cache_probe(request: Request, body: KnowledgeRequest):
    authorize_http(request)
    if body.type != "ask" or not body.question.strip():
        raise HTTPException(status_code=400, detail="提供非空的检索问题")
    retriever = request.app.state.knowledge_retriever
    # These are ordinary retrievals; never evict live keys just to make a 'cold' benchmark.
    try:
        first = await retriever.search_with_trace(body.question)
        second = await retriever.search_with_trace(body.question)
    except Exception as error:
        raise HTTPException(status_code=503, detail="检索验证失败，请检查模型、索引状态和后端日志") from error
    identity = lambda result: (result.chunk.chunk_id, result.chunk.content_hash, result.score, result.kb_version)
    stable = [identity(r) for r in first.results] == [identity(r) for r in second.results]
    ready = retriever.query_cache.info()["state"] == "ready"
    passed = bool(second.results) and stable and ready and second.metrics["recallCacheHit"]
    if retriever.settings.knowledge_rerank_enabled:
        passed = passed and second.metrics["rerankCacheHit"]
    return {"passed": passed, "sameResults": stable, "cache": retriever.query_cache.info(),
            "first": first.metrics, "second": second.metrics,
            "sourceCount": len(second.results),
            "indexVersions": sorted({r.kb_version for r in second.results if r.kb_version})}


@router.post("/api/knowledge/indexes/rebuild")
async def rebuild_knowledge_index(request: Request):
    authorize_http(request)
    try:
        await request.app.state.knowledge_retriever.rebuild_index()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="索引构建失败，原发布版本保持不变") from error
    return knowledge_payload(request)


@router.post("/api/knowledge/indexes/{version}/activate")
async def activate_knowledge_index(version: str, request: Request):
    authorize_http(request)
    try:
        await request.app.state.knowledge_retriever.activate_index(version)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return knowledge_payload(request)


@router.get("/api/knowledge/indexes/{version}/chunks/{chunk_id}")
async def knowledge_index_chunk(version: str, chunk_id: str, request: Request):
    authorize_http(request)
    publisher = request.app.state.knowledge_retriever.publisher
    chunk = publisher.get_chunk(version, chunk_id) if publisher else None
    if chunk is None:
        raise HTTPException(status_code=404, detail="指定版本中的片段不存在")
    return chunk


@router.post("/api/knowledge/documents", status_code=201)
async def upload_knowledge_document(
    request: Request,
    filename: str = Query(min_length=1, max_length=255),
    content: bytes = Body(..., media_type="application/octet-stream"),
):
    service: KnowledgeService = request.app.state.knowledge_service
    try:
        document = service.upload_document(filename, content)
    except KnowledgeDocumentConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except KnowledgeValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    await refresh_retrieval(request)
    return {
        **knowledge_payload(request),
        "uploaded": document,
    }


@router.delete("/api/knowledge/documents/{document_name}")
async def delete_knowledge_document(document_name: str, request: Request):
    service: KnowledgeService = request.app.state.knowledge_service
    try:
        service.delete_document(document_name)
    except KnowledgeDocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except KnowledgeValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    await refresh_retrieval(request)
    return {
        **knowledge_payload(request),
        "deleted": document_name,
    }


@router.websocket("/ws/knowledge")
async def knowledge_socket(websocket: WebSocket):
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()

    settings = get_settings()
    knowledge: KnowledgeService = websocket.app.state.knowledge_service
    default_workflow = getattr(websocket.app.state, "knowledge_workflow", None) or KnowledgeWorkflow(
        settings, knowledge, websocket.app.state.knowledge_llm_service,
        getattr(websocket.app.state, "knowledge_retriever", None),
    )
    session = ConversationSession(getattr(websocket.app.state, "conversation_store", None))

    await send_event(websocket, "connected", "故障诊断知识问答服务已连接")
    await send_event(websocket, "ready")

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                request = KnowledgeRequest.model_validate(payload)
            except ValidationError as error:
                request_id = (
                    payload.get("requestId") if isinstance(payload, dict) else None
                )
                await send_event(
                    websocket,
                    "error",
                    "请求格式不正确，请检查问题长度、请求类型和 requestId。",
                    request_id=request_id,
                )
                await send_event(websocket, "ready", request_id=request_id)
                continue

            if request.type == "close":
                await send_event(websocket, "closed")
                break
            if request.type in ("clear", "resume"):
                try:
                    if request.type == "resume":
                        result = await session.resume(request)
                        await send_event(websocket, "history", data=result, request_id=request.request_id, conversation_id=session.identity)
                    else:
                        await session.clear(request)
                        await send_event(websocket, "cleared", "对话上下文已清空", request_id=request.request_id, conversation_id=session.identity)
                except HistoryError as error:
                    await send_event(websocket, "error", str(error), request_id=request.request_id,
                                     data={"code": type(error).__name__})
                await send_event(websocket, "ready", request_id=request.request_id)
                continue

            question = request.question.strip()
            if not question:
                await send_event(
                    websocket,
                    "error",
                    "question must not be empty",
                    request_id=request.request_id,
                )
                continue

            request_id = request.request_id
            provider = request.provider or settings.knowledge_provider
            workflows = getattr(websocket.app.state, "knowledge_workflows", {})
            workflow = workflows.get(provider)
            if workflow is None and provider == settings.knowledge_provider:
                workflow = default_workflow
            if workflow is None:
                await send_event(websocket, "error", "所选生成模型未配置，请联系管理员检查模型服务。",
                                 request_id=request_id)
                await send_event(websocket, "ready", request_id=request_id)
                continue
            await send_event(
                websocket,
                "status",
                "正在判断是否需要检索知识库",
                request_id=request_id,
            )
            try:
                async with aclosing(session.stream(workflow, request)) as events:
                    async for event in events:
                        sources = [KnowledgeSource.model_validate(item) for item in event.get("sources", [])]
                        await send_event(
                            websocket, event["type"], event.get("content", ""),
                            sources=sources, request_id=request_id, data=event.get("data"),
                            conversation_id=session.identity,
                        )

                await send_event(websocket, "done", request_id=request_id,
                    conversation_id=session.identity, data={"persisted": session.identity is not None})
                await send_event(websocket, "ready", request_id=request_id)
            except WebSocketDisconnect:
                raise
            except HistoryError as error:
                await send_event(websocket, "error", str(error), request_id=request_id,
                                 data={"code": type(error).__name__}, conversation_id=session.identity)
                await send_event(websocket, "ready", request_id=request_id)
            except ContextBudgetExceeded as error:
                await send_event(websocket, "error", str(error), request_id=request_id,
                                 data={"code": "ContextBudgetExceeded"}, conversation_id=session.identity)
                await send_event(websocket, "ready", request_id=request_id)
            except Exception as error:
                logging.getLogger("uvicorn.error").warning("RAG request failed (%s)", type(error).__name__)
                await send_event(
                    websocket,
                    "error",
                    "问答处理失败，请检查后端日志、模型连接或额度后重试。",
                    request_id=request_id,
                )
                await send_event(websocket, "ready", request_id=request_id)
    except WebSocketDisconnect:
        pass
