from fastapi import APIRouter, Body, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.security import authorize_http, authorize_websocket
from app.schemas.knowledge import KnowledgeEvent, KnowledgeRequest, KnowledgeSource
from app.services.knowledge_llm import KnowledgeLlmService
from app.services.knowledge import (
    KnowledgeDocumentConflictError,
    KnowledgeDocumentNotFoundError,
    KnowledgeSearchResult,
    KnowledgeService,
    KnowledgeValidationError,
    is_contextual_follow_up,
    should_retrieve_knowledge,
)


router = APIRouter()


def knowledge_payload(request: Request) -> dict:
    settings = get_settings()
    payload = {
        **request.app.state.knowledge_service.info(),
        "model": settings.knowledge_model,
        "provider": settings.knowledge_provider,
    }
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
):
    event = KnowledgeEvent(
        type=event_type,
        content=content,
        sources=sources or [],
        request_id=request_id,
    )
    await websocket.send_json(event.model_dump(by_alias=True, exclude_none=True))


def source_models(results: list[KnowledgeSearchResult]) -> list[KnowledgeSource]:
    return [KnowledgeSource.model_validate(result.source_dict()) for result in results]


@router.get("/api/knowledge/info")
async def knowledge_info(request: Request):
    return knowledge_payload(request)


@router.get("/api/knowledge/indexes")
async def knowledge_indexes(request: Request):
    authorize_http(request)
    retriever = request.app.state.knowledge_retriever
    return {"activeVersion": retriever.info().get("indexVersion"), "versions": retriever.index_versions()}


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
    llm: KnowledgeLlmService = websocket.app.state.knowledge_llm_service
    history: list[dict[str, str]] = []
    last_answer_mode: str | None = None

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
                    str(error),
                    request_id=request_id,
                )
                continue

            if request.type == "close":
                await send_event(websocket, "closed")
                break
            if request.type == "clear":
                history.clear()
                last_answer_mode = None
                await send_event(websocket, "cleared", "对话上下文已清空")
                await send_event(websocket, "ready")
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
            await send_event(
                websocket,
                "status",
                "正在判断是否需要检索知识库",
                request_id=request_id,
            )
            try:
                contextual_follow_up = bool(history) and is_contextual_follow_up(
                    question
                )
                if contextual_follow_up:
                    answer_mode = last_answer_mode or (
                        "knowledge" if should_retrieve_knowledge(question) else "direct"
                    )
                    recent_user_questions = [
                        message["content"]
                        for message in history[-4:]
                        if message["role"] == "user"
                    ]
                    retrieval_query = " ".join(
                        [*recent_user_questions[-1:], question]
                    )
                else:
                    # A standalone question starts a new diagnostic topic. Keeping
                    # the prior topic here can dominate both retrieval and GLM output.
                    history.clear()
                    answer_mode = (
                        "knowledge" if should_retrieve_knowledge(question) else "direct"
                    )
                    retrieval_query = question

                await send_event(
                    websocket,
                    "route",
                    answer_mode,
                    request_id=request_id,
                )

                answer = ""
                if answer_mode == "knowledge":
                    await send_event(
                        websocket,
                        "status",
                        "正在检索故障诊断知识库",
                        request_id=request_id,
                    )
                    retriever = getattr(websocket.app.state, "knowledge_retriever", None)
                    results = (
                        await retriever.search(retrieval_query)
                        if retriever is not None else knowledge.search(retrieval_query)
                    )
                    if not results:
                        answer = (
                            "当前知识库中没有足够信息。"
                            "请补充设备类型、故障现象和监测数据。"
                        )
                        await send_event(
                            websocket,
                            "delta",
                            answer,
                            request_id=request_id,
                        )
                    else:
                        await send_event(
                            websocket,
                            "sources",
                            sources=source_models(results),
                            request_id=request_id,
                        )
                        await send_event(
                            websocket,
                            "status",
                            f"{llm.model_label} 正在分析检索结果",
                            request_id=request_id,
                        )
                        context = knowledge.format_context(results)
                        async for delta in llm.stream_answer(
                            question, context, history
                        ):
                            answer += delta
                            await send_event(
                                websocket,
                                "delta",
                                delta,
                                request_id=request_id,
                            )
                else:
                    await send_event(
                        websocket,
                        "status",
                        f"当前问题无需检索，{llm.model_label} 正在直接回答",
                        request_id=request_id,
                    )
                    async for delta in llm.stream_direct_answer(question, history):
                        answer += delta
                        await send_event(
                            websocket,
                            "delta",
                            delta,
                            request_id=request_id,
                        )

                history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                )
                history_limit = settings.knowledge_history_turns * 2
                if history_limit:
                    history = history[-history_limit:]
                else:
                    history.clear()
                last_answer_mode = answer_mode

                await send_event(websocket, "done", request_id=request_id)
                await send_event(websocket, "ready", request_id=request_id)
            except Exception as error:
                await send_event(
                    websocket,
                    "error",
                    str(error),
                    request_id=request_id,
                )
                await send_event(websocket, "ready", request_id=request_id)
    except WebSocketDisconnect:
        pass
