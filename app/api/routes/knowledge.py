from fastapi import APIRouter, Body, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.security import authorize_websocket
from app.schemas.knowledge import KnowledgeEvent, KnowledgeRequest, KnowledgeSource
from app.services.glm import GlmService
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
    service: KnowledgeService = request.app.state.knowledge_service
    return {
        **service.info(),
        "model": get_settings().glm_model,
    }


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
    return {
        **service.info(),
        "model": get_settings().glm_model,
        "uploaded": document,
    }


@router.delete("/api/knowledge/documents/{document_name}")
async def delete_knowledge_document(document_name: str, request: Request):
    service: KnowledgeService = request.app.state.knowledge_service
    try:
        info = service.delete_document(document_name)
    except KnowledgeDocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except KnowledgeValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        **info,
        "model": get_settings().glm_model,
        "deleted": document_name,
    }


@router.websocket("/ws/knowledge")
async def knowledge_socket(websocket: WebSocket):
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()

    settings = get_settings()
    knowledge: KnowledgeService = websocket.app.state.knowledge_service
    glm: GlmService = websocket.app.state.glm_service
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
                    results = knowledge.search(retrieval_query)
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
                            "GLM 正在分析检索结果",
                            request_id=request_id,
                        )
                        context = knowledge.format_context(results)
                        async for delta in glm.stream_answer(
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
                        "当前问题无需检索，GLM 正在直接回答",
                        request_id=request_id,
                    )
                    async for delta in glm.stream_direct_answer(question, history):
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
