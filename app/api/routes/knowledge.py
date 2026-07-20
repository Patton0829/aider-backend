from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.security import authorize_websocket
from app.schemas.knowledge import KnowledgeEvent, KnowledgeRequest, KnowledgeSource
from app.services.glm import GlmService
from app.services.knowledge import KnowledgeSearchResult, KnowledgeService


router = APIRouter()


async def send_event(
    websocket: WebSocket,
    event_type: str,
    content: str = "",
    sources: list[KnowledgeSource] | None = None,
):
    event = KnowledgeEvent(type=event_type, content=content, sources=sources or [])
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


@router.websocket("/ws/knowledge")
async def knowledge_socket(websocket: WebSocket):
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()

    settings = get_settings()
    knowledge: KnowledgeService = websocket.app.state.knowledge_service
    glm: GlmService = websocket.app.state.glm_service
    history: list[dict[str, str]] = []

    await send_event(websocket, "connected", "故障诊断知识问答服务已连接")
    await send_event(websocket, "ready")

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                request = KnowledgeRequest.model_validate(payload)
            except ValidationError as error:
                await send_event(websocket, "error", str(error))
                continue

            if request.type == "close":
                await send_event(websocket, "closed")
                break
            if request.type == "clear":
                history.clear()
                await send_event(websocket, "cleared", "对话上下文已清空")
                await send_event(websocket, "ready")
                continue

            question = request.question.strip()
            if not question:
                await send_event(websocket, "error", "question must not be empty")
                continue

            await send_event(websocket, "status", "正在检索故障诊断知识库")
            try:
                recent_user_questions = [
                    message["content"]
                    for message in history[-4:]
                    if message["role"] == "user"
                ]
                retrieval_query = " ".join([*recent_user_questions[-1:], question])
                results = knowledge.search(retrieval_query)
                if not results:
                    content = "当前知识库中没有足够信息。请补充设备类型、故障现象和监测数据。"
                    await send_event(websocket, "delta", content)
                    await send_event(websocket, "done")
                    await send_event(websocket, "ready")
                    continue

                await send_event(websocket, "sources", sources=source_models(results))
                await send_event(websocket, "status", "GLM 正在分析检索结果")

                answer = ""
                context = knowledge.format_context(results)
                async for delta in glm.stream_answer(question, context, history):
                    answer += delta
                    await send_event(websocket, "delta", delta)

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

                await send_event(websocket, "done")
                await send_event(websocket, "ready")
            except Exception as error:
                await send_event(websocket, "error", str(error))
                await send_event(websocket, "ready")
    except WebSocketDisconnect:
        pass
