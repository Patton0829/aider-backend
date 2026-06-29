import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.security import authorize_websocket
from app.schemas.agent import AgentEvent, AgentRequest
from app.services.aider import AiderSession
from app.services.qwen import QwenService


router = APIRouter()


async def send_event(websocket: WebSocket, event_type: str, content: str = "", **extra):
    event = AgentEvent(type=event_type, content=content, **extra)
    await websocket.send_json(event.model_dump())


@router.websocket("/ws/agent")
async def agent_socket(websocket: WebSocket):
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()

    settings = get_settings()
    manager = websocket.app.state.aider_sessions
    qwen = QwenService(settings) if settings.dashscope_api_key else None
    history: list[dict[str, str]] = []
    aider_session: AiderSession | None = None
    aider_forwarder: asyncio.Task | None = None

    async def forward_aider_output(session: AiderSession):
        while True:
            content = await session.output_queue.get()
            if content is None:
                await send_event(websocket, "closed", "aider process ended")
                return
            if content == "":
                await send_event(websocket, "ready", request_type="edit")
            else:
                await send_event(websocket, "output", content, request_type="edit")

    await send_event(websocket, "connected", "aider agent websocket connected")
    await send_event(websocket, "ready")

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                request = AgentRequest.model_validate(payload)
            except ValidationError as error:
                await send_event(websocket, "error", str(error))
                continue

            if request.type == "close":
                break
            if request.type == "interrupt":
                if aider_session:
                    await aider_session.interrupt()
                await send_event(websocket, "ready")
                continue
            if not request.content.strip():
                await send_event(websocket, "error", "content must not be empty")
                continue

            if request.type == "chat":
                if qwen is None:
                    await send_event(websocket, "error", "DASHSCOPE_API_KEY is not configured")
                    continue
                await send_event(websocket, "status", "正在调用千问模型", request_type="chat")
                history.append({"role": "user", "content": request.content})
                assistant_content = ""
                try:
                    async for delta in qwen.stream(history, request.model):
                        assistant_content += delta
                        await send_event(websocket, "delta", delta, request_type="chat")
                    history.append({"role": "assistant", "content": assistant_content})
                    await send_event(websocket, "ready", request_type="chat")
                except Exception as error:
                    history.pop()
                    await send_event(websocket, "error", str(error), request_type="chat")
                continue

            try:
                if aider_session is None or aider_session.workspace.name != request.workspace:
                    if aider_session:
                        await manager.close(aider_session.session_id)
                    aider_session = await manager.create(request.workspace, request.files)
                    if aider_forwarder:
                        aider_forwarder.cancel()
                    aider_forwarder = asyncio.create_task(
                        forward_aider_output(aider_session)
                    )
                await send_event(
                    websocket,
                    "status",
                    f"正在使用 aider 修改 {request.workspace}",
                    request_type="edit",
                )
                await aider_session.send(request.content)
            except Exception as error:
                await send_event(websocket, "error", str(error), request_type="edit")

    except WebSocketDisconnect:
        pass
    finally:
        if aider_forwarder:
            aider_forwarder.cancel()
        if aider_session:
            await manager.close(aider_session.session_id)

