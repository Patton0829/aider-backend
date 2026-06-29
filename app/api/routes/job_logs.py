from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.core.security import authorize_websocket
from app.schemas.agent import LogEvent
from app.services.job_logs import JobLogService


router = APIRouter()


@router.websocket("/ws/jobs/{job_id}/logs")
async def job_log_socket(websocket: WebSocket, job_id: int):
    if not await authorize_websocket(websocket):
        return
    await websocket.accept()

    service = JobLogService(get_settings())
    output_dir = service.job_output_dir(job_id)
    await websocket.send_json(
        LogEvent(
            type="connected",
            content="job log websocket connected",
            path=str(output_dir),
        ).model_dump()
    )

    try:
        async for event_type, content in service.follow(job_id):
            path = service.find_log_file(job_id)
            await websocket.send_json(
                LogEvent(
                    type=event_type,
                    content=content,
                    path=str(path) if path else None,
                ).model_dump()
            )
    except WebSocketDisconnect:
        pass

