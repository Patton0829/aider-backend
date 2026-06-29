from fastapi import WebSocket, status

from app.core.config import get_settings


async def authorize_websocket(websocket: WebSocket) -> bool:
    expected = get_settings().agent_api_token
    if not expected:
        return True

    supplied = websocket.query_params.get("token")
    if not supplied:
        authorization = websocket.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()

    if supplied == expected:
        return True

    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    return False

