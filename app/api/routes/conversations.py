import re

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.security import authorize_http
from app.services.conversation_store import HistoryError, owner_hash


router = APIRouter()


def owner(request):
    authorize_http(request)
    token = request.headers.get("x-conversation-token", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
        raise HTTPException(status_code=401, detail="缺少有效的历史会话访问凭据。")
    return owner_hash(token)


def store(request):
    result = getattr(request.app.state, "conversation_store", None)
    if result is None:
        raise HTTPException(status_code=503, detail="当前服务尚未启用 MySQL 历史会话。")
    return result


async def call(operation):
    try:
        return await operation
    except HistoryError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None


@router.get("/api/knowledge/history/status")
async def history_status(request: Request):
    authorize_http(request)
    result = getattr(request.app.state, "conversation_store", None)
    if result is None:
        return {"backend": "off", "state": "disabled"}
    return await call(result.check())


@router.post("/api/knowledge/conversations", status_code=201)
async def create_conversation(request: Request):
    identity = owner(request)
    return await call(store(request).create(identity))


@router.get("/api/knowledge/conversations")
async def list_conversations(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    identity = owner(request)
    return await call(store(request).list(identity, limit, offset))


@router.get("/api/knowledge/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request, limit: int = Query(50, ge=1, le=100), before: int | None = Query(None, ge=1)):
    identity = owner(request)
    result = await call(store(request).restore(identity, conversation_id, limit, before))
    result.pop("state", None)
    return result


@router.delete("/api/knowledge/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request):
    identity = owner(request)
    await call(store(request).delete(identity, conversation_id))
    return {"deleted": conversation_id}
