"""Exercise persistence/reconnect/idempotency using an owned disposable conversation."""
import asyncio
import json
import secrets

import httpx

from app.evaluation import connection, run_turn, wait_ready


async def history_smoke(settings, api_base, question):
    token = secrets.token_hex(32)
    headers = {"X-Conversation-Token": token}
    if settings.agent_api_token:
        headers["Authorization"] = f"Bearer {settings.agent_api_token}"
    identity = None
    async with httpx.AsyncClient(base_url=api_base.rstrip("/") + "/", headers=headers, timeout=20, trust_env=False) as http:
        try:
            created = await http.post("api/knowledge/conversations")
            created.raise_for_status()
            identity = created.json()["id"]
            credentials = {"conversationId": identity, "ownerToken": token}
            async with connection(settings, api_base) as socket:
                await wait_ready(socket)
                first = await run_turn(socket, question, "history-check-first", credentials)
            async with connection(settings, api_base) as socket:
                await wait_ready(socket)
                await socket.send(json.dumps({"type": "resume", "requestId": "history-resume", **credentials}))
                restored = None
                async with asyncio.timeout(20):
                    while True:
                        event = json.loads(await socket.recv())
                        if event["type"] == "error":
                            raise RuntimeError("History restore failed")
                        if event["type"] == "history":
                            restored = event["data"]
                        if event["type"] == "ready":
                            break
                replay = await run_turn(socket, question, "history-check-first", credentials)
                follow = await run_turn(socket, "那对应的电流特征是什么？", "history-check-follow", credentials)
            saved = await http.get(f"api/knowledge/conversations/{identity}")
            saved.raise_for_status()
            denied = await http.get(f"api/knowledge/conversations/{identity}", headers={"X-Conversation-Token": secrets.token_hex(32)})
            checks = {
                "firstSaved": first.get("persisted") is True,
                "restoredAfterReconnect": bool(restored and restored["turns"] and restored["turns"][0]["status"] == "completed"),
                "requestReplayed": replay.get("metrics", {}).get("replayed") is True and replay["answer"] == first["answer"],
                "followupRetainsTopic": follow.get("metrics", {}).get("followUp") is True and follow.get("persisted") is True,
                "twoCompletedTurns": len(saved.json()["turns"]) == 2 and all(t["status"] == "completed" for t in saved.json()["turns"]),
                "otherOwnerDenied": denied.status_code == 404,
            }
            print(json.dumps({"passed": all(checks.values()), "checks": checks}, ensure_ascii=False, indent=2))
            return 0 if all(checks.values()) else 1
        finally:
            if identity is not None:
                # Deletes only the random conversation created by this invocation.
                response = await http.delete(f"api/knowledge/conversations/{identity}")
                if response.is_error:
                    print("Temporary test conversation cleanup did not complete; inspect backend logs.")
