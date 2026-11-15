"""Exercise summary eviction/reconnect/old-history recall in a disposable session."""
import asyncio
import json
import secrets

import httpx

from app.evaluation import connection, run_turn, wait_ready


async def memory_smoke(settings, api_base):
    token, marker = secrets.token_hex(32), "memory-" + secrets.token_hex(4)
    headers = {"X-Conversation-Token": token}
    if settings.agent_api_token:
        headers["Authorization"] = f"Bearer {settings.agent_api_token}"
    identity = None
    async with httpx.AsyncClient(base_url=api_base.rstrip("/") + "/", headers=headers, timeout=30, trust_env=False) as http:
        info = await http.get("api/knowledge/info")
        info.raise_for_status()
        config = info.json().get("memory", {})
        recent = config.get("recentTurns", 0)
        if not config.get("enabled") or not 1 <= recent <= 20 or info.json().get("conversationPersistence") != "mysql":
            raise RuntimeError("Enable MySQL history and memory with at least one recent turn")
        try:
            created = await http.post("api/knowledge/conversations")
            created.raise_for_status()
            identity = created.json()["id"]
            credentials = {"conversationId": identity, "ownerToken": token}
            results = []
            async with connection(settings, api_base) as socket:
                await wait_ready(socket)
                for number in range(recent + 1):
                    question = (f"请记住本次会话的测试标记：{marker}。只回答收到。" if number == 0
                                else f"继续第{number+1}轮，请只回答收到。")
                    results.append(await run_turn(socket, question, f"memory-{number}", credentials))
            # A fresh socket must recover both recent history and the summary.
            async with connection(settings, api_base) as socket:
                await wait_ready(socket)
                await socket.send(json.dumps({"type": "resume", "requestId": "memory-resume", **credentials}))
                restored = False
                async with asyncio.timeout(30):
                    while True:
                        event = json.loads(await socket.recv())
                        if event["type"] == "error":
                            raise RuntimeError("Memory checkpoint restore failed")
                        if event["type"] == "history":
                            restored = len(event["data"]["turns"]) == recent + 1
                        if event["type"] == "ready":
                            break
                recalled = await run_turn(socket, "我第一条问题是什么？请复述原文。", "memory-recall", credentials)
            saved = await http.get(f"api/knowledge/conversations/{identity}")
            saved.raise_for_status()
            metrics = [result.get("metrics", {}) for result in [*results, recalled]]
            checks = {
                "allTurnsSaved": all(r.get("persisted") is True for r in [*results, recalled]),
                "summaryCreated": any(m.get("memorySummaryUpdated") and m.get("memorySummaryItems", 0) > 0 for m in metrics),
                "restoredAfterReconnect": restored,
                "oldQuestionRecalled": marker in recalled.get("answer", ""),
                "ownedDatabaseRecall": recalled.get("metrics", {}).get("memoryRecallMode") == "ordinal"
                    and recalled.get("metrics", {}).get("memoryRecallCount") == 1,
                "inputWithinBudget": all(m.get("estimatedInputTokens", 1) <= m.get("inputTokenBudget", 0) for m in metrics),
                "completeRecordsPreserved": len(saved.json()["turns"]) == recent + 2
                    and all(t["status"] == "completed" for t in saved.json()["turns"]),
            }
            print(json.dumps({"passed": all(checks.values()), "turnCount": recent + 2,
                              "summaryMethods": sorted({m.get("memorySummaryMethod", "unknown") for m in metrics if m.get("memorySummaryUpdated")}),
                              "checks": checks}, ensure_ascii=False, indent=2))
            return 0 if all(checks.values()) else 1
        finally:
            if identity is not None:
                response = await http.delete(f"api/knowledge/conversations/{identity}")
                if response.is_error:
                    print("Temporary memory-test conversation cleanup did not complete; inspect backend logs.")
