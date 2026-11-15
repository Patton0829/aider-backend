"""Repeatable WS evaluation against a running backend, without opening Milvus."""
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import websockets

from app.core.config import Settings


def connection(settings: Settings, api_base: str):
    headers = {"Authorization": f"Bearer {settings.agent_api_token}"} if settings.agent_api_token else {}
    base = api_base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    return websockets.connect(base.rstrip("/") + "/ws/knowledge", proxy=None,
                              additional_headers=headers, open_timeout=10)


async def wait_ready(socket):
    async with asyncio.timeout(15):
        while json.loads(await socket.recv())["type"] != "ready":
            pass


async def run_turn(socket, question: str, request_id: str) -> dict:
    result = {"question": question, "deltaCount": 0, "sources": [], "answer": ""}
    started = time.perf_counter()
    await socket.send(json.dumps({"type": "ask", "question": question, "requestId": request_id}))
    async with asyncio.timeout(180):
        while True:
            event = json.loads(await socket.recv())
            if event.get("requestId") != request_id:
                continue
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            kind = event["type"]
            if kind == "route":
                result["route"] = event["content"]
            elif kind == "sources":
                result["retrievalReadyMs"] = elapsed
                result["sources"] = event["sources"]
                result["sourceCount"] = len(event["sources"])
                result["indexVersions"] = sorted({s.get("kbVersion", "") for s in event["sources"]})
            elif kind == "delta" and event.get("content"):
                result.setdefault("firstContentMs", elapsed)
                result["deltaCount"] += 1
                result["answer"] += event["content"]
            elif kind in ("assessment", "validation", "metrics"):
                result[kind] = event["data"]
            elif kind == "done":
                result["totalMs"] = elapsed
            elif kind == "error":
                raise RuntimeError("Backend returned a retrieval/model error; inspect server logs")
            elif kind == "ready":
                break
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 2)


def score_turn(actual: dict, expected: dict) -> dict:
    checks = {"completed": bool(actual.get("deltaCount")) and "totalMs" in actual}
    if "route" in expected:
        checks["route"] = actual.get("route") == expected["route"]
    if "answerable" in expected:
        assessment = actual.get("assessment", {})
        checks["answerable"] = assessment.get("answerable") == expected["answerable"] and assessment.get("reason") != "check_unavailable"
    if expected.get("answerable"):
        checks["citations"] = actual.get("validation", {}).get("passed") is True
    gold = set(expected.get("entryIds", []))
    if gold:
        found = {s.get("entryId") for s in actual.get("sources", [])}
        actual["recallAt5"] = len(gold & found) / len(gold)
        checks["retrieval"] = gold.issubset(found)
    return {**actual, "checks": checks, "passed": all(checks.values())}


def summarize(rows: list[dict]) -> dict:
    def latency(group):
        report = {"count": len(group)}
        for key in ("retrievalMs", "assessmentMs", "conversationMs", "firstContentMs", "totalMs"):
            values = [r.get("metrics", {}).get(key) if key.endswith("Ms") and key not in ("firstContentMs", "totalMs") else r.get(key) for r in group]
            values = [v for v in values if isinstance(v, (float, int))]
            report[key] = {"mean": round(sum(values) / len(values), 2) if values else None,
                           "p50": percentile(values, .5), "p95": percentile(values, .95)}
        return report
    valid = [r for r in rows if "error" not in r]
    recalls = [r["recallAt5"] for r in rows if "recallAt5" in r]
    generated = [r for r in valid if r.get("assessment", {}).get("answerable") is True]
    cached = [r["metrics"] for r in valid if r.get("metrics", {}).get("queryCacheEnabled")]
    reranked = [m for m in cached if m.get("rerankCacheEligible")]
    return {
        "turnCount": len(rows), "passed": sum(r.get("passed", False) for r in rows),
        "errors": sum("error" in r for r in rows),
        "routeAccuracy": _check_rate(rows, "route"),
        "answerabilityAccuracy": _check_rate(rows, "answerable"),
        "meanRecallAt5": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "referenceIdentityPassRate": round(sum(r.get("validation", {}).get("passed", False) for r in generated) / len(generated), 4) if generated else None,
        "cache": {"enabledTurns": len(cached), "coalescedTurns": sum(m.get("cacheCoalesced", False) for m in cached),
                  "recallHitRate": round(sum(m.get("recallCacheHit", False) for m in cached) / len(cached), 4) if cached else None,
                  "rerankHitRate": round(sum(m.get("rerankCacheHit", False) for m in reranked) / len(reranked), 4) if reranked else None,
                  "degradedTurns": sum(m.get("queryCacheState") == "degraded" for m in cached)},
        "all": latency(valid), "standaloneKnowledge": latency([r for r in valid if r.get("route") == "knowledge" and not r.get("metrics", {}).get("followUp")]),
        "followupKnowledge": latency([r for r in valid if r.get("route") == "knowledge" and r.get("metrics", {}).get("followUp")]),
        "limits": "Reference identity and labelled retrieval/route checks only; claim entailment and diagnostic correctness require human review. Percentiles use nearest rank.",
    }


def _check_rate(rows: list[dict], key: str):
    checks = [r["checks"][key] for r in rows if key in r.get("checks", {})]
    return round(sum(checks) / len(checks), 4) if checks else None


def write_report(dataset: Path, output: Path, rows: list[dict], completed: bool):
    summary = summarize(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "createdAt": datetime.now(timezone.utc).isoformat(), "completed": completed,
        "dataset": dataset.name, "datasetHash": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "summary": summary, "turns": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return summary


async def evaluate(settings: Settings, api_base: str, dataset: Path, output: Path, limit: int | None = None):
    cases = json.loads(dataset.read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases[:limit] if limit else cases:
        remaining = list(enumerate(case["turns"]))
        try:
            async with connection(settings, api_base) as socket:
                await wait_ready(socket)
                while remaining:
                    number, expected = remaining[0]
                    actual = await run_turn(socket, expected["question"], f"eval-{case['id']}-{number}")
                    row = score_turn(actual, expected)
                    rows.append({"caseId": case["id"], "turn": number + 1, "category": case["category"], **row})
                    write_report(dataset, output, rows, False)
                    print(f"{case['id']} turn {number + 1}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
                    remaining.pop(0)
        except Exception as error:
            for number, expected in remaining:
                row = score_turn({}, expected)
                rows.append({**row, "caseId": case["id"], "turn": number + 1, "question": expected["question"], "error": type(error).__name__, "passed": False})
            write_report(dataset, output, rows, False)
            print(f"{case['id']}: ERROR ({type(error).__name__})", flush=True)
    summary = write_report(dataset, output, rows, True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    return 0 if rows and all(r["passed"] for r in rows) else 1
