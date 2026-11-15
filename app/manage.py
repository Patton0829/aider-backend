"""Server administration through the running API; never opens a second Lite client."""
import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

from app.core.config import PROJECT_ROOT, Settings


async def smoke(settings: Settings, api_base: str, question: str) -> int:
    from app.evaluation import connection, run_turn, wait_ready
    async with connection(settings, api_base) as socket:
        await wait_ready(socket)
        result = await run_turn(socket, question, "deployment-smoke")
    result.pop("answer", None)
    result.pop("sources", None)
    passed = (result["deltaCount"] > 0 and result.get("sourceCount", 0) > 0
              and "totalMs" in result and result.get("validation", {}).get("passed") is True
              and result.get("assessment", {}).get("answerable") is True)
    if settings.knowledge_vector_store == "milvus_lite":
        versions = result.get("indexVersions", [])
        passed = passed and len(versions) == 1 and len(versions[0]) == 64
    print(json.dumps(result | {"passed": passed}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "indexes", "rebuild", "activate", "smoke", "evaluate", "cache-check", "cache-smoke"])
    parser.add_argument("version", nargs="?")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="转辙机锁闭不到位，可能有哪些原因？应按什么顺序检查？")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "fault_diagnosis.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "storage" / "rag" / "evaluations" / "latest.json")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    settings = Settings.from_env()
    if args.command == "evaluate":
        from app.evaluation import evaluate
        return asyncio.run(evaluate(settings, args.api_base, args.dataset, args.output, args.limit))
    if args.command == "smoke":
        try:
            return asyncio.run(smoke(settings, args.api_base, args.question))
        except Exception as error:
            print(f"RAG smoke test did not complete ({type(error).__name__}); check backend logs.", file=sys.stderr)
            return 1
    if args.command == "check":
        failures = []
        if sys.version_info < (3, 12):
            failures.append("Python >= 3.12 is required")
        needs_api_key = settings.knowledge_provider == "qwen" or settings.knowledge_retrieval_mode != "keyword" or settings.knowledge_rerank_enabled
        if needs_api_key and not settings.dashscope_api_key.strip():
            failures.append("DASHSCOPE_API_KEY is missing from .env")
        versions = {}
        try:
            versions["langgraph"] = importlib.metadata.version("langgraph")
        except importlib.metadata.PackageNotFoundError:
            failures.append("Missing package: langgraph; run scripts/setup.sh")
        try:
            versions["redis"] = importlib.metadata.version("redis")
        except importlib.metadata.PackageNotFoundError:
            failures.append("Missing package: redis; run scripts/setup.sh")
        if settings.knowledge_vector_store == "milvus_lite":
            for package in ["pymilvus", "milvus-lite", "faiss-cpu", "pyarrow"]:
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    failures.append(f"Missing package: {package}; run scripts/setup.sh")
        print(json.dumps({
            "python": sys.version.split()[0], "provider": settings.knowledge_provider,
            "model": settings.knowledge_model, "vectorStore": settings.knowledge_vector_store,
            "workflow": "langgraph_single", "evidenceCheck": settings.knowledge_evidence_check,
            "rewriteFollowups": settings.knowledge_rewrite_followups,
            "queryCacheEnabled": settings.knowledge_cache_enabled,
            "queryCacheTtlSeconds": settings.knowledge_cache_ttl_seconds,
            "knowledgeDirectory": str(settings.knowledge_base_dir),
            "milvusPath": str(settings.knowledge_milvus_uri),
            "catalogPath": str(settings.knowledge_index_catalog_path),
            "dependencies": versions, "errors": failures,
        }, ensure_ascii=False, indent=2))
        return 1 if failures else 0

    paths = {"indexes": ("GET", "/api/knowledge/indexes"), "rebuild": ("POST", "/api/knowledge/indexes/rebuild"),
             "cache-check": ("GET", "/api/knowledge/cache"),
             "cache-smoke": ("POST", "/api/knowledge/cache/probe")}
    if args.command == "activate":
        if not args.version or not all(c in "0123456789abcdef" for c in args.version) or len(args.version) != 64:
            parser.error("activate requires the full 64-character index version")
        paths["activate"] = ("POST", f"/api/knowledge/indexes/{args.version}/activate")
    method, path = paths[args.command]
    headers = {"Authorization": f"Bearer {settings.agent_api_token}"} if settings.agent_api_token else {}
    data = None
    if args.command == "cache-smoke":
        data = json.dumps({"type": "ask", "question": args.question}).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(args.api_base.rstrip("/") + path, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=600) as response:
            payload = json.load(response)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if args.command == "cache-check":
                return 0 if payload.get("state") == "ready" else 1
            if args.command == "cache-smoke":
                return 0 if payload.get("passed") else 1
        return 0
    except urllib.error.HTTPError as error:
        print(f"API returned HTTP {error.code}; inspect current index status and backend logs", file=sys.stderr)
    except urllib.error.URLError:
        print("Backend is not reachable. Start scripts/start.sh before running index commands.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
