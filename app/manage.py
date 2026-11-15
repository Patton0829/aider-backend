"""Server administration through the running API; never opens a second Lite client."""
import argparse
import asyncio
import importlib.metadata
import json
import sys
import time
import urllib.error
import urllib.request

from app.core.config import Settings


async def smoke(settings: Settings, api_base: str, question: str) -> int:
    import websockets
    headers = {"Authorization": f"Bearer {settings.agent_api_token}"} if settings.agent_api_token else {}
    websocket_base = api_base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    result = {"question": question, "deltaCount": 0}
    async with websockets.connect(
        websocket_base.rstrip("/") + "/ws/knowledge", proxy=None,
        additional_headers=headers, open_timeout=10,
    ) as socket:
        async with asyncio.timeout(120):
            while json.loads(await socket.recv())["type"] != "ready":
                pass
            started = time.perf_counter()
            await socket.send(json.dumps({"type": "ask", "question": question, "requestId": "deployment-smoke"}))
            while True:
                event = json.loads(await socket.recv())
                if event.get("requestId") != "deployment-smoke":
                    continue
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                if event["type"] == "sources":
                    result["retrievalMs"] = elapsed
                    result["sourceCount"] = len(event["sources"])
                    result["indexVersions"] = sorted({s.get("kbVersion", "") for s in event["sources"]})
                elif event["type"] == "delta" and event.get("content"):
                    if not result["deltaCount"]:
                        result["firstContentMs"] = elapsed
                    result["deltaCount"] += 1
                elif event["type"] == "done":
                    result["totalMs"] = elapsed
                elif event["type"] == "error":
                    print("RAG smoke test returned a retrieval/model error; check backend logs.", file=sys.stderr)
                    return 1
                elif event["type"] == "ready":
                    break
    passed = result["deltaCount"] > 0 and result.get("sourceCount", 0) > 0 and "totalMs" in result
    if settings.knowledge_vector_store == "milvus_lite":
        versions = result.get("indexVersions", [])
        passed = passed and len(versions) == 1 and len(versions[0]) == 64
    print(json.dumps(result | {"passed": passed}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "indexes", "rebuild", "activate", "smoke"])
    parser.add_argument("version", nargs="?")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="转辙机锁闭不到位，可能有哪些原因？应按什么顺序检查？")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == "smoke":
        try:
            return asyncio.run(smoke(settings, args.api_base, args.question))
        except Exception as error:
            print(f"RAG smoke test did not complete ({type(error).__name__}); check backend logs.", file=sys.stderr)
            return 1
    if args.command == "check":
        failures = []
        if sys.version_info < (3, 10):
            failures.append("Python >= 3.10 is required")
        needs_api_key = settings.knowledge_provider == "qwen" or settings.knowledge_retrieval_mode != "keyword" or settings.knowledge_rerank_enabled
        if needs_api_key and not settings.dashscope_api_key.strip():
            failures.append("DASHSCOPE_API_KEY is missing from .env")
        versions = {}
        if settings.knowledge_vector_store == "milvus_lite":
            for package in ["pymilvus", "milvus-lite", "faiss-cpu", "pyarrow"]:
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    failures.append(f"Missing package: {package}; run scripts/setup.sh")
        print(json.dumps({
            "python": sys.version.split()[0], "provider": settings.knowledge_provider,
            "model": settings.knowledge_model, "vectorStore": settings.knowledge_vector_store,
            "knowledgeDirectory": str(settings.knowledge_base_dir),
            "milvusPath": str(settings.knowledge_milvus_uri),
            "catalogPath": str(settings.knowledge_index_catalog_path),
            "dependencies": versions, "errors": failures,
        }, ensure_ascii=False, indent=2))
        return 1 if failures else 0

    paths = {"indexes": ("GET", "/api/knowledge/indexes"), "rebuild": ("POST", "/api/knowledge/indexes/rebuild")}
    if args.command == "activate":
        if not args.version or not all(c in "0123456789abcdef" for c in args.version) or len(args.version) != 64:
            parser.error("activate requires the full 64-character index version")
        paths["activate"] = ("POST", f"/api/knowledge/indexes/{args.version}/activate")
    method, path = paths[args.command]
    headers = {"Authorization": f"Bearer {settings.agent_api_token}"} if settings.agent_api_token else {}
    request = urllib.request.Request(args.api_base.rstrip("/") + path, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=600) as response:
            print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
        return 0
    except urllib.error.HTTPError as error:
        print(f"API returned HTTP {error.code}; inspect current index status and backend logs", file=sys.stderr)
    except urllib.error.URLError:
        print("Backend is not reachable. Start scripts/start.sh before running index commands.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
