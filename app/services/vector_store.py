"""Persistent dense-search adapter. One owner process; all client calls are serialized."""
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Protocol


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float


class VectorStore(Protocol):
    def build(self, collection: str, dimensions: int, rows: list[dict]) -> None: ...
    def valid(self, collection: str, dimensions: int, count: int) -> bool: ...
    def search(self, collection: str, vector: list[float], top_k: int) -> list[VectorHit]: ...
    def close(self) -> None: ...


class MilvusVectorStore:
    _owners_lock = RLock()
    _owned_paths: set[str] = set()

    def __init__(self, uri: Path):
        try:
            from pymilvus import MilvusClient
        except ImportError as error:
            raise RuntimeError("Install the Milvus dependencies: pip install -e '.[milvus]'") from error
        uri.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._uri = str(uri.resolve())
        self._closed = False
        with self._owners_lock:
            if self._uri in self._owned_paths:
                raise RuntimeError("Reuse the existing Milvus Lite client for this data directory")
            self.client = MilvusClient(uri=self._uri)
            self._owned_paths.add(self._uri)

    def build(self, collection: str, dimensions: int, rows: list[dict]) -> None:
        from pymilvus import DataType

        with self._lock:
            # Only unpublished staging collections are passed here by the index publisher.
            if self.client.has_collection(collection_name=collection):
                self.client.drop_collection(collection_name=collection)
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimensions)
            indexes = self.client.prepare_index_params()
            indexes.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
            self.client.create_collection(collection_name=collection, schema=schema, index_params=indexes)
            for start in range(0, len(rows), 200):
                self.client.insert(collection_name=collection, data=rows[start:start + 200])
            self.client.flush(collection_name=collection)
            self.client.load_collection(collection_name=collection)
            if not self.valid(collection, dimensions, len(rows)):
                raise RuntimeError("Milvus collection failed row-count or schema validation")
            if rows:
                hits = self.search(collection, rows[0]["vector"], 1)
                if not hits:
                    raise RuntimeError("Milvus staging collection cannot be searched")

    def valid(self, collection: str, dimensions: int, count: int) -> bool:
        with self._lock:
            if not self.client.has_collection(collection_name=collection):
                return False
            description = self.client.describe_collection(collection_name=collection)
            vector = next((f for f in description["fields"] if f["name"] == "vector"), None)
            if vector is None or int(vector.get("params", {}).get("dim", 0)) != dimensions:
                return False
            stats = self.client.get_collection_stats(collection_name=collection)
            if int(stats["row_count"]) != count:
                return False
            # Load state is process-local; a persisted collection starts released after restart.
            self.client.load_collection(collection_name=collection)
            return True

    def search(self, collection: str, vector: list[float], top_k: int) -> list[VectorHit]:
        with self._lock:
            result = self.client.search(
                collection_name=collection, data=[vector], limit=top_k,
                search_params={"metric_type": "COSINE", "params": {}},
            )
            return [VectorHit(str(hit["id"]), float(hit["distance"])) for hit in result[0]]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.client.close()
            finally:
                # PyMilvus.close closes the RPC connection; Lite's engine otherwise lives until exit.
                from milvus_lite.server_manager import server_manager_instance
                server_manager_instance.release_server(self._uri)
                with self._owners_lock:
                    self._owned_paths.discard(self._uri)
                self._closed = True
