"""Transactional index publication metadata, separate from reusable embedding cache."""
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path


class IndexCatalog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS generations (
                version TEXT PRIMARY KEY, collection_name TEXT NOT NULL UNIQUE,
                profile TEXT NOT NULL, chunks TEXT NOT NULL, documents TEXT NOT NULL,
                created_at TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
                reused_vectors INTEGER NOT NULL DEFAULT 0, new_vectors INTEGER NOT NULL DEFAULT 0
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS publication (
                id INTEGER PRIMARY KEY CHECK(id=1), active_version TEXT, source_version TEXT
            )""")
            db.execute("INSERT OR IGNORE INTO publication(id) VALUES (1)")

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    def state(self) -> dict:
        with self._connection() as db:
            return dict(db.execute("SELECT * FROM publication WHERE id=1").fetchone())

    def get(self, version: str) -> dict | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM generations WHERE version=?", (version,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["chunks"] = json.loads(result["chunks"])
        result["documents"] = json.loads(result["documents"])
        return result

    def begin(self, version: str, collection: str, profile: str, chunks: list[dict], documents: list[dict]):
        with self._connection() as db:
            db.execute("""INSERT INTO generations
                (version, collection_name, profile, chunks, documents, created_at, status)
                VALUES (?,?,?,?,?,?,'building')
                ON CONFLICT(version) DO UPDATE SET status='building', error=NULL""", (
                version, collection, profile, json.dumps(chunks, ensure_ascii=False),
                json.dumps(documents, ensure_ascii=False), datetime.now(timezone.utc).isoformat(),
            ))

    def ready(self, version: str, reused: int, new: int):
        with self._connection() as db:
            db.execute("UPDATE generations SET status='ready', error=NULL, reused_vectors=?, new_vectors=? WHERE version=?", (reused, new, version))

    def fail(self, version: str, error_kind: str):
        with self._connection() as db:
            db.execute("UPDATE generations SET status='failed', error=? WHERE version=? AND status='building'", (error_kind, version))

    def replace_collection(self, version: str, collection: str, reused: int, new: int):
        """Publish a validated physical repair without dropping the old collection."""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM generations WHERE version=?", (version,)).fetchone()
            if row is None or row["status"] != "ready":
                raise ValueError("Only ready index versions can be repaired")
            db.execute("UPDATE generations SET collection_name=?, reused_vectors=?, new_vectors=? WHERE version=?", (collection, reused, new, version))

    def activate(self, version: str, *, observe_source: bool):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM generations WHERE version=?", (version,)).fetchone()
            if row is None or row["status"] != "ready":
                raise ValueError("Only validated ready indexes can be activated")
            if observe_source:
                db.execute("UPDATE publication SET active_version=?, source_version=? WHERE id=1", (version, version))
            else:
                db.execute("UPDATE publication SET active_version=? WHERE id=1", (version,))

    def list(self) -> list[dict]:
        state = self.state()
        with self._connection() as db:
            rows = db.execute("SELECT version, collection_name, profile, created_at, status, error, reused_vectors, new_vectors FROM generations ORDER BY created_at DESC").fetchall()
        return [dict(row) | {"active": row["version"] == state["active_version"]} for row in rows]
