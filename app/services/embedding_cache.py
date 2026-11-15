"""Content-addressed vector cache; contains hashes and vectors, not document text."""
import sqlite3
import struct
from contextlib import closing
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path: Path, dimensions: int):
        self.path = path
        self.dimensions = dimensions
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS vectors "
                "(cache_key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, vector BLOB NOT NULL)"
            )
            database.commit()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        results = {}
        with closing(sqlite3.connect(self.path)) as database:
            for key in keys:
                row = database.execute(
                    "SELECT dimensions, vector FROM vectors WHERE cache_key = ?", (key,)
                ).fetchone()
                if row and row[0] == self.dimensions and len(row[1]) == self.dimensions * 4:
                    results[key] = list(struct.unpack(f"<{self.dimensions}f", row[1]))
        return results

    def put_many(self, vectors: dict[str, list[float]]):
        rows = [
            (key, self.dimensions, struct.pack(f"<{self.dimensions}f", *vector))
            for key, vector in vectors.items()
        ]
        with closing(sqlite3.connect(self.path)) as database:
            database.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)", rows)
            database.commit()
