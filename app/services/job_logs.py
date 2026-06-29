import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from app.core.config import Settings


LOG_CANDIDATES = (
    "live.log",
    "stdout.log",
    "stderr.log",
    "job.log",
)


class JobLogService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def job_output_dir(self, job_id: int) -> Path:
        if job_id <= 0:
            raise ValueError("job_id must be positive")
        root = self.settings.job_workspace_root.resolve()
        output_dir = (root / str(job_id) / "output").resolve()
        if root not in output_dir.parents:
            raise ValueError("Invalid job output directory")
        return output_dir

    def find_log_file(self, job_id: int) -> Path | None:
        output_dir = self.job_output_dir(job_id)
        for name in LOG_CANDIDATES:
            candidate = output_dir / name
            if candidate.is_file():
                return candidate
        if output_dir.is_dir():
            files = sorted(
                (
                    path
                    for path in output_dir.iterdir()
                    if path.is_file() and path.suffix in {".log", ".out", ".err"}
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if files:
                return files[0]
        return None

    async def follow(self, job_id: int) -> AsyncIterator[tuple[str, str]]:
        current_path: Path | None = None
        position = 0
        waiting_announced = False

        while True:
            path = self.find_log_file(job_id)
            if path is None:
                if not waiting_announced:
                    yield "status", "等待算法日志文件..."
                    waiting_announced = True
                await asyncio.sleep(self.settings.job_log_poll_interval)
                continue

            if path != current_path:
                current_path = path
                waiting_announced = False
                size = path.stat().st_size
                position = max(0, size - self.settings.job_log_max_initial_bytes)
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(position)
                    snapshot = stream.read()
                    position = stream.tell()
                if snapshot:
                    yield "snapshot", snapshot

            size = path.stat().st_size
            if size < position:
                position = 0
            if size > position:
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(position)
                    content = stream.read()
                    position = stream.tell()
                if content:
                    yield "append", content

            await asyncio.sleep(self.settings.job_log_poll_interval)
