import asyncio
import json
import logging
import os
import re
import shutil
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from app.code_conversion.engine import CodeConversionEngine
from app.core.config import Settings
from app.schemas.code_conversion import (
    ConversionFile,
    ConversionFileContent,
    ConversionJob,
    ConversionLogEntry,
)


JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_LOG_ENTRIES = 300
MAX_VIEWABLE_FILE_BYTES = 2 * 1024 * 1024
VIEWABLE_JSON_FILES = {"algorithm.json", "acc.json", "loss.json"}
RECOVERABLE_STATES = {
    "queued",
    "extracting",
    "detecting",
    "converting",
    "validating",
    "packaging",
}
LOGGER = logging.getLogger(__name__)


class ConversionJobNotFoundError(LookupError):
    pass


class ConversionJobConflictError(ValueError):
    pass


class ConversionUploadError(ValueError):
    pass


class CodeConversionService:
    def __init__(self, settings: Settings, client: Any):
        self.settings = settings
        self.storage_dir = settings.code_conversion_storage_dir.resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.engine = CodeConversionEngine(settings, client)
        self._tasks: dict[str, asyncio.Task] = {}
        self._records: dict[str, ConversionJob] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        job_id: str,
        filename: str,
        chunks: AsyncIterator[bytes],
    ) -> ConversionJob:
        self._validate_job_id(job_id)
        safe_filename = Path(filename or "algorithm.zip").name
        if not safe_filename.lower().endswith(".zip"):
            raise ConversionUploadError("算法包必须是 ZIP 文件")
        job_dir = self.storage_dir / job_id
        async with self._lock:
            if job_dir.exists():
                raise ConversionJobConflictError("转换任务已存在")
            job_dir.mkdir(mode=0o750)

        upload_temp = job_dir / "input.uploading"
        upload_path = job_dir / "input.zip"
        size = 0
        try:
            with upload_temp.open("xb") as output:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self.settings.code_conversion_max_upload_bytes:
                        raise ConversionUploadError("算法包超过上传大小限制")
                    output.write(chunk)
            if size <= 0:
                raise ConversionUploadError("算法包不能为空")
            os.replace(upload_temp, upload_path)
            now = datetime.now(timezone.utc).isoformat()
            record = ConversionJob(
                job_id=job_id,
                message="算法包已上传，等待语言检测",
                logs=[
                    ConversionLogEntry(
                        sequence=1,
                        timestamp=now,
                        phase="queued",
                        message=(
                            f"已接收算法包 {safe_filename}，"
                            f"大小 {self._format_bytes(size)}，转换任务进入队列"
                        ),
                    )
                ],
            )
            await self._save(record)
            self._tasks[job_id] = asyncio.create_task(
                self._run(job_id, upload_path), name=f"code-conversion-{job_id}"
            )
            return record
        except Exception:
            await asyncio.to_thread(shutil.rmtree, job_dir, True)
            raise

    async def get(self, job_id: str) -> ConversionJob:
        self._validate_job_id(job_id)
        record = self._records.get(job_id)
        if record:
            return record
        record_path = self.storage_dir / job_id / "job.json"
        try:
            value = json.loads(record_path.read_text(encoding="utf-8"))
            record = ConversionJob.model_validate(value)
        except FileNotFoundError as error:
            raise ConversionJobNotFoundError("转换任务不存在") from error
        self._records[job_id] = record
        return record

    async def recover_incomplete_jobs(self) -> int:
        recovered = 0
        for job_dir in sorted(self.storage_dir.iterdir()):
            if not job_dir.is_dir() or not JOB_ID_PATTERN.fullmatch(job_dir.name):
                continue
            try:
                record = await self.get(job_dir.name)
            except Exception:
                LOGGER.exception("忽略无法读取的代码转换任务 %s", job_dir.name)
                continue
            if record.status not in RECOVERABLE_STATES:
                continue
            upload_path = job_dir / "input.zip"
            if not upload_path.is_file():
                await self._update(
                    record.job_id,
                    status="failed",
                    phase="failed",
                    message="服务重启后无法恢复转换任务",
                    error="原始算法 ZIP 不存在",
                    append_log=True,
                    log_level="error",
                )
                continue
            await self._update(
                record.job_id,
                status="queued",
                phase="queued",
                progress=0,
                message="检测到服务重启，正在从原始算法包恢复转换",
                append_log=True,
            )
            self._tasks[record.job_id] = asyncio.create_task(
                self._run(record.job_id, upload_path),
                name=f"code-conversion-recovery-{record.job_id}",
            )
            recovered += 1
        if recovered:
            LOGGER.info("已恢复 %s 个未完成的代码转换任务", recovered)
        return recovered

    async def artifact(self, job_id: str) -> Path:
        record = await self.get(job_id)
        path = self.storage_dir / job_id / "artifact.zip"
        if record.status != "completed" or not record.artifact_available or not path.is_file():
            raise ConversionJobNotFoundError("转换产物尚未生成")
        return path

    async def converted_files(self, job_id: str) -> list[ConversionFile]:
        record = await self.get(job_id)
        viewable_root = self._viewable_root(job_id, record)

        files: list[ConversionFile] = []
        for file_path in viewable_root.rglob("*"):
            if not file_path.is_file():
                continue
            file_type = self._viewable_file_type(file_path)
            if not file_type:
                continue
            size = file_path.stat().st_size
            if size > MAX_VIEWABLE_FILE_BYTES:
                continue
            relative = file_path.relative_to(viewable_root).as_posix()
            files.append(
                ConversionFile(
                    name=file_path.name,
                    path=relative,
                    size=size,
                    type=file_type,
                    language="python" if file_type == "code" else "json",
                )
            )
        order = {"code": 0, "manifest": 1, "metric": 2}
        return sorted(files, key=lambda item: (order[item.type], item.path))

    async def converted_file_content(
        self, job_id: str, relative_path: str
    ) -> ConversionFileContent:
        record = await self.get(job_id)
        viewable_root = self._viewable_root(job_id, record).resolve()
        path_value = self._safe_relative_path(relative_path)
        file_path = (viewable_root / Path(*path_value.parts)).resolve()
        if viewable_root not in file_path.parents or not file_path.is_file():
            raise ConversionJobNotFoundError("转换文件不存在")
        file_type = self._viewable_file_type(file_path)
        if not file_type:
            raise ConversionUploadError("该文件类型不允许在线查看")
        size = file_path.stat().st_size
        if size > MAX_VIEWABLE_FILE_BYTES:
            raise ConversionUploadError("文件过大，不能在线查看")
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ConversionUploadError("文件不是 UTF-8 文本") from error
        return ConversionFileContent(
            name=file_path.name,
            path=path_value.as_posix(),
            size=size,
            type=file_type,
            language="python" if file_type == "code" else "json",
            content=content,
        )

    async def remove(self, job_id: str) -> None:
        self._validate_job_id(job_id)
        task = self._tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        job_dir = self.storage_dir / job_id
        if not job_dir.exists():
            raise ConversionJobNotFoundError("转换任务不存在")
        await asyncio.to_thread(shutil.rmtree, job_dir)
        self._records.pop(job_id, None)

    async def close(self) -> None:
        tasks = [
            (job_id, task)
            for job_id, task in self._tasks.items()
            if not task.done()
        ]
        for _, task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(
                *(task for _, task in tasks), return_exceptions=True
            )
        for job_id, _ in tasks:
            record = await self.get(job_id)
            if record.status in RECOVERABLE_STATES:
                await self._update(
                    job_id,
                    status="queued",
                    phase="queued",
                    message="转换服务正在重启，任务将在启动后自动恢复",
                    append_log=True,
                )

    async def _run(self, job_id: str, upload_path: Path) -> None:
        try:
            outcome = await self.engine.convert(
                upload_path,
                upload_path.parent,
                lambda progress, phase, message, details=None: self._progress(
                    job_id,
                    progress,
                    phase,
                    message,
                    details,
                ),
            )
            await self._update(
                job_id,
                status="completed",
                phase="completed",
                progress=100,
                message=(
                    "检测到 Python 源码，无需转换"
                    if not outcome.conversion_required
                    else "已完成 Python 转换并生成算法包"
                ),
                conversion_required=outcome.conversion_required,
                source_language=outcome.source_language,
                source_files=outcome.source_files,
                generated_files=outcome.generated_files,
                artifact_available=outcome.artifact_available,
                error=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._update(
                job_id,
                status="failed",
                phase="failed",
                message="代码语言检测或转换失败",
                error=str(error),
                append_log=True,
                log_level="error",
            )
            LOGGER.exception("代码转换任务失败 job_id=%s", job_id)
        finally:
            self._tasks.pop(job_id, None)

    async def _progress(
        self,
        job_id: str,
        progress: int,
        phase: str,
        message: str,
        details: dict[str, Any] | None,
    ) -> None:
        changes: dict[str, Any] = {
            "status": phase,
            "phase": phase,
            "progress": progress,
            "message": message,
        }
        if details:
            changes.update(details)
        await self._update(
            job_id,
            append_log=True,
            log_level="success" if phase == "completed" else "info",
            **changes,
        )
        LOGGER.info("代码转换任务 %s [%s %s%%] %s", job_id, phase, progress, message)

    async def _update(
        self,
        job_id: str,
        *,
        append_log: bool = False,
        log_level: str = "info",
        **changes: Any,
    ) -> None:
        record = await self.get(job_id)
        updated_at = datetime.now(timezone.utc).isoformat()
        changes["updated_at"] = updated_at
        if append_log:
            message = str(changes.get("error") or changes.get("message") or "").strip()
            if message:
                sequence = record.logs[-1].sequence + 1 if record.logs else 1
                changes["logs"] = [
                    *record.logs,
                    ConversionLogEntry(
                        sequence=sequence,
                        timestamp=updated_at,
                        phase=str(changes.get("phase") or record.phase),
                        level=log_level,
                        message=message,
                    ),
                ][-MAX_LOG_ENTRIES:]
        await self._save(record.model_copy(update=changes))

    async def _save(self, record: ConversionJob) -> None:
        job_dir = self.storage_dir / record.job_id
        path = job_dir / "job.json"
        temp_path = job_dir / "job.json.tmp"
        payload = json.dumps(record.public_dict(), ensure_ascii=False, indent=2) + "\n"
        async with self._lock:
            temp_path.write_text(payload, encoding="utf-8")
            os.replace(temp_path, path)
            self._records[record.job_id] = record

    def _validate_job_id(self, job_id: str) -> None:
        if not JOB_ID_PATTERN.fullmatch(job_id or ""):
            raise ConversionUploadError("转换任务 ID 不合法")

    def _safe_relative_path(self, value: str) -> PurePosixPath:
        if not value or "\\" in value or "\x00" in value or len(value) > 1024:
            raise ConversionUploadError("转换文件路径不合法")
        path_value = PurePosixPath(value)
        if (
            path_value.is_absolute()
            or any(part in {"", ".", ".."} for part in path_value.parts)
        ):
            raise ConversionUploadError("转换文件路径不合法")
        return path_value

    def _viewable_root(self, job_id: str, record: ConversionJob) -> Path:
        if record.status != "completed":
            raise ConversionJobNotFoundError("算法文件尚未准备完成")
        converted_root = self.storage_dir / job_id / "converted"
        if converted_root.is_dir():
            return converted_root
        if record.conversion_required is False or record.source_language == "python":
            extracted_root = self.storage_dir / job_id / "extracted"
            if extracted_root.is_dir():
                return extracted_root
        raise ConversionJobNotFoundError("算法文件尚未生成")

    def _viewable_file_type(self, file_path: Path) -> str | None:
        if file_path.suffix.lower() == ".py":
            return "code"
        if file_path.name == "algorithm.json":
            return "manifest"
        if file_path.name in VIEWABLE_JSON_FILES:
            return "metric"
        return None

    def _format_bytes(self, size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024 or unit == "GB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{size} B"
