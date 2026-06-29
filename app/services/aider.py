import asyncio
import os
import re
import sys
import uuid
from pathlib import Path

import pexpect

from app.core.config import Settings, get_settings


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PROMPT_RE = re.compile(r"(?:diff|ask|architect|code|help)>\s*$")


def clean_terminal_output(text: str) -> str:
    text = ANSI_RE.sub("", text)
    while "\b" in text:
        cleaned = re.sub(r".\x08", "", text)
        if cleaned == text:
            text = text.replace("\b", "")
            break
        text = cleaned
    return PROMPT_RE.sub("", text)


class AiderSession:
    def __init__(self, session_id: str, workspace: Path, settings: Settings):
        self.session_id = session_id
        self.workspace = workspace
        self.settings = settings
        self.child: pexpect.spawn | None = None
        self.output_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.reader_task: asyncio.Task | None = None
        self.busy = False

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = self.settings.dashscope_api_key
        env["OPENAI_API_BASE"] = self.settings.dashscope_base_url
        source_root = str(self.settings.aider_source_root.resolve())
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in [source_root, env.get("PYTHONPATH", "")] if item
        )
        env["TERM"] = "dumb"
        return env

    def _build_args(self, files: list[str]) -> list[str]:
        args = [
            "-m",
            "aider",
            "--model",
            self.settings.aider_model,
            "--edit-format",
            self.settings.aider_edit_format,
            "--no-pretty",
            "--yes-always",
            "--no-stream",
        ]
        args.append(
            "--auto-commits"
            if self.settings.aider_auto_commits
            else "--no-auto-commits"
        )
        args.extend(files)
        return args

    def _resolve_files(self, files: list[str]) -> list[str]:
        resolved: list[str] = []
        for value in files:
            candidate = (self.workspace / value).resolve()
            if self.workspace not in candidate.parents and candidate != self.workspace:
                raise ValueError(f"File is outside workspace: {value}")
            if not candidate.exists():
                raise ValueError(f"File does not exist: {value}")
            resolved.append(str(candidate))
        return resolved

    async def start(self, files: list[str]) -> None:
        if self.child and self.child.isalive():
            return
        if not self.settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        resolved_files = self._resolve_files(files)
        self.child = await asyncio.to_thread(
            pexpect.spawn,
            command=sys.executable,
            args=self._build_args(resolved_files),
            cwd=str(self.workspace),
            env=self._build_env(),
            encoding="utf-8",
            codec_errors="replace",
            timeout=1,
            dimensions=(40, 160),
        )
        self.child.setecho(False)
        self.reader_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> None:
        assert self.child is not None
        while self.child.isalive():
            try:
                text = await asyncio.to_thread(
                    self.child.read_nonblocking,
                    4096,
                    0.25,
                )
            except pexpect.TIMEOUT:
                await asyncio.sleep(0.05)
                continue
            except pexpect.EOF:
                break
            cleaned = clean_terminal_output(text)
            if cleaned:
                await self.output_queue.put(cleaned)
            if PROMPT_RE.search(text) or text.rstrip().endswith(">"):
                self.busy = False
                await self.output_queue.put("")
        self.busy = False
        await self.output_queue.put(None)

    async def send(self, content: str) -> None:
        if not self.child or not self.child.isalive():
            raise RuntimeError("Aider session is not running")
        if self.busy:
            raise RuntimeError("Aider is still processing the previous request")
        self.busy = True
        self.child.sendline(content)

    async def interrupt(self) -> None:
        if self.child and self.child.isalive():
            self.child.sendcontrol("c")
        self.busy = False

    async def close(self) -> None:
        if self.child and self.child.isalive():
            self.child.close(force=True)
        if self.reader_task:
            self.reader_task.cancel()
            try:
                await self.reader_task
            except asyncio.CancelledError:
                pass


class AiderSessionManager:
    def __init__(self):
        self.sessions: dict[str, AiderSession] = {}

    async def create(
        self,
        workspace_name: str,
        files: list[str],
    ) -> AiderSession:
        settings = get_settings()
        workspace = settings.resolve_workspace(workspace_name)
        session_id = uuid.uuid4().hex
        session = AiderSession(session_id, workspace, settings)
        await session.start(files)
        self.sessions[session_id] = session
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        await asyncio.gather(
            *(session.close() for session in self.sessions.values()),
            return_exceptions=True,
        )
        self.sessions.clear()
