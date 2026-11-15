import math
import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


ENTRY_ID_RE = re.compile(r"(?<![A-Z0-9_-])[A-Z]{2,8}-\d{3}(?![A-Z0-9_-])", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", re.IGNORECASE)
TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
SUPPORTED_KNOWLEDGE_EXTENSIONS = {".md", ".markdown", ".txt"}
KNOWLEDGE_INTENT_TERMS = (
    "锁闭", "密贴", "堵转", "齿轮", "电流曲线", "润滑", "机械磨损", "牵引", "制动", "车门",
    "故障",
    "异常",
    "报警",
    "告警",
    "诊断",
    "排查",
    "检修",
    "维修",
    "失效",
    "损坏",
    "转辙机",
    "道岔机",
    "道岔",
    "牵引电机",
    "轴承",
    "卡尔曼",
    "制动系统",
    "车门系统",
    "轨道电路",
    "受电弓",
    "轮对",
)
CONTEXTUAL_FOLLOW_UP_PREFIXES = (
    "如果", "若", "那", "那么", "是否就能", "为什么", "怎么判断", "还有", "能否排除",
    "这个",
    "这种",
    "该故障",
    "上述",
    "前面",
    "刚才",
    "它",
    "其",
    "那应该",
    "那么应该",
    "那要",
    "那么要",
    "那还",
    "那么还",
    "还需要",
    "下一步",
    "继续",
)
CONVERSATION_RECALL_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"(?:我|我们|你|您)(?:刚刚|刚才|之前|前面|上次|先前|曾经|刚)(?:都)?(?:问|说|提|聊|讨论)(?:了|过)?(?:你|您)?(?:什么|哪些|啥)",
    r"(?:我|我们|你|您)(?:问|说|提|聊|讨论)(?:了|过)(?:你|您)?(?:什么|哪些|啥)",
    r"(?:上一[条轮个次]|刚才|刚刚|之前|前面|先前).{0,8}(?:问题|消息|对话|聊天|提问).{0,8}(?:什么|哪些|总结|回顾|复述|记得)",
    r"(?:总结|回顾|复述|重复|列出).{0,8}(?:对话|聊天|提问|问过的问题|(?:上一[条轮个次]|刚才|刚刚|之前|前面).{0,4}(?:问题|消息))",
    r"还记得.{0,10}(?:问|说|聊)",
))


class KnowledgeUnavailableError(RuntimeError):
    pass


class KnowledgeValidationError(ValueError):
    pass


class KnowledgeDocumentConflictError(FileExistsError):
    pass


class KnowledgeDocumentNotFoundError(FileNotFoundError):
    pass


def is_contextual_follow_up(question: str) -> bool:
    normalized = "".join(question.split())
    return any(
        normalized.startswith(prefix)
        for prefix in CONTEXTUAL_FOLLOW_UP_PREFIXES
    )


def is_conversation_recall(question: str) -> bool:
    normalized = "".join(question.split())
    return any(pattern.search(normalized) for pattern in CONVERSATION_RECALL_PATTERNS)


def should_retrieve_knowledge(question: str) -> bool:
    normalized = "".join(question.lower().split())
    if not normalized:
        return False
    if ENTRY_ID_RE.search(question):
        return True
    return any(term in normalized for term in KNOWLEDGE_INTENT_TERMS)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document: str
    chapter: str
    section: str
    entry_id: str | None
    content: str
    line_start: int
    line_end: int

    @property
    def document_id(self) -> str:
        return hashlib.sha256(unicodedata.normalize("NFC", self.document).encode()).hexdigest()

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()

    @property
    def source_label(self) -> str:
        parts = [part for part in (self.chapter, self.section, self.entry_id) if part]
        return " / ".join(parts)


@dataclass(frozen=True)
class KnowledgeSearchResult:
    chunk: KnowledgeChunk
    score: float
    kb_version: str | None = None

    def source_dict(self) -> dict[str, str | int | float | None]:
        return {
            "document": self.chunk.document,
            "chapter": self.chunk.chapter,
            "section": self.chunk.section,
            "entry_id": self.chunk.entry_id,
            "line_start": self.chunk.line_start,
            "line_end": self.chunk.line_end,
            "score": round(self.score, 4),
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "content_hash": self.chunk.content_hash,
            "kb_version": self.kb_version,
        }


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(TABLE_SEPARATOR_RE.fullmatch(cell) for cell in cells)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())


def _tokens(text: str) -> list[str]:
    terms: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                terms.append(token)
                continue
            for size in (2, 3):
                if len(token) < size:
                    continue
                terms.extend(token[index : index + size] for index in range(len(token) - size + 1))
        else:
            terms.append(token)
    return terms


def parse_markdown(text: str, document: str = "") -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    chapter = ""
    section = ""
    table_headers: list[str] | None = None
    buffer: list[str] = []
    buffer_start = 0
    occurrences: Counter[str] = Counter()

    def append_chunk(content: str, line_start: int, line_end: int):
        content = content.strip()
        if not content:
            return
        entry_match = ENTRY_ID_RE.search(content)
        identity = json.dumps([
            "chunk-id-v2", unicodedata.normalize("NFC", document), chapter, section,
            entry_match.group(0).upper() if entry_match else None,
            re.sub(r"\s+", " ", content).strip(),
        ], ensure_ascii=False)
        occurrences[identity] += 1
        chunk_id = hashlib.sha256(f"{identity}\0{occurrences[identity]}".encode()).hexdigest()
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                document=document,
                chapter=chapter,
                section=section,
                entry_id=entry_match.group(0).upper() if entry_match else None,
                content=content,
                line_start=line_start,
                line_end=line_end,
            )
        )

    def flush_buffer(line_end: int):
        nonlocal buffer, buffer_start
        if buffer:
            append_chunk("\n".join(buffer), buffer_start, line_end)
            buffer = []
            buffer_start = 0

    lines = text.splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if line.startswith("## "):
            flush_buffer(line_number - 1)
            chapter = line[3:].strip()
            section = ""
            table_headers = None
            continue
        if line.startswith("### "):
            flush_buffer(line_number - 1)
            section = line[4:].strip()
            table_headers = None
            continue
        if line.startswith("# "):
            flush_buffer(line_number - 1)
            table_headers = None
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_buffer(line_number - 1)
            cells = _table_cells(line)
            if _is_table_separator(cells):
                continue
            if table_headers is None:
                table_headers = cells
                continue

            fields = []
            for index, value in enumerate(cells):
                if not value:
                    continue
                header = table_headers[index] if index < len(table_headers) else f"字段{index + 1}"
                fields.append(f"{header}: {value}")
            append_chunk("；".join(fields), line_number, line_number)
            continue

        table_headers = None
        if not line or line == "---":
            flush_buffer(line_number - 1)
            continue

        if not buffer:
            buffer_start = line_number
        buffer.append(line)
        if sum(len(item) for item in buffer) >= 1200:
            flush_buffer(line_number)

    flush_buffer(len(lines))
    return chunks


class Bm25Index:
    """Immutable lexical index shared by one published knowledge snapshot."""
    def __init__(self, chunks: tuple[KnowledgeChunk, ...]):
        self.chunks = chunks
        self._term_counts = [
            Counter(_tokens(f"{chunk.document} {chunk.chapter} {chunk.section} {chunk.content}"))
            for chunk in chunks
        ]
        self._document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())
        self._average_length = (
            sum(sum(counts.values()) for counts in self._term_counts) / len(chunks)
            if chunks else 0.0
        )

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeSearchResult]:
        query = query.strip()
        if not query:
            return []

        query_terms = Counter(_tokens(query))
        query_entry_ids = {match.group(0).upper() for match in ENTRY_ID_RE.finditer(query)}
        query_compact = _compact(query)
        document_count = len(self.chunks)
        average_length = self._average_length or 1.0
        results: list[KnowledgeSearchResult] = []

        for chunk, counts in zip(self.chunks, self._term_counts):
            document_length = sum(counts.values()) or 1
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * document_length / average_length
                )
                score += query_frequency * inverse_document_frequency * frequency * 2.5 / denominator

            if chunk.entry_id and chunk.entry_id in query_entry_ids:
                score += 100.0

            title_compact = _compact(f"{chunk.document}{chunk.chapter}{chunk.section}")
            content_compact = _compact(chunk.content)
            if len(query_compact) >= 2 and query_compact in content_compact:
                score += 8.0
            if len(query_compact) >= 2 and query_compact in title_compact:
                score += 5.0

            if score > 0:
                results.append(KnowledgeSearchResult(chunk=chunk, score=score))

        results.sort(
            key=lambda item: (
                -item.score,
                item.chunk.document.lower(),
                item.chunk.line_start,
            )
        )
        return results[:top_k]


class KnowledgeService:
    def __init__(
        self,
        path: Path,
        top_k: int = 5,
        storage_dir: Path | None = None,
        max_upload_bytes: int = 10 * 1024 * 1024,
    ):
        self.path = path.resolve()
        self.storage_dir = storage_dir.resolve() if storage_dir else None
        self.top_k = top_k
        self.max_upload_bytes = max_upload_bytes
        self.chunks: list[KnowledgeChunk] = []
        self._keyword_index = Bm25Index(())
        self._mtime_ns: int | None = None
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._load_error: str | None = None
        self._documents: list[dict[str, str | int]] = []

    def _prepare_storage(self):
        if self.storage_dir is None:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        initialized_marker = self.storage_dir / ".initialized"
        if initialized_marker.exists():
            return

        has_documents = any(self._iter_storage_documents())
        if self.path.is_file() and not has_documents:
            shutil.copy2(self.path, self.storage_dir / self.path.name)
        initialized_marker.touch(exist_ok=True)

    def _iter_storage_documents(self):
        if self.storage_dir is None or not self.storage_dir.is_dir():
            return iter(())
        return (
            item
            for item in sorted(self.storage_dir.iterdir(), key=lambda value: value.name.lower())
            if item.is_file() and item.suffix.lower() in SUPPORTED_KNOWLEDGE_EXTENSIONS
        )

    def _document_paths(self) -> list[Path]:
        if self.storage_dir is None:
            return [self.path] if self.path.is_file() else []
        self._prepare_storage()
        return list(self._iter_storage_documents())

    @staticmethod
    def _signature_for(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (path.name, stat.st_mtime_ns, stat.st_size)
            for path in paths
            for stat in (path.stat(),)
        )

    @staticmethod
    def _document_info(path: Path, chunk_count: int) -> dict[str, str | int]:
        stat = path.stat()
        return {
            "name": path.name,
            "size": stat.st_size,
            "chunkCount": chunk_count,
            "lastModified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        }

    def _replace_index(
        self,
        chunks: list[KnowledgeChunk],
        documents: list[dict[str, str | int]],
        signature: tuple[tuple[str, int, int], ...],
    ):
        self.chunks = chunks
        self._keyword_index = Bm25Index(tuple(chunks))
        self._documents = documents
        self._signature = signature
        self._mtime_ns = max((item[1] for item in signature), default=None)

    def reload_if_changed(self, force: bool = False) -> bool:
        try:
            paths = self._document_paths()
            signature = self._signature_for(paths)
            if not force and self._signature == signature:
                return False

            chunks: list[KnowledgeChunk] = []
            documents: list[dict[str, str | int]] = []
            for path in paths:
                text = path.read_text(encoding="utf-8-sig")
                document_chunks = parse_markdown(text, document=path.name)
                if not document_chunks:
                    raise ValueError(f"{path.name} contains no searchable content")
                chunks.extend(document_chunks)
                documents.append(self._document_info(path, len(document_chunks)))

            self._replace_index(chunks, documents, signature)
            self._load_error = None if chunks else "knowledge base is empty"
            return True
        except (OSError, UnicodeError, ValueError) as error:
            self._load_error = str(error)
            if not self.chunks:
                self._mtime_ns = None
            return False

    def _ensure_ready(self):
        self.reload_if_changed()
        if not self.chunks:
            detail = self._load_error or "knowledge base is not loaded"
            raise KnowledgeUnavailableError(detail)

    def search(self, query: str, top_k: int | None = None) -> list[KnowledgeSearchResult]:
        self._ensure_ready()
        return self._keyword_index.search(query, top_k or self.top_k)

    def snapshot(self) -> tuple[KnowledgeChunk, ...]:
        """Return immutable chunks for one retrieval, even if documents change later."""
        self._ensure_ready()
        return tuple(self.chunks)

    def format_context(self, results: list[KnowledgeSearchResult]) -> str:
        blocks = []
        for index, result in enumerate(results, start=1):
            blocks.append(
                f"[资料{index}] 文档：{result.chunk.document}；"
                f"来源：{result.chunk.source_label}；"
                f"片段ID：{result.chunk.chunk_id}；版本：{result.kb_version or '未发布'}；"
                f"原文行：{result.chunk.line_start}-{result.chunk.line_end}\n"
                f"{result.chunk.content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _safe_document_name(filename: str) -> str:
        name = Path(filename).name.strip()
        if not name or name != filename.strip() or name.startswith("."):
            raise KnowledgeValidationError("invalid knowledge document name")
        if Path(name).suffix.lower() not in SUPPORTED_KNOWLEDGE_EXTENSIONS:
            raise KnowledgeValidationError(
                "only .md, .markdown and .txt knowledge documents are supported"
            )
        return name

    def upload_document(self, filename: str, content: bytes) -> dict[str, str | int]:
        if self.storage_dir is None:
            raise KnowledgeValidationError("knowledge document storage is not configured")
        name = self._safe_document_name(filename)
        self._prepare_storage()
        target = self.storage_dir / name
        if target.exists():
            raise KnowledgeDocumentConflictError(f"knowledge document already exists: {name}")
        if not content:
            raise KnowledgeValidationError("knowledge document must not be empty")
        if len(content) > self.max_upload_bytes:
            raise KnowledgeValidationError(
                f"knowledge document exceeds {self.max_upload_bytes} bytes"
            )
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise KnowledgeValidationError("knowledge document must use UTF-8 encoding") from error
        if not parse_markdown(text, document=name):
            raise KnowledgeValidationError("knowledge document contains no searchable content")

        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=self.storage_dir,
                prefix=".upload-",
                suffix=target.suffix,
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
            temporary_path = None
            if not self.reload_if_changed(force=True):
                raise KnowledgeValidationError(
                    self._load_error or "knowledge index rebuild failed"
                )
        except Exception:
            target.unlink(missing_ok=True)
            self.reload_if_changed(force=True)
            raise
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return next(document for document in self._documents if document["name"] == name)

    def delete_document(self, filename: str) -> dict[str, str | int | bool | None | list]:
        if self.storage_dir is None:
            raise KnowledgeValidationError("knowledge document storage is not configured")
        name = self._safe_document_name(filename)
        self._prepare_storage()
        target = self.storage_dir / name
        if not target.is_file():
            raise KnowledgeDocumentNotFoundError(f"knowledge document not found: {name}")
        target.unlink()
        self.reload_if_changed(force=True)
        return self.info()

    def info(self) -> dict[str, str | int | bool | None | list]:
        self.reload_if_changed()
        last_modified = None
        if self._mtime_ns is not None:
            last_modified = datetime.fromtimestamp(
                self._mtime_ns / 1_000_000_000
            ).astimezone().isoformat()
        if len(self._documents) == 1:
            document_label = str(self._documents[0]["name"])
        elif self._documents:
            document_label = f"{len(self._documents)} 个知识库文档"
        else:
            document_label = "暂无知识库文档"
        return {
            "ready": bool(self.chunks),
            "document": document_label,
            "documentCount": len(self._documents),
            "documents": list(self._documents),
            "chunkCount": len(self.chunks),
            "lastModified": last_modified,
            "loadError": self._load_error,
        }
