import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ENTRY_ID_RE = re.compile(r"\b[A-Z]{2,8}-\d{3}\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", re.IGNORECASE)
TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")


class KnowledgeUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    chapter: str
    section: str
    entry_id: str | None
    content: str
    line_start: int
    line_end: int

    @property
    def source_label(self) -> str:
        parts = [part for part in (self.chapter, self.section, self.entry_id) if part]
        return " / ".join(parts)


@dataclass(frozen=True)
class KnowledgeSearchResult:
    chunk: KnowledgeChunk
    score: float

    def source_dict(self) -> dict[str, str | int | float | None]:
        return {
            "chapter": self.chunk.chapter,
            "section": self.chunk.section,
            "entry_id": self.chunk.entry_id,
            "line_start": self.chunk.line_start,
            "line_end": self.chunk.line_end,
            "score": round(self.score, 4),
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


def parse_markdown(text: str) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    chapter = ""
    section = ""
    table_headers: list[str] | None = None
    buffer: list[str] = []
    buffer_start = 0

    def append_chunk(content: str, line_start: int, line_end: int):
        content = content.strip()
        if not content:
            return
        entry_match = ENTRY_ID_RE.search(content)
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"chunk-{len(chunks) + 1}",
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


class KnowledgeService:
    def __init__(self, path: Path, top_k: int = 5):
        self.path = path.resolve()
        self.top_k = top_k
        self.chunks: list[KnowledgeChunk] = []
        self._term_counts: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0
        self._mtime_ns: int | None = None
        self._load_error: str | None = None

    def reload_if_changed(self, force: bool = False) -> bool:
        try:
            stat = self.path.stat()
            if not force and self._mtime_ns == stat.st_mtime_ns and self.chunks:
                return False

            text = self.path.read_text(encoding="utf-8")
            chunks = parse_markdown(text)
            if not chunks:
                raise ValueError("knowledge base contains no searchable content")

            term_counts = [
                Counter(_tokens(f"{chunk.chapter} {chunk.section} {chunk.content}"))
                for chunk in chunks
            ]
            document_frequency: Counter[str] = Counter()
            for counts in term_counts:
                document_frequency.update(counts.keys())

            self.chunks = chunks
            self._term_counts = term_counts
            self._document_frequency = document_frequency
            self._average_length = sum(sum(counts.values()) for counts in term_counts) / len(chunks)
            self._mtime_ns = stat.st_mtime_ns
            self._load_error = None
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

            title_compact = _compact(f"{chunk.chapter}{chunk.section}")
            content_compact = _compact(chunk.content)
            if len(query_compact) >= 2 and query_compact in content_compact:
                score += 8.0
            if len(query_compact) >= 2 and query_compact in title_compact:
                score += 5.0

            if score > 0:
                results.append(KnowledgeSearchResult(chunk=chunk, score=score))

        results.sort(key=lambda item: (-item.score, item.chunk.line_start))
        return results[: top_k or self.top_k]

    def format_context(self, results: list[KnowledgeSearchResult]) -> str:
        blocks = []
        for index, result in enumerate(results, start=1):
            blocks.append(
                f"[资料{index}] 来源：{result.chunk.source_label}，"
                f"原文行：{result.chunk.line_start}-{result.chunk.line_end}\n"
                f"{result.chunk.content}"
            )
        return "\n\n".join(blocks)

    def info(self) -> dict[str, str | int | bool | None]:
        self.reload_if_changed()
        last_modified = None
        if self._mtime_ns is not None:
            last_modified = datetime.fromtimestamp(self._mtime_ns / 1_000_000_000).astimezone().isoformat()
        return {
            "ready": bool(self.chunks),
            "document": self.path.name,
            "chunkCount": len(self.chunks),
            "lastModified": last_modified,
            "loadError": self._load_error,
        }
