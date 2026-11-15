"""Validate reference identity before streaming it; this is not claim entailment."""
import re

from app.services.knowledge import ENTRY_ID_RE, KnowledgeSearchResult


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


class CitationGuard:
    def __init__(self, results: list[KnowledgeSearchResult], allowed_numbers: list[int]):
        self.pending = ""
        self.references = {}
        self.validated: dict[int, dict] = {}
        self.invalid: list[str] = []
        self.seen_text = ""
        # Cross-references explicitly present in an approved passage may be mentioned,
        # but cannot be cited as independent sources without their own approved chunk.
        self.allowed_entries = {
            entry.upper() for number in allowed_numbers
            for entry in ENTRY_ID_RE.findall(results[number - 1].chunk.content)
        } | {
            results[number - 1].chunk.entry_id.upper() for number in allowed_numbers
            if results[number - 1].chunk.entry_id
        }
        for number in allowed_numbers:
            result = results[number - 1]
            # Bare entry IDs must be unambiguous across documents.
            aliases = [f"资料{number}", result.chunk.source_label]
            if result.chunk.entry_id:
                aliases.append(f"资料{number} / {result.chunk.entry_id}")
            chapter = re.search(r"第[一二三四五六七八九十百零〇\d]+章", result.chunk.chapter)
            section = re.match(r"\d+(?:\.\d+)*", result.chunk.section)
            if chapter and section and result.chunk.entry_id:
                aliases.append(f"{chapter[0]} / {section[0]} / {result.chunk.entry_id}")
                aliases.append(f"{chapter[0]} / {result.chunk.entry_id}")
                aliases.append(f"资料{number} / {chapter[0]} / {section[0]} / {result.chunk.entry_id}")
            if result.chunk.entry_id and sum(
                r.chunk.entry_id == result.chunk.entry_id for r in results
            ) == 1:
                aliases.append(result.chunk.entry_id)
            for alias in set(aliases):
                if alias:
                    self.references.setdefault(compact(alias), []).append((number, result))

    def _reference(self, value: str) -> str:
        reference = re.sub(r"^\s*(?:来源|引用)\s*[:：]\s*", "", value)
        matches = self.references.get(compact(reference), [])
        if len(matches) == 1:
            number, result = matches[0]
            self.validated[number] = {"sourceNumber": number, **result.source_dict()}
            return f"[资料{number}]"
        looks_like_reference = bool(
            re.search(r"资料|第.{1,12}[章节]|[A-Za-z]{2,8}-\d+", value)
            or re.fullmatch(r"\s*\d+\s*", value)
        )
        if looks_like_reference:
            self.invalid.append(value[:200])
            return "〔引用未通过校验〕"
        return f"[{value}]"

    def feed(self, text: str, *, final: bool = False) -> str:
        self.seen_text += text
        self.pending += text
        output = []
        while self.pending:
            start = self.pending.find("[")
            if start < 0:
                output.append(self.pending)
                self.pending = ""
                break
            output.append(self.pending[:start])
            self.pending = self.pending[start:]
            end = self.pending.find("]")
            if end < 0:
                if final or len(self.pending) > 256:
                    self.invalid.append(self.pending[1:201])
                    output.append("〔引用不完整，未通过校验〕")
                    self.pending = ""
                break
            output.append(self._reference(self.pending[1:end]))
            self.pending = self.pending[end + 1:]
        return "".join(output)

    def report(self) -> dict:
        unknown_entries = sorted({entry.upper() for entry in ENTRY_ID_RE.findall(self.seen_text)} - self.allowed_entries)
        invalid = list(self.invalid) + [f"未核验条目：{entry}" for entry in unknown_entries]
        return {
            "passed": bool(self.validated) and not invalid,
            "missingCitation": not bool(self.validated),
            "invalidReferences": invalid,
            "citations": list(self.validated.values()),
            "scope": "reference_identity_only",
        }
