"""Conservative text-only context admission without downloading a model tokenizer.

UTF-8 bytes plus framing are a deliberately conservative token *estimate*, not a
provider-specific tokenizer. A configured input ceiling and output reserve are
both enforced. Mandatory instructions/question/evidence are never cut in half.
"""
from dataclasses import dataclass


class ContextBudgetExceeded(RuntimeError):
    def __init__(self):
        super().__init__("当前问题或检索资料超过上下文预算，请缩短问题或减少检索片段后重试。")


def text_tokens(text: str) -> int:
    return len(text.encode("utf-8"))


def clip_text(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    return text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")


def message_tokens(messages: list[dict]) -> int:
    return 64 + sum(16 + text_tokens(m["role"]) + text_tokens(m["content"]) for m in messages)


@dataclass
class PackedContext:
    messages: list[dict]
    metrics: dict


class ContextBudget:
    def __init__(self, settings, output_tokens: int):
        self.limit = min(settings.knowledge_max_input_tokens,
                         settings.knowledge_context_window_tokens - output_tokens - settings.knowledge_context_safety_tokens)

    def pack(self, messages: list[dict], limit: int | None = None) -> PackedContext:
        limit = min(self.limit, limit) if limit is not None else self.limit
        if not messages:
            raise ContextBudgetExceeded()
        mandatory = ([messages[0], *[m for m in messages[1:-1] if m.get("_context_required")], messages[-1]]
                     if len(messages) > 1 else messages[:])
        if message_tokens(mandatory) > limit:
            raise ContextBudgetExceeded()
        middle = list(messages[1:-1])
        removed = 0
        while len(messages) > 1 and message_tokens([messages[0], *middle, messages[-1]]) > limit:
            index = next(i for i, m in enumerate(middle) if not m.get("_context_required"))
            middle.pop(index)
            removed += 1
            # A cropped conversation must not start with an orphaned answer.
            while middle and middle[0]["role"] == "assistant" and not middle[0].get("_context_required"):
                middle.pop(0)
                removed += 1
        packed = [messages[0], *middle, messages[-1]] if len(messages) > 1 else messages[:]
        packed = [{"role": m["role"], "content": m["content"]} for m in packed]
        return PackedContext(packed, {"estimatedInputTokens": message_tokens(packed),
            "inputTokenBudget": limit, "contextMessagesDropped": removed,
            "tokenEstimator": "utf8_bytes_v1"})
