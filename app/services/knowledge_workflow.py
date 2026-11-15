"""Bounded single-pass LangGraph workflow, keeping the existing WS protocol."""
import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.services.citation_guard import CitationGuard
from app.services.conversation_memory import ConversationMemory
from app.services.context_budget import ContextBudget, ContextBudgetExceeded, text_tokens
from app.services.knowledge import (
    ENTRY_ID_RE, KnowledgeSearchResult, KnowledgeService,
    citation_labels, is_contextual_follow_up, is_conversation_recall, should_retrieve_knowledge,
)


logger = logging.getLogger("uvicorn.error")
EQUIPMENT = (("转辙机", "道岔"), ("车门",), ("轴承",), ("牵引",), ("制动",), ("受电弓",), ("轨道电路",), ("轮对",))


class QuestionState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    topic: str
    last_mode: str | None
    follow_up: bool
    conversation_recall: bool
    summary: list[dict]
    recall_records: dict
    recall_ms: float
    memory_metrics: dict
    context_metrics: dict
    mode: str
    query: str
    rewrite: str
    results: list[KnowledgeSearchResult]
    answerable: bool
    evidence_numbers: list[int]
    missing: list[str]
    assessment_reason: str
    answer: str
    validation: dict
    timings: dict
    started: float
    first_content_ms: float | None
    retrieval_trace: dict


def emit(kind: str, content: str = "", **extra):
    get_stream_writer()({"type": kind, "content": content, **extra})


def timed(state: QuestionState, name: str, started: float) -> dict:
    return {**state.get("timings", {}), name: round((time.perf_counter() - started) * 1000, 2)}


class KnowledgeWorkflow:
    def __init__(self, settings: Settings, knowledge: KnowledgeService, llm, retriever=None):
        self.settings, self.knowledge, self.llm, self.retriever = settings, knowledge, llm, retriever
        self.memory = ConversationMemory(settings, llm)
        graph = StateGraph(QuestionState)
        for name in ("conversation", "retrieve", "assess", "generate", "clarify", "validate", "finish"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "conversation")
        graph.add_conditional_edges("conversation", lambda s: s["mode"], {"knowledge": "retrieve", "direct": "generate"})
        graph.add_edge("retrieve", "assess")
        graph.add_conditional_edges("assess", lambda s: s["answerable"], {True: "generate", False: "clarify"})
        graph.add_edge("generate", "validate")
        graph.add_edge("clarify", "validate")
        graph.add_edge("validate", "finish")
        graph.add_edge("finish", END)
        self.graph = graph.compile()

    async def stream(self, question: str, session: dict) -> AsyncIterator[dict]:
        state = {**session, "question": question, "started": time.perf_counter(), "timings": {}}
        # No checkpointer: each socket owns its session, and clear/disconnect drops it.
        async with aclosing(self.graph.astream(
            state, stream_mode="custom", version="v2", config={"recursion_limit": 12},
        )) as events:
            async for event in events:
                yield event["data"]

    async def conversation(self, state: QuestionState):
        started = time.perf_counter()
        question, history = state["question"], state.get("history", [])
        history, summary, memory_metrics = await self.memory.compact(history, state.get("summary", []))
        memory_metrics["memoryMs"] = (time.perf_counter() - started) * 1000
        topic = state.get("topic", "")
        recall = is_conversation_recall(question)
        follow_up = bool(history or summary) and (recall or is_contextual_follow_up(question))
        # Explicitly naming a different equipment family starts a fresh topic.
        previous_equipment = {n for n, terms in enumerate(EQUIPMENT) if any(term in topic for term in terms)}
        current_equipment = {n for n, terms in enumerate(EQUIPMENT) if any(term in question for term in terms)}
        if not recall and current_equipment and previous_equipment and not current_equipment & previous_equipment:
            follow_up = False
        if not recall and follow_up and state.get("last_mode") == "direct" and should_retrieve_knowledge(question):
            follow_up = False
        if recall:
            mode = "direct"
        elif follow_up:
            mode = state.get("last_mode") or "direct"
        else:
            mode = "knowledge" if should_retrieve_knowledge(question) else "direct"
        if not follow_up and not recall:
            history, summary, topic = [], [], question
        query, method = question, "standalone"
        emit("route", mode)
        if follow_up and mode == "knowledge":
            latest = next((m["content"] for m in reversed(history) if m["role"] == "user"), topic)
            query = f"{topic}\n{latest}\n{question}"
            method = "anchored_fallback"
            if self.settings.knowledge_rewrite_followups:
                emit("status", "正在补全追问中的设备和故障上下文")
                try:
                    rewritten = await self.llm.rewrite_question(question, topic, self.memory.assemble(history, summary))
                    query = f"{topic}\n{rewritten}\n{question}"
                    method = "model"
                except Exception as error:
                    logger.warning("RAG follow-up rewrite fallback (%s)", type(error).__name__)
        return {"history": history, "summary": summary, "memory_metrics": memory_metrics,
                "topic": topic, "follow_up": follow_up, "conversation_recall": recall, "mode": mode,
                "query": query, "rewrite": method, "timings": timed(state, "conversationMs", started)}

    async def retrieve(self, state: QuestionState):
        started = time.perf_counter()
        emit("status", "正在检索故障诊断知识库")
        trace = {}
        if self.retriever is not None and hasattr(self.retriever, "search_with_trace"):
            outcome = await self.retriever.search_with_trace(state["query"])
            results, trace = outcome.results, outcome.metrics
        else:
            results = await self.retriever.search(state["query"]) if self.retriever else self.knowledge.search(state["query"])
        output_tokens = self.settings.knowledge_qwen_max_tokens if self.settings.knowledge_provider == "qwen" else self.settings.glm_max_tokens
        evidence_budget = min(self.settings.knowledge_evidence_budget_tokens, ContextBudget(self.settings, output_tokens).limit // 2)
        selected = []
        for result in results:
            if text_tokens(self.knowledge.format_context([*selected, result])) <= evidence_budget:
                selected.append(result)
        trace["evidenceBudgetDropped"] = len(results) - len(selected)
        results = selected
        emit("sources", sources=[result.source_dict() for result in results])
        return {"results": results, "retrieval_trace": trace, "timings": timed(state, "retrievalMs", started)}

    async def assess(self, state: QuestionState):
        started = time.perf_counter()
        results = state["results"]
        answerable, numbers, missing, reason = False, [], [], "no_results"
        requested = {s.upper() for s in ENTRY_ID_RE.findall(state["question"])}
        available = {r.chunk.entry_id.upper() for r in results if r.chunk.entry_id}
        if not results and state.get("retrieval_trace", {}).get("evidenceBudgetDropped"):
            reason = "evidence_budget_exhausted"
            missing = ["候选片段超过上下文证据预算，请缩小问题范围或调整证据预算。"]
        elif requested - available:
            reason = "requested_entry_missing"
            missing = ["本轮检索未找到问题指定的全部条目，请核对条目编号或补充资料。"]
        elif results:
            emit("status", "正在检查检索证据是否足以回答问题")
            if self.settings.knowledge_evidence_check == "off":
                answerable, numbers, reason = True, list(range(1, len(results) + 1)), "check_disabled"
            else:
                try:
                    question = state["question"]
                    if state["follow_up"]:
                        question = f"已知话题：{state['topic']}\n本轮追问：{question}\n只判断本轮追问所需的证据。"
                    assessment = await self.llm.assess_evidence(
                        question, self.knowledge.format_context(results), len(results),
                    )
                    answerable = assessment.answerable
                    numbers = sorted(set(assessment.evidence_numbers)) if answerable else []
                    missing = assessment.missing_information
                    reason = ("model_partial" if assessment.coverage == "partial" else "model_sufficient") if answerable else "model_insufficient"
                except ContextBudgetExceeded:
                    reason = "evidence_budget_exhausted"
                    missing = ["问题与证据合计超过审核上下文预算，请缩小问题范围或减少检索片段。"]
                except Exception as error:
                    reason = "check_unavailable"
                    logger.warning("RAG evidence check unavailable (%s)", type(error).__name__)
        emit("assessment", data={"answerable": answerable, "reason": reason, "evidenceNumbers": numbers, "missingInformation": missing})
        return {"answerable": answerable, "evidence_numbers": numbers, "missing": missing,
                "assessment_reason": reason, "timings": timed(state, "assessmentMs", started)}

    async def clarify(self, state: QuestionState):
        if state["assessment_reason"] == "check_unavailable":
            answer = "本次证据检查暂未完成，无法确认资料是否足以支持回答，请稍后重试。"
        elif state["assessment_reason"] == "evidence_budget_exhausted":
            answer = "本轮检索资料超过上下文预算，尚不能完成证据审核。" + "；".join(state["missing"])
        else:
            answer = "当前知识库中没有足够信息支持回答。"
            answer += ("还需要：" + "；".join(state["missing"]) if state["missing"] else "请补充设备类型、故障现象、动作阶段和监测数据。")
        emit("delta", answer)
        return {"answer": answer, "first_content_ms": round((time.perf_counter() - state["started"]) * 1000, 2)}

    async def generate(self, state: QuestionState):
        started = time.perf_counter()
        knowledge_mode = state["mode"] == "knowledge"
        emit("status", f"{self.llm.model_label} 正在分析检索结果" if knowledge_mode else f"当前问题无需检索，{self.llm.model_label} 正在直接回答")
        guard = CitationGuard(state["results"], state["evidence_numbers"]) if knowledge_mode else None
        if knowledge_mode:
            # Preserve original source numbers while excluding unapproved evidence.
            labels = citation_labels(state["results"])
            context = "\n\n".join(
                f"[资料{n}] 回答引用标识：[{labels[n-1]}]；来源：{state['results'][n-1].chunk.source_label}\n{state['results'][n-1].chunk.content}"
                for n in state["evidence_numbers"]
            )
            if state["missing"]:
                context = "本轮资料不能支撑以下内容，请在回答中明确说明不足，禁止推测补全：" + "；".join(state["missing"]) + "\n\n" + context
            question = state["question"]
            if state["follow_up"]:
                question = f"已知话题：{state['topic']}\n本轮追问：{question}\n请只回答本轮追问。"
            stream = self.llm.stream_answer(question, context, self.memory.assemble(state["history"], state["summary"]))
        else:
            stream = self.llm.stream_direct_answer(state["question"], self.memory.assemble(
                state["history"], state["summary"], state.get("recall_records")))
        answer, first = "", None
        async with aclosing(stream):
            async for delta in stream:
                output = guard.feed(delta) if guard else delta
                if output:
                    first = first if first is not None else round((time.perf_counter() - state["started"]) * 1000, 2)
                    answer += output
                    emit("delta", output)
        tail = guard.feed("", final=True) if guard else ""
        if tail:
            first = first if first is not None else round((time.perf_counter() - state["started"]) * 1000, 2)
            answer += tail
            emit("delta", tail)
        if not answer.strip():
            raise RuntimeError("Empty model answer")
        return {"answer": answer, "first_content_ms": first,
                "context_metrics": self.llm.context_metrics() if hasattr(self.llm, "context_metrics") else {},
                "validation": guard.report() if guard else {"passed": True, "scope": "not_applicable"},
                "timings": timed(state, "generationMs", started)}

    async def validate(self, state: QuestionState):
        report = state.get("validation", {"passed": True, "scope": "no_generated_answer"})
        answer = state["answer"]
        if state.get("assessment_reason") == "model_partial":
            scope = "\n\n资料范围：当前知识库对以下内容仍缺少足够依据：" + "；".join(state["missing"]) + "。"
            answer += scope
            emit("delta", scope)
        if report.get("missingCitation"):
            warning = "\n\n引用检查：本回答缺少可核验的资料引用，请勿将其作为已核实结论。"
            answer += warning
            emit("delta", warning)
        elif report.get("invalidReferences"):
            warning = "\n\n引用检查：部分引用未通过校验，已标记；相关结论仍需核实。"
            answer += warning
            emit("delta", warning)
        emit("validation", data=report)
        return {"answer": answer, "validation": report}

    async def finish(self, state: QuestionState):
        history = state["history"] + [{"role": "user", "content": state["question"]}, {"role": "assistant", "content": state["answer"]}]
        memory_started = time.perf_counter()
        history, summary, memory_metrics = await self.memory.compact(history, state["summary"])
        metrics = {**state["timings"], **state.get("retrieval_trace", {}), **state.get("context_metrics", {}),
                   **state.get("memory_metrics", {}), **memory_metrics,
                   "memorySummaryUpdated": memory_metrics["memorySummaryUpdated"] or state["memory_metrics"]["memorySummaryUpdated"],
                   "memorySummaryMethod": (memory_metrics["memorySummaryMethod"] if memory_metrics["memorySummaryUpdated"]
                                           else state["memory_metrics"]["memorySummaryMethod"]),
                   "memoryMs": round(state["memory_metrics"]["memoryMs"] + (time.perf_counter() - memory_started) * 1000, 2),
                   "memoryRecallMode": state.get("recall_records", {}).get("mode", "none"),
                   "memoryRecallCount": len(state.get("recall_records", {}).get("records", [])),
                   "memoryRecallMs": state.get("recall_ms", 0),
                   "firstContentMs": state.get("first_content_ms"),
                   "totalMs": round((time.perf_counter() - state["started"]) * 1000, 2),
                   "followUp": state["follow_up"], "conversationRecall": state["conversation_recall"], "rewrite": state["rewrite"],
                   "answerMode": state["mode"], "assessment": state.get("assessment_reason", "not_applicable"),
                   "indexVersions": sorted({r.kb_version for r in state.get("results", []) if r.kb_version}),
                   "citationPassed": state["validation"]["passed"]}
        emit("metrics", data=metrics)
        logger.info("RAG workflow timings: %s", metrics)
        # Recalling the conversation must not replace the diagnostic topic/route.
        last_mode = state.get("last_mode") if state["conversation_recall"] else state["mode"]
        emit("_complete", data={"history": history, "summary": summary, "topic": state["topic"], "last_mode": last_mode})
        return {}
