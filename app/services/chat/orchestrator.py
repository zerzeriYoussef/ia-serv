"""
Chat Orchestrator — drives one full conversational turn.

Flow per turn:
  1. Load analysis context + staleness check
  2. Build conversation history (last N turns + rolling summary)
  3. Embed query → retrieve semantic chunks from Chroma
  4. Call Gemini (JSON mode) → OrchestratorPlan
  5. Yield IntentEvent
  6. Yield RetrievalEvent
  7. For each tool call → AnalysisAgent.run() → yield ToolStart + ToolResult events
  8. (optional) VizAgent.build_spec() → yield ChartEvent
  9. Stream Gemini narrative → yield TokenEvent × N
 10. Yield DoneEvent

Yields raw SSE frame strings. Writes final metadata to `result_out` dict so
the calling route can persist the message without re-parsing SSE events.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.chat_schema import (
    AllowedOp,
    ChartEvent,
    ChartSpec,
    DoneEvent,
    ErrorEvent,
    Intent,
    IntentEvent,
    OrchestratorPlan,
    RetrievalEvent,
    SseEventType,
    ToolArgs,
    ToolResult,
    ToolResultEvent,
    ToolStartEvent,
    TokenEvent,
)
from app.core.config import settings
from app.models.conversation import ChatMessage, MessageRole
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.dataset_repository import DatasetRepository
from app.services.chat.analysis_agent import AnalysisAgent, AnalysisAgentError
from app.services.chat.viz_agent import VizAgent
from app.services.data.parser_service import ParserService
from app.services.rag.dataset_indexer import retrieve_chunks
from app.services.rag.gemini_client import generate_json_response, stream_text_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """\
You are a data analytics orchestrator. Your ONLY job is to produce a structured JSON plan.

Output a single JSON object with these EXACT keys:
{
  "intent": "<one of: retrieve_only | analyze | visualize | clarify | refuse_unsafe>",
  "tool_calls": [ ... ],
  "needs_chart": <true|false>,
  "clarification_question": "<string or null>",
  "refuse_reason": "<string or null>"
}

RULES:
1. intent must be exactly one of: retrieve_only, analyze, visualize, clarify, refuse_unsafe
2. refuse_unsafe if the question asks to delete, modify, execute code, or access PII
3. tool_calls items MUST use these exact field names: "op", "group_col", "agg_col", "agg_func",
   "filter_col", "filter_val", "top_n" (no synonyms: do not use "group", "agg", "column", "func")
4. "op" must be one of: groupby_agg, value_counts, describe, correlation, filter_agg
5. agg_func must be one of: sum, mean, median, count, nunique, min, max, std
6. Column names MUST come from DATASET_CONTEXT.columns — never invent names
7. For a single-column statistic (mean, min, max, std, quartiles) or "first numeric column",
   use op "describe" with "agg_col" set to that column — NOT groupby_agg without a grouping column
8. groupby_agg REQUIRES both group_col and agg_col (breakdown by category + metric)
9. value_counts uses "group_col" as the categorical column to count
10. For simple factual questions answerable from row_count/columns only, use retrieve_only with []
11. needs_chart true only when a chart genuinely helps
12. RECENT_CONVERSATION (in the user prompt) is the same thread, oldest→newest.
    Use it to interpret short replies, pronouns, and follow-ups to your prior clarifications.
    If the user is elaborating on an earlier question (e.g. about a named column), prefer
    analyze or retrieve_only with concrete tool_calls — do NOT use intent=clarify again
    unless the thread still lacks any identifiable column or metric.
13. If ambiguous AND the thread has no usable topic yet, intent=clarify and set clarification_question

EXAMPLES (valid tool_calls shapes):
{"op":"describe","agg_col":"revenue"}
{"op":"groupby_agg","group_col":"region","agg_col":"revenue","agg_func":"sum"}
{"op":"value_counts","group_col":"status"}

Output ONLY the JSON object, no markdown, no explanation.
"""

NARRATOR_SYSTEM = """\
You are a friendly data analyst explaining results to a business user.
Be concise, clear, and specific. Use plain language.
Do NOT mention internal implementation details, column names as variables, or code.
If citing retrieved facts, reference them naturally (not as [chunk_id] codes).
If tool results are available, interpret them directly — do not say "based on the data".
Just give the direct answer with numbers and insights.
End with a brief caveat if relevant (e.g., correlation ≠ causation, sample size).
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sse(event_type: SseEventType, data: Any) -> str:
    """Format one SSE frame."""
    if isinstance(data, dict):
        payload = json.dumps(data)
    elif hasattr(data, "model_dump_json"):
        payload = data.model_dump_json()
    else:
        payload = json.dumps(str(data))
    return f"event: {event_type.value}\ndata: {payload}\n\n"


def _build_history_contents(
    messages: List[ChatMessage],
    rolling_summary: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Convert DB messages → Gemini contents list (chronological).
    messages are expected in CHRONOLOGICAL order (oldest first).
    """
    contents: List[Dict[str, Any]] = []
    if rolling_summary:
        contents.append({
            "role": "user",
            "parts": [{"text": f"[CONVERSATION SUMMARY SO FAR]\n{rolling_summary}"}],
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Understood. I have the conversation summary."}],
        })

    for msg in messages:
        role = "user" if msg.role.value == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg.content}]})
    return contents


def _format_conversation_for_planner(
    messages: List[ChatMessage],
    *,
    max_messages: int,
) -> str:
    """
    Compact transcript tail for the orchestrator plan call.

    The planner used to see only USER_QUESTION; short follow-ups then looked
    context-free and repeatedly triggered intent=clarify.
    """
    if not messages:
        return "(no prior messages in this conversation — first turn.)"
    tail = messages[-max_messages:]
    lines: List[str] = []
    for m in tail:
        role = m.role.value if isinstance(m.role, MessageRole) else str(m.role)
        text = (m.content or "").strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _retrieval_query_with_history(
    user_message: str,
    history_messages: List[ChatMessage],
    *,
    short_len: int = 100,
) -> str:
    """
    For very short user messages, prepend recent transcript so embeddings match
    the ongoing topic (e.g. Total_Goals + 'means disparate numbers').
    """
    msg = (user_message or "").strip()
    if len(msg) >= short_len or not history_messages:
        return msg
    tail = history_messages[-8:]
    parts: List[str] = []
    for m in tail:
        role = m.role.value if isinstance(m.role, MessageRole) else str(m.role)
        bit = (m.content or "").strip().replace("\n", " ")
        if len(bit) > 320:
            bit = bit[:320] + "…"
        parts.append(f"{role}: {bit}")
    parts.append(f"user: {msg}")
    return "\n".join(parts)


def _build_retrieval_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No additional context retrieved."
    parts = []
    for c in chunks:
        parts.append(f"[{c['chunk_id']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _trim_tool_result_preview(result: ToolResult, max_chars: int = 500) -> str:
    raw = json.dumps(result.payload, default=str)
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "…"
    return raw


def _make_analysis_context(analysis_row: Any, dataset: Any) -> Dict[str, Any]:
    return {
        "dataset_id": dataset.id,
        "filename": dataset.original_filename,
        "row_count": dataset.row_count,
        "columns": dataset.columns or [],
        "column_types": dataset.column_types or {},
        "primary_metric": analysis_row.primary_metric,
        "metrics": analysis_row.metrics or [],
        "dimensions": analysis_row.dimensions or [],
        "temporal": analysis_row.temporal or [],
        "geographic": (analysis_row.geographic or []),
        "identifiers": (analysis_row.identifiers or []),
    }


def _normalize_tool_call_dict(tc: dict) -> dict:
    """
    Map common model mistakes to ToolArgs field names before validation.
    """
    out = dict(tc)
    if "op" not in out and isinstance(out.get("operation"), str):
        out["op"] = out.pop("operation")

    # Copy known synonyms into canonical keys (only if target not already set)
    synonyms = [
        ("group", "group_col"),
        ("group_by", "group_col"),
        ("groupCol", "group_col"),
        ("by", "group_col"),
        ("agg", "agg_col"),
        ("value_col", "agg_col"),
        ("metric", "agg_col"),
        ("func", "agg_func"),
        ("aggregation", "agg_func"),
    ]
    for src, dst in synonyms:
        if src in out and out.get(dst) in (None, "") and out[src] not in (None, ""):
            out[dst] = out.pop(src)

    # "column" → agg_col when no other target (typical for describe / single-metric)
    col = out.get("column")
    if (
        col not in (None, "")
        and out.get("agg_col") in (None, "")
        and out.get("group_col") in (None, "")
    ):
        out["agg_col"] = out.pop("column")

    return out


def _repair_tool_calls(
    tool_calls: List[ToolArgs], analysis_ctx: Dict[str, Any]
) -> List[ToolArgs]:
    """
    Fix plans where the model chose groupby_agg but omitted required columns —
    common for questions that need describe / global stats instead.
    """
    columns = set(analysis_ctx.get("columns") or [])
    pm = analysis_ctx.get("primary_metric")
    repaired: List[ToolArgs] = []

    for tc in tool_calls:
        if tc.op != AllowedOp.groupby_agg:
            repaired.append(tc)
            continue
        if tc.group_col and tc.agg_col:
            repaired.append(tc)
            continue

        # Prefer an explicit partial column if valid
        fallback_col = None
        if tc.agg_col and tc.agg_col in columns:
            fallback_col = tc.agg_col
        elif tc.group_col and tc.group_col in columns:
            fallback_col = tc.group_col
        elif pm and pm in columns:
            fallback_col = pm
        else:
            ct = analysis_ctx.get("column_types") or {}
            for c in analysis_ctx.get("columns") or []:
                if c not in columns:
                    continue
                t = str(ct.get(c, "")).lower()
                if any(x in t for x in ("int", "float", "double", "number", "decimal")):
                    fallback_col = c
                    break

        if fallback_col:
            logger.info(
                "Repairing invalid groupby_agg → describe(agg_col=%s)", fallback_col
            )
            repaired.append(
                ToolArgs(
                    op=AllowedOp.describe,
                    agg_col=fallback_col,
                    agg_func=tc.agg_func,
                )
            )
        else:
            logger.warning("Dropping invalid groupby_agg (no usable column): %s", tc)

    return repaired


def _parse_plan_robust(raw: Any) -> OrchestratorPlan:
    """
    Validate raw LLM dict → OrchestratorPlan with generous fallbacks.
    Logs warnings rather than crashing on minor schema violations.
    """
    if not isinstance(raw, dict):
        logger.warning("Plan response is not a dict: %s", type(raw))
        return OrchestratorPlan(intent=Intent.retrieve_only, tool_calls=[], needs_chart=False)

    # Coerce intent to valid enum value
    raw_intent = str(raw.get("intent", "retrieve_only")).strip().lower()
    valid_intents = {e.value for e in Intent}
    if raw_intent not in valid_intents:
        logger.warning("Unknown intent '%s', falling back to retrieve_only", raw_intent)
        raw_intent = "retrieve_only"

    # Parse tool_calls defensively
    raw_tools = raw.get("tool_calls") or []
    tool_calls = []
    for tc in raw_tools:
        if not isinstance(tc, dict):
            continue
        try:
            tool_calls.append(ToolArgs.model_validate(_normalize_tool_call_dict(tc)))
        except Exception as e:
            logger.warning("Skipping invalid tool_call %s: %s", tc, e)

    return OrchestratorPlan(
        intent=Intent(raw_intent),
        tool_calls=tool_calls,
        needs_chart=bool(raw.get("needs_chart", False)),
        clarification_question=raw.get("clarification_question"),
        refuse_reason=raw.get("refuse_reason"),
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

async def run_chat_turn(
    *,
    db: AsyncSession,
    dataset_id: int,
    conversation_id: int,
    user_message: str,
    # history in CHRONOLOGICAL order (oldest first), NOT including current message
    history_messages: List[ChatMessage],
    rolling_summary: Optional[str] = None,
    # Mutable dict — caller passes an empty dict and reads it after the generator ends
    result_out: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Drive one full chat turn. Yields raw SSE frame strings.
    Writes final turn metadata into `result_out` when done.
    """
    if result_out is None:
        result_out = {}

    t0 = time.monotonic()

    # ── 1. Load analysis context ───────────────────────────────────────
    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)
    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset or not analysis_row:
        msg = "Dataset or analysis not found. Run POST /datasets/{id}/analyze first."
        result_out.update({"full_answer": msg, "intent": "error", "chunk_ids": [], "latency_ms": 0})
        yield _sse(SseEventType.error, ErrorEvent(error=msg).model_dump())
        return

    analysis_ctx = _make_analysis_context(analysis_row, dataset)
    logger.info(
        "chat_turn: dataset=%s conv=%s user_msg='%s...'",
        dataset_id, conversation_id, user_message[:60],
    )

    planner_history_cap = max(4, settings.CHAT_PLANNER_HISTORY_MESSAGES)
    transcript_for_planner = _format_conversation_for_planner(
        history_messages,
        max_messages=planner_history_cap,
    )
    retrieval_query = _retrieval_query_with_history(user_message, history_messages)

    # ── 2. Retrieve semantic chunks ────────────────────────────────────
    try:
        chunks = await retrieve_chunks(
            dataset_id, retrieval_query, top_k=settings.CHAT_RETRIEVAL_TOP_K
        )
    except Exception as exc:
        logger.warning("Retrieval failed (falling back to empty): %s", exc)
        chunks = []

    chunk_ids = [c["chunk_id"] for c in chunks]
    retrieval_ctx = _build_retrieval_context(chunks)
    logger.info("chat_turn: retrieved %d chunks", len(chunks))

    # ── 3. Build OrchestratorPlan via Gemini ──────────────────────────
    plan_user_prompt = f"""\
DATASET_CONTEXT (use ONLY these column names in tool_calls):
{json.dumps(analysis_ctx, default=str, indent=2)}

RETRIEVED_CONTEXT:
{retrieval_ctx}

RECENT_CONVERSATION (same thread; oldest message first in this block):
{transcript_for_planner}

USER_QUESTION (current user message — interpret using RECENT_CONVERSATION when it is short or a follow-up):
{user_message}

Produce a JSON plan with keys: intent, tool_calls, needs_chart, clarification_question, refuse_reason.
"""
    try:
        raw_plan = await generate_json_response(ORCHESTRATOR_SYSTEM, plan_user_prompt, timeout_s=45.0)
        logger.info("chat_turn: raw plan = %s", json.dumps(raw_plan, default=str)[:300])
        plan = _parse_plan_robust(raw_plan)
    except Exception as exc:
        logger.error("Orchestrator planning failed: %s", exc)
        # Fall back to retrieve_only so the user still gets an answer
        plan = OrchestratorPlan(intent=Intent.retrieve_only, tool_calls=[], needs_chart=False)

    if plan.tool_calls and plan.intent in (Intent.analyze, Intent.visualize):
        fixed_tools = _repair_tool_calls(plan.tool_calls, analysis_ctx)
        if fixed_tools != plan.tool_calls:
            plan = plan.model_copy(update={"tool_calls": fixed_tools})

    logger.info("chat_turn: intent=%s tools=%d chart=%s", plan.intent, len(plan.tool_calls), plan.needs_chart)

    # ── 4. Emit IntentEvent ────────────────────────────────────────────
    tool_summary = ", ".join(tc.op for tc in plan.tool_calls) if plan.tool_calls else "none"
    yield _sse(
        SseEventType.intent,
        IntentEvent(
            intent=plan.intent.value,
            plan_summary=f"Tools: {tool_summary} | chart: {plan.needs_chart}",
        ).model_dump(),
    )

    # Early exits
    if plan.intent == Intent.refuse_unsafe:
        msg = plan.refuse_reason or "Request refused for safety reasons."
        result_out.update({"full_answer": msg, "intent": plan.intent.value, "chunk_ids": chunk_ids,
                           "latency_ms": int((time.monotonic() - t0) * 1000)})
        yield _sse(SseEventType.token, TokenEvent(delta=msg).model_dump())
        yield _sse(SseEventType.done, DoneEvent(conversation_id=conversation_id, message_id=0,
                                                 latency_ms=result_out["latency_ms"]).model_dump())
        return

    if plan.intent == Intent.clarify:
        msg = plan.clarification_question or "Could you clarify your question?"
        result_out.update({"full_answer": msg, "intent": plan.intent.value, "chunk_ids": chunk_ids,
                           "latency_ms": int((time.monotonic() - t0) * 1000)})
        yield _sse(SseEventType.token, TokenEvent(delta=msg).model_dump())
        yield _sse(SseEventType.done, DoneEvent(conversation_id=conversation_id, message_id=0,
                                                 latency_ms=result_out["latency_ms"]).model_dump())
        return

    # ── 5. Emit RetrievalEvent ─────────────────────────────────────────
    yield _sse(SseEventType.retrieval, RetrievalEvent(chunk_ids=chunk_ids).model_dump())

    # ── 6. Execute tool calls ──────────────────────────────────────────
    tool_results: List[Tuple[ToolArgs, ToolResult]] = []
    chart_spec: Optional[ChartSpec] = None

    if plan.tool_calls and plan.intent in (Intent.analyze, Intent.visualize):
        try:
            df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
        except Exception as exc:
            logger.error("Failed to parse dataset file: %s", exc)
            df = None

        if df is not None:
            agent = AnalysisAgent(df)
            for tool_args in plan.tool_calls:
                yield _sse(
                    SseEventType.tool_start,
                    ToolStartEvent(
                        tool=tool_args.op.value,
                        args_summary=(
                            f"{tool_args.op.value}("
                            f"group={tool_args.group_col}, "
                            f"agg={tool_args.agg_col}, "
                            f"func={tool_args.agg_func})"
                        ),
                    ).model_dump(),
                )
                try:
                    result = await agent.run(tool_args)
                    tool_results.append((tool_args, result))
                    yield _sse(
                        SseEventType.tool_result,
                        ToolResultEvent(preview=_trim_tool_result_preview(result)).model_dump(),
                    )
                    logger.info("chat_turn: tool %s succeeded", tool_args.op.value)
                except Exception as exc:
                    logger.warning("Tool %s failed: %s", tool_args.op.value, exc)
                    yield _sse(
                        SseEventType.tool_result,
                        ToolResultEvent(preview=f"Error: {exc!s}").model_dump(),
                    )

            # ── 7. Visualization ───────────────────────────────────────────
            if plan.needs_chart and tool_results:
                _, last_result = tool_results[-1]
                viz = VizAgent()
                try:
                    chart_spec = viz.build_spec(
                        last_result,
                        intent_hint=user_message,
                        col_types=dataset.column_types or {},
                    )
                    if chart_spec:
                        yield _sse(
                            SseEventType.chart,
                            ChartEvent(chart_spec=chart_spec.model_dump()).model_dump(),
                        )
                except Exception as exc:
                    logger.warning("VizAgent failed (non-fatal): %s", exc)

    # ── 8. Build narrator prompt & stream answer ───────────────────────
    tool_results_text = ""
    if tool_results:
        parts = []
        for args, res in tool_results:
            parts.append(
                f"Op: {args.op.value}\n"
                f"Result: {json.dumps(res.payload, default=str)[:2000]}"
            )
        tool_results_text = "\n\n".join(parts)
    else:
        tool_results_text = "No tool calls were executed."

    chart_note = (
        f"\nA {chart_spec.chart_type.value} chart titled '{chart_spec.title}' was generated."
        if chart_spec else ""
    )

    history_contents = _build_history_contents(history_messages, rolling_summary)

    narrator_prompt = f"""\
DATASET INFO:
- File: {dataset.original_filename}
- Rows: {dataset.row_count}
- Primary metric: {analysis_row.primary_metric}

RETRIEVED_CONTEXT:
{retrieval_ctx}

TOOL_RESULTS:
{tool_results_text}
{chart_note}

USER_QUESTION: {user_message}

Answer directly and specifically. If the tool results contain the answer, give the exact number/values.
"""

    full_answer_parts: List[str] = []
    try:
        async for delta in stream_text_response(
            NARRATOR_SYSTEM,
            narrator_prompt,
            history=history_contents,
            timeout_s=60.0,
        ):
            full_answer_parts.append(delta)
            yield _sse(SseEventType.token, TokenEvent(delta=delta).model_dump())
    except Exception as exc:
        logger.error("Streaming narrator failed: %s", exc)
        # Fall back to non-streaming narrator
        try:
            from app.services.rag.gemini_client import generate_json_response as _gen
            # Use plain generate for fallback — won't stream but will give an answer
            import httpx
            key = settings.GEMINI_API_KEY
            mdl = settings.GEMINI_MODEL
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent"
            body = {
                "systemInstruction": {"parts": [{"text": NARRATOR_SYSTEM}]},
                "contents": history_contents + [{"role": "user", "parts": [{"text": narrator_prompt}]}],
                "generationConfig": {"temperature": 0.4},
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, params={"key": key}, json=body)
                resp.raise_for_status()
            data = resp.json()
            cands = data.get("candidates") or []
            parts_list = (cands[0].get("content") or {}).get("parts") or [] if cands else []
            fallback_text = parts_list[0].get("text", "") if parts_list else "I could not generate a response."
            full_answer_parts.append(fallback_text)
            yield _sse(SseEventType.token, TokenEvent(delta=fallback_text).model_dump())
        except Exception as exc2:
            logger.error("Fallback narrator also failed: %s", exc2)
            fallback = "I could not generate a response due to an API error."
            full_answer_parts.append(fallback)
            yield _sse(SseEventType.token, TokenEvent(delta=fallback).model_dump())

    latency_ms = int((time.monotonic() - t0) * 1000)
    full_answer = "".join(full_answer_parts)

    # ── 9. Populate result_out for the route layer ─────────────────────
    result_out.update({
        "full_answer": full_answer,
        "intent": plan.intent.value,
        "chunk_ids": chunk_ids,
        "chart_spec": chart_spec.model_dump() if chart_spec else None,
        "latency_ms": latency_ms,
        "tool_calls": [tc.model_dump() for tc in plan.tool_calls],
    })

    logger.info(
        "chat_turn: done conv=%s latency=%dms answer_len=%d",
        conversation_id, latency_ms, len(full_answer),
    )

    yield _sse(
        SseEventType.done,
        DoneEvent(
            conversation_id=conversation_id,
            message_id=0,  # caller updates with persisted message id
            latency_ms=latency_ms,
        ).model_dump(),
    )
