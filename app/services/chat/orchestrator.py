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

import math

import re

import time

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple



from sqlalchemy.ext.asyncio import AsyncSession



from app.api.v1.schemas.chat_schema import (

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











ORCHESTRATOR_SYSTEM = """\
You are a data analytics orchestrator. Your ONLY job is to produce a structured JSON plan.

Output a single JSON object with these EXACT keys:
{
  "intent": "<one of: retrieve_only | analyze | visualize | clarify | generate_report | refuse_unsafe>",
  "tool_calls": [ ... ],
  "needs_chart": <true|false>,
  "clarification_question": "<string or null>",
  "refuse_reason": "<string or null>"
}

RULES:
1. intent must be exactly one of: retrieve_only, analyze, visualize, clarify, generate_report, refuse_unsafe
2. refuse_unsafe ONLY if the question explicitly asks to delete, drop, overwrite data, or access
   credentials/PII. Analytics questions (even complex ones) must NEVER be refused.
3. Each tool_call has exactly two fields:
   - "code_expr": a single valid Python expression using `df` (the dataset DataFrame) and `pd` (pandas)
   - "label": a short human-readable description of what it computes
4. The expression must be a single eval()-able Python expression — no statements, no assignments,
   no imports, no loops. It must return a scalar, pd.Series, or pd.DataFrame.
5. Column names MUST come from DATASET_CONTEXT.columns — never invent names.
6. When checking categories and values, always use EXACT values and case from `sample_values` in DATASET_CONTEXT. For example, if sample_values says `high`, don't use `High`.
7. You MAY chain any pandas operations: filter, groupby, agg, pivot, sort, corr, etc.
8. For multi-condition questions, use boolean masks:
   df[(df['col1']=='val1') & (df['col2']=='val2')].groupby('cat')['metric'].mean()
9. For top-N questions, chain .sort_values().head(N)
10. For comparisons across groups, use groupby + agg
11. needs_chart=true AND add a tool_call whenever the answer compares
    multiple categories, shows a trend, or involves a distribution —
    even if intent=retrieve_only. A chart ALWAYS requires at least one
    tool_call to have data to plot. If you set needs_chart=true, you
    MUST also add a tool_call that computes the data for the chart.
12. retrieve_only with [] when answerable from row_count/columns alone
13. RECENT_CONVERSATION: use it to resolve follow-ups, pronouns, short replies.
    Prefer analyze with concrete code_expr over intent=clarify unless the question
    truly cannot be resolved from context.
14. If still ambiguous after checking conversation context, intent=clarify.
15. IMPLICIT/SPECULATIVE ANALYSIS: Questions like "I wonder if X affects Y", "Could X improve Y",
    "Is there a link between X and Y", or "What if..." ARE analysis questions. Do NOT use
    retrieve_only. You MUST use intent=analyze and provide a tool_call (e.g. corr(),
    groupby().mean()) to test the hypothesis against the data.
16. GENERATE REPORT: If the user explicitly asks for a "report", "comprehensive summary", "executive summary", or "generate a report", you MUST use intent=generate_report and leave tool_calls empty.

EXAMPLES (valid tool_calls shapes):
{"code_expr": "df['revenue'].describe()", "label": "Revenue stats"}
{"code_expr": "df.groupby('region')['revenue'].sum().sort_values(ascending=False)", "label": "Revenue by region"}
{"code_expr": "df[(df['continent']=='europe') & (df['income_level']=='high')].groupby('drink_preference')['monthly_spend'].mean()", "label": "Monthly spend by drink, Europe+High income"}
{"code_expr": "df['status'].value_counts()", "label": "Status distribution"}
{"code_expr": "df[['col_a','col_b']].corr()", "label": "Correlation"}

Output ONLY the JSON object, no markdown, no explanation.
"""



NARRATOR_SYSTEM = """\
You are a friendly data analyst explaining results to a business user.
Be concise, clear, and specific. Use plain language.
Do NOT mention internal implementation details, column names as variables, or code.
If citing retrieved facts, reference them naturally (not as [chunk_id] codes).
If tool results are available, interpret them directly — do not say "based on the data".
If the prompt says a chart WAS generated, describe the insights briefly — the UI renders it automatically below your message.
NEVER say you cannot generate, draw, or display charts/visuals — the application renders charts for the user.
If the prompt says NO chart was generated, do NOT claim to show a chart; give the numeric answer from tool results instead.
Just give the direct answer with numbers and insights.
End with a brief caveat if relevant (e.g., correlation ≠ causation, sample size).
"""











def _json_safe(value: Any) -> Any:

    """Return a strict-JSON-safe copy of streamed payload data."""

    if isinstance(value, dict):

        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):

        return [_json_safe(v) for v in value]

    if isinstance(value, float):

        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:

        return value

    if hasattr(value, "item"):

        try:

            return _json_safe(value.item())

        except Exception:

            pass

    if hasattr(value, "isoformat"):

        try:

            return value.isoformat()

        except Exception:

            pass

    return str(value)





def _sse(event_type: SseEventType, data: Any) -> str:

    """Format one SSE frame."""

    if hasattr(data, "model_dump"):

        payload_data = data.model_dump()

    elif isinstance(data, dict):

        payload_data = data

    else:

        payload_data = data

    payload = json.dumps(_json_safe(payload_data), allow_nan=False)

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

        text = (msg.content or "").strip()

        if not text:

            text = "(empty message)"



        if contents and contents[-1]["role"] == role:

            contents[-1]["parts"][0]["text"] += f"\n\n{text}"

        else:

            contents.append({"role": role, "parts": [{"text": text}]})





    if contents and contents[-1]["role"] == "user":

        contents.append({"role": "model", "parts": [{"text": "(No response recorded)"}]})



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





def _fallback_chart_narrative(chart_spec: Optional[ChartSpec]) -> str:

    """Small deterministic explanation when the narrator model returns no usable text."""

    if not chart_spec:

        return "I prepared the result from the available data."



    data = chart_spec.data or {}

    title = chart_spec.title or "chart"

    points: List[Tuple[str, float]] = []



    axis_x = data.get("axis_x") or []

    axis_y = data.get("axis_y") or []

    if isinstance(axis_x, list) and isinstance(axis_y, list):

        for label, raw_value in zip(axis_x, axis_y):

            try:

                value = float(raw_value)

            except (TypeError, ValueError):

                continue

            if math.isfinite(value):

                points.append((str(label), value))



    if not points and isinstance(data.get("series"), list):

        for series in data["series"]:

            if not isinstance(series, dict):

                continue

            values = series.get("values") or []

            if not isinstance(values, list) or not values:

                continue

            finite_values: List[float] = []

            for raw_value in values:

                try:

                    value = float(raw_value)

                except (TypeError, ValueError):

                    continue

                if math.isfinite(value):

                    finite_values.append(value)

            if finite_values:

                points.append((str(series.get("name") or "series"), max(finite_values)))



    if points:

        top = sorted(points, key=lambda item: item[1], reverse=True)[:3]

        top_text = ", ".join(f"{label}: {value:,.2f}" for label, value in top)

        return f"Here is the {chart_spec.chart_type.value} chart for {title}. The strongest values are {top_text}."



    return f"Here is the {chart_spec.chart_type.value} chart for {title}."





def _make_analysis_context(

    analysis_row: Any,

    dataset: Any,

    sample_values: Optional[Dict[str, List[Any]]] = None,

) -> Dict[str, Any]:

    ctx: Dict[str, Any] = {

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

    if sample_values:

        ctx["sample_values"] = sample_values

    return ctx





def _extract_sample_values(

    df: Any,

    max_cols: int = 150,

    max_vals: int = 15,

) -> Dict[str, List[Any]]:

    """
    For each low-cardinality categorical column, return up to `max_vals`
    unique values (preserving actual casing from the data).
    """

    import pandas as pd

    samples: Dict[str, List[Any]] = {}

    for col in df.columns[:max_cols]:

        if df[col].dtype == object or str(df[col].dtype) in ("category", "string"):

            uniq = df[col].dropna().unique()

            if 1 < len(uniq) <= 50:

                samples[col] = [str(v) for v in uniq[:max_vals]]

    return samples







def _parse_plan_robust(raw: Any) -> OrchestratorPlan:

    """
    Validate raw LLM dict → OrchestratorPlan with generous fallbacks.
    Now expects tool_calls with {code_expr, label} shapes.
    """

    if not isinstance(raw, dict):

        logger.warning("Plan response is not a dict: %s", type(raw))

        return OrchestratorPlan(intent=Intent.retrieve_only, tool_calls=[], needs_chart=False)



    raw_intent = str(raw.get("intent", "retrieve_only")).strip().lower()

    valid_intents = {e.value for e in Intent}

    if raw_intent not in valid_intents:

        logger.warning("Unknown intent '%s', falling back to retrieve_only", raw_intent)

        raw_intent = "retrieve_only"



    raw_tools = raw.get("tool_calls") or []

    tool_calls = []

    for tc in raw_tools:

        if not isinstance(tc, dict):

            continue

        code_expr = tc.get("code_expr") or ""

        if not code_expr.strip():

            logger.warning("Skipping tool_call with empty code_expr: %s", tc)

            continue

        try:

            tool_calls.append(ToolArgs(

                code_expr=code_expr.strip(),

                label=tc.get("label", ""),

            ))

        except Exception as e:

            logger.warning("Skipping invalid tool_call %s: %s", tc, e)



    return OrchestratorPlan(

        intent=Intent(raw_intent),

        tool_calls=tool_calls,

        needs_chart=bool(raw.get("needs_chart", False)),

        clarification_question=raw.get("clarification_question"),

        refuse_reason=raw.get("refuse_reason"),

    )





_CHART_REQUEST_RE = re.compile(

    r"\b("

    r"chart|charts|graph|graphs|graphique|graphiques|graphe|graphes|"

    r"visuali[sz]e?|visuali[sz]ation|plot|plots|draw|diagram|figure|"

    r"affich(?:e|er).*graph|montre.*graph|show.*chart|generate.*graph"

    r")\b",

    re.IGNORECASE,

)



_COMPARE_CHART_RE = re.compile(

    r"\b(difference|compare|comparison|between|vs\.?|versus|by month|monthly|trend|over time)\b",

    re.IGNORECASE,

)





def _last_substantive_user_question(

    user_message: str,

    history_messages: List[ChatMessage],

) -> str:

    """For chart-only follow-ups, recover the prior data question from the thread."""

    if not _CHART_REQUEST_RE.search(user_message):

        return user_message

    for msg in reversed(history_messages):

        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

        if role_str != "user":

            continue

        text = (msg.content or "").strip()

        if len(text) > 15 and not _CHART_REQUEST_RE.search(text):

            return text

    return user_message





async def _replan_tools_for_chart(

    analysis_ctx: Dict[str, Any],

    retrieval_ctx: str,

    transcript: str,

    data_question: str,

) -> List[ToolArgs]:

    """Second planner pass when the user wants a chart but the first plan has no tool_calls."""

    replan_prompt = f"""\
DATASET_CONTEXT:
{json.dumps(analysis_ctx, default=str, indent=2)}

RETRIEVED_CONTEXT:
{retrieval_ctx}

RECENT_CONVERSATION:
{transcript}

DATA_QUESTION (build tool_calls to answer this with plottable aggregates):
{data_question}

Output JSON with needs_chart=true and at least one tool_call (code_expr + label).
Use only column names from DATASET_CONTEXT.columns.
Prefer groupby + agg that returns a Series or DataFrame suitable for a line/bar chart.
"""

    try:

        raw = await generate_json_response(ORCHESTRATOR_SYSTEM, replan_prompt, timeout_s=35.0)

        replan = _parse_plan_robust(raw)

        if replan.tool_calls:

            logger.info("chat_turn: chart replan produced %d tool_calls", len(replan.tool_calls))

            return replan.tool_calls

    except Exception as exc:

        logger.warning("chat_turn: chart replan failed: %s", exc)

    return []













async def run_chat_turn(

    *,

    db: AsyncSession,

    dataset_id: int,

    conversation_id: int,

    user_message: str,



    history_messages: List[ChatMessage],

    rolling_summary: Optional[str] = None,

    report_context: Optional[str] = None,

    active_report_section: Optional[str] = None,



    result_out: Optional[Dict[str, Any]] = None,

) -> AsyncGenerator[str, None]:

    """
    Drive one full chat turn. Yields raw SSE frame strings.
    Writes final turn metadata into `result_out` when done.
    """

    if result_out is None:

        result_out = {}



    t0 = time.monotonic()





    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)

    dataset = await DatasetRepository.get_by_id(db, dataset_id)



    if not dataset or not analysis_row:

        msg = "Dataset or analysis not found. Run POST /datasets/{id}/analyze first."

        result_out.update({"full_answer": msg, "intent": "error", "chunk_ids": [], "latency_ms": 0})

        yield _sse(SseEventType.error, ErrorEvent(error=msg).model_dump())

        return





    try:

        df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)

    except Exception as exc:

        logger.error("Failed to parse dataset file early: %s", exc)

        df = None



    sample_values = _extract_sample_values(df) if df is not None else {}

    analysis_ctx = _make_analysis_context(analysis_row, dataset, sample_values)



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





    try:

        chunks = await retrieve_chunks(

            dataset_id, retrieval_query, top_k=settings.CHAT_RETRIEVAL_TOP_K

        )





        if not chunks:

            logger.info("Initial retrieval returned 0 chunks, attempting query rewrite...")

            rewrite_prompt = (

                f"Extract only the core searchable analytical keywords from this query. "

                f"Ignore conversational words. Provide 2-3 keywords max.\nQuery: {retrieval_query}"

            )

            try:



                from app.services.rag.gemini_client import generate_json_response as _gen_json

                raw_rewritten = await _gen_json(

                    "You are a search term extractor. Output JSON: {\"keywords\": \"...\"}",

                    rewrite_prompt,

                    timeout_s=5.0

                )

                keywords = raw_rewritten.get("keywords", "")

                if keywords:

                    logger.info("Retrying retrieval with keywords: %s", keywords)

                    chunks = await retrieve_chunks(

                        dataset_id, keywords, top_k=settings.CHAT_RETRIEVAL_TOP_K

                    )

            except Exception as rewrite_exc:

                logger.warning("Query rewrite failed: %s", rewrite_exc)



    except Exception as exc:

        logger.warning("Retrieval failed (falling back to empty): %s", exc)

        chunks = []



    chunk_ids = [c["chunk_id"] for c in chunks]

    retrieval_ctx = _build_retrieval_context(chunks)

    logger.info("chat_turn: retrieved %d chunks", len(chunks))





    report_block = ""
    if report_context:
        section_hint = ""
        if active_report_section:
            section_hint = f"\nACTIVE_REPORT_SECTION: {active_report_section}\n"
        report_block = f"""
REPORT_CONTEXT (user is asking a follow-up about this AI-generated report — prefer the report for narrative answers; use tool_calls for proof, numbers, or charts):
{report_context}
{section_hint}
IMPORTANT: This is report Q&A. Answer in French. Do not ask for clarification when the question references a recommendation, section, source, or report item that exists in REPORT_CONTEXT.
"""

    plan_user_prompt = f"""\
DATASET_CONTEXT (use ONLY these column names in tool_calls):
{json.dumps(analysis_ctx, default=str, indent=2)}
{report_block}
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



        plan = OrchestratorPlan(intent=Intent.retrieve_only, tool_calls=[], needs_chart=False)





    _RELATION_PATTERNS = re.compile(

        r"\b(correlat|affect|impact|improve|relate|wonder if|cut.*hour|cause|between)\b",

        re.IGNORECASE,

    )

    if plan.intent == Intent.retrieve_only and not plan.tool_calls:

        if _RELATION_PATTERNS.search(user_message):

            metrics = analysis_ctx.get("metrics", [])

            if len(metrics) >= 2:

                col_list = json.dumps(metrics)

                plan = OrchestratorPlan(

                    intent=Intent.analyze,

                    tool_calls=[ToolArgs(

                        code_expr=f"df[{col_list}].corr()",

                        label="Correlation between numeric columns",

                    )],

                    needs_chart=False,

                )

                logger.info("chat_turn: post-plan guard upgraded speculative query to analyze with corr()")



    if plan.tool_calls and plan.intent in (Intent.analyze, Intent.visualize):

        pass





    if plan.needs_chart and not plan.tool_calls:

        logger.info("chat_turn: needs_chart is true but no tool_calls. Looking in history...")

        for msg in reversed(history_messages):

            role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

            if role_str == "assistant" and getattr(msg, "tool_calls", None):

                try:

                    plan.tool_calls = [ToolArgs(**tc) for tc in msg.tool_calls]

                    logger.info("chat_turn: recovered tool_calls from history: %s", plan.tool_calls)

                    break

                except Exception as e:

                    logger.warning("chat_turn: failed to recover tool_calls from history: %s", e)





    if plan.tool_calls and _COMPARE_CHART_RE.search(user_message):

        plan.needs_chart = True





    user_wants_chart = bool(_CHART_REQUEST_RE.search(user_message))

    if user_wants_chart:

        plan.needs_chart = True

        if not plan.tool_calls:

            for msg in reversed(history_messages):

                role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

                if role_str == "assistant" and getattr(msg, "tool_calls", None):

                    try:

                        plan.tool_calls = [ToolArgs(**tc) for tc in msg.tool_calls]

                        logger.info(

                            "chat_turn: chart request recovered tool_calls from history: %s",

                            plan.tool_calls,

                        )

                        break

                    except Exception as e:

                        logger.warning("chat_turn: chart recovery failed: %s", e)

        if not plan.tool_calls:

            data_q = _last_substantive_user_question(user_message, history_messages)

            plan.tool_calls = await _replan_tools_for_chart(

                analysis_ctx,

                retrieval_ctx,

                transcript_for_planner,

                data_q,

            )

        if plan.tool_calls and plan.intent == Intent.retrieve_only:

            plan.intent = Intent.visualize

        logger.info("chat_turn: chart request detected → needs_chart=True tools=%d", len(plan.tool_calls))



    logger.info("chat_turn: intent=%s tools=%d chart=%s", plan.intent, len(plan.tool_calls), plan.needs_chart)





    tool_summary = ", ".join(tc.label or tc.code_expr[:40] for tc in plan.tool_calls) if plan.tool_calls else "none"

    yield _sse(

        SseEventType.intent,

        IntentEvent(

            intent=plan.intent.value,

            plan_summary=f"Tools: {tool_summary} | chart: {plan.needs_chart}",

        ).model_dump(),

    )





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
        if report_context and msg.strip().lower().startswith("could you clarify"):
            msg = "Pouvez-vous préciser votre question ?"

        result_out.update({"full_answer": msg, "intent": plan.intent.value, "chunk_ids": chunk_ids,

                           "latency_ms": int((time.monotonic() - t0) * 1000)})

        yield _sse(SseEventType.token, TokenEvent(delta=msg).model_dump())

        yield _sse(SseEventType.done, DoneEvent(conversation_id=conversation_id, message_id=0,

                                                 latency_ms=result_out["latency_ms"]).model_dump())

        return



    if plan.intent == Intent.generate_report:

        msg = "I'm generating a comprehensive report for you right now based on our conversation..."

        result_out.update({"full_answer": msg, "intent": plan.intent.value, "chunk_ids": chunk_ids,

                           "latency_ms": int((time.monotonic() - t0) * 1000)})

        yield _sse(SseEventType.token, TokenEvent(delta=msg).model_dump())

        yield _sse(SseEventType.done, DoneEvent(conversation_id=conversation_id, message_id=0,

                                                 latency_ms=result_out["latency_ms"]).model_dump())

        return





    yield _sse(SseEventType.retrieval, RetrievalEvent(chunk_ids=chunk_ids).model_dump())





    tool_results: List[Tuple[ToolArgs, ToolResult]] = []

    chart_spec: Optional[ChartSpec] = None



    if df is not None:

        agent = AnalysisAgent(df)

        for tool_args in plan.tool_calls:

            yield _sse(

                SseEventType.tool_start,

                ToolStartEvent(

                    tool="pandas_eval",

                    args_summary=tool_args.label or tool_args.code_expr[:80],

                ).model_dump(),

            )

            try:

                result = await agent.run(tool_args)

                tool_results.append((tool_args, result))

                yield _sse(

                    SseEventType.tool_result,

                    ToolResultEvent(preview=_trim_tool_result_preview(result)).model_dump(),

                )

                logger.info("chat_turn: expr succeeded: %s", tool_args.code_expr[:80])

            except Exception as exc:

                logger.warning("Expr failed: %s | error: %s", tool_args.code_expr[:80], exc)

                yield _sse(

                    SseEventType.tool_result,

                    ToolResultEvent(preview=f"Error: {exc!s}").model_dump(),

                )





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





    tool_results_text = ""

    if tool_results:

        parts = []

        for args, res in tool_results:

            parts.append(

                f"Query: {args.label or args.code_expr}\n"

                f"Result: {json.dumps(res.payload, default=str)[:2000]}"

            )

        tool_results_text = "\n\n".join(parts)

    else:

        tool_results_text = "No tool calls were executed."



    if chart_spec:

        chart_note = (

            f"\nA {chart_spec.chart_type.value} chart titled '{chart_spec.title}' was generated. "

            "The UI renders it below your text — summarize the key insight in 1-2 sentences. "

            "Do NOT say you cannot display charts."

        )

    elif user_wants_chart:

        chart_note = (

            "\nThe user asked for a chart but it could not be built (query or data shape). "

            "Give the numbers from TOOL_RESULTS. NEVER say you cannot generate or display charts."

        )

    else:

        chart_note = (

            "\nNo chart was generated. Do NOT claim to show a chart. "

            "NEVER say you cannot generate or display charts."

        )



    history_contents = _build_history_contents(history_messages, rolling_summary)



    report_narrator_block = ""
    if report_context:
        cite_hint = (
            "When relevant, cite report sections in French (e.g. "
            "\"Voir section Quoi faire · Reco #2\", \"Voir section Pourquoi\"). "
            "Answer in French using the report first; use TOOL_RESULTS for proof or when the user asks for numbers/charts."
        )
        report_narrator_block = f"""
GENERATED REPORT (primary narrative source):
{report_context}

{cite_hint}
"""

    narrator_prompt = f"""\
DATASET INFO:
- File: {dataset.original_filename}
- Rows: {dataset.row_count}
- Primary metric: {analysis_row.primary_metric}
{report_narrator_block}
RETRIEVED_CONTEXT:
{retrieval_ctx}

TOOL_RESULTS:
{tool_results_text}
{chart_note}

USER_QUESTION: {user_message}

Answer directly and specifically. If the tool results contain the answer, give the exact number/values.
If GENERATED REPORT is present, answer in French.
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



        try:

            from app.services.rag.gemini_client import generate_json_response as _gen



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

                try:

                    resp.raise_for_status()

                except httpx.HTTPStatusError as e:

                    with open("err.log", "a") as f:

                        f.write(f"Fallback Error: {e.response.status_code} {e.response.text}\n")

                    raise

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

    if chart_spec and len(full_answer.strip()) < 25:

        fallback_text = _fallback_chart_narrative(chart_spec)

        joiner = "\n\n" if full_answer.strip() else ""

        full_answer = f"{full_answer}{joiner}{fallback_text}"

        yield _sse(SseEventType.token, TokenEvent(delta=f"{joiner}{fallback_text}").model_dump())





    chart_payload = _json_safe(chart_spec.model_dump()) if chart_spec else None



    result_out.update({

        "full_answer": full_answer,

        "intent": plan.intent.value,

        "chunk_ids": chunk_ids,

        "chart_spec": chart_payload,

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

            message_id=0,

            latency_ms=latency_ms,

            chart_spec=chart_payload,

        ).model_dump(),

    )

