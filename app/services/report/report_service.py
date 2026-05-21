"""
Report Service — the main streaming pipeline.

Step 1 : Build internal RAG context via LangChain Chroma retrievals
Step 2 : Fetch external context via LangChain + Serper
Step 3 : Generate each report section via LangChain ChatGoogleGenerativeAI
Step 4 : Stream narrative tokens, emit section events, emit final report_done

SSE event sequence:
  report_start → report_step(loading_context)
              → report_step(rag_retrieval)
              → report_step(web_search)
              → report_step(merging_context)
              → report_step(generating)
              → report_token* (what_happened narrative)
              → report_token* (why_it_happened narrative)
              → report_token* (what_to_do narrative)
              → report_section (what_to_do JSON array)
              → report_step(finalizing)
              → report_done (full ReportDocument)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.report_schema import (
    WhatHappened,
    WhyItHappened,
    ActionableRecommendation,
    ReportSource,
    ReportDocument,
    ReportDoneEvent,
    ReportErrorEvent,
    ReportSseEventType,
    ReportStartEvent,
    ReportStep,
    ReportStepEvent,
    ReportSectionEvent,
    ReportTokenEvent,
)
from app.core.config import settings
from app.services.report.report_context_builder import build_internal_context
from app.services.report.report_prompt_builder import (
    build_what_happened_prompt,
    build_why_it_happened_prompt,
    build_what_to_do_prompt,
)
from app.services.report.serper_agent import (
    build_report_search_queries,
    fetch_external_context,
    format_serper_context,
)
from app.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event_type: ReportSseEventType, data: Any) -> str:
    """Format one SSE frame."""
    if hasattr(data, "model_dump_json"):
        payload = data.model_dump_json()
    elif isinstance(data, dict):
        payload = json.dumps(data, default=str)
    else:
        payload = json.dumps(str(data))
    return f"event: {event_type.value}\ndata: {payload}\n\n"


def _step(step: ReportStep, message: str) -> str:
    return _sse(ReportSseEventType.report_step, ReportStepEvent(step=step, message=message))


def _token(section: str, delta: str) -> str:
    return _sse(ReportSseEventType.report_token, ReportTokenEvent(section=section, delta=delta))


def _section(section_name: str, data: Any) -> str:
    return _sse(ReportSseEventType.report_section, ReportSectionEvent(section=section_name, data=data))


# ---------------------------------------------------------------------------
# LangChain LLM helpers
# ---------------------------------------------------------------------------

def _message_content_to_str(content: Any) -> str:
    """Normalize LangChain / Gemini message content to a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _strip_code_fences(text: str) -> str:
    """Remove ``` or ```json fences if the model wrapped JSON."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    # Drop opening fence (first line may be ```json)
    lines = t.split("\n", 1)
    if len(lines) < 2:
        return t
    rest = lines[1]
    if rest.rstrip().endswith("```"):
        rest = rest.rstrip()[:-3].rstrip()
    return rest


def _first_balanced_json_substring(s: str) -> Optional[str]:
    """
    If the model added prose before/after JSON, take the first top-level
    object or array by bracket matching (handles nested structures).
    """
    i = 0
    while i < len(s) and s[i] not in "{[":
        i += 1
    if i >= len(s):
        return None
    stack: List[str] = []
    in_string = False
    escape = False
    for j in range(i, len(s)):
        c = s[j]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c in "}]":
            if not stack or c != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return s[i : j + 1]
    return None


def parse_llm_json_response(raw_text: str) -> Any:
    """
    Best-effort JSON parse for Gemini output: full document, then fenced, then
    first balanced { } or [ ] substring.
    """
    text = raw_text.strip()
    if not text:
        raise ValueError("empty model response")
    for candidate in (text, _strip_code_fences(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    stripped = _strip_code_fences(text)
    sub = _first_balanced_json_substring(stripped) or _first_balanced_json_substring(text)
    if sub:
        return json.loads(sub)
    raise ValueError("no parseable JSON in model output")


def _get_llm(streaming: bool = False):
    """Return a LangChain ChatGoogleGenerativeAI instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
        streaming=streaming,
    )


def _language_label(language: str) -> str:
    normalized = (language or "fr").strip().lower()
    if normalized.startswith("en"):
        return "English"
    if normalized.startswith("ar"):
        return "Arabic"
    if normalized.startswith("es"):
        return "Spanish"
    return "French"


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _fallback_what_happened(fact_pack: Dict[str, Any], language: str) -> WhatHappened:
    dataset = fact_pack.get("dataset") or {}
    quality = fact_pack.get("quality") or {}
    metric_facts = fact_pack.get("metric_facts") or []
    trend = fact_pack.get("trend") or {}
    metric = metric_facts[0] if metric_facts else {}

    if language == "English":
        narrative = (
            f"The dataset contains {quality.get('row_count', 0)} rows and "
            f"{quality.get('column_count', 0)} columns. The main metric is "
            f"{dataset.get('primary_metric') or 'not detected'}."
        )
        if metric:
            maximum = metric.get("maximum") or {}
            minimum = metric.get("minimum") or {}
            narrative += (
                f" {metric.get('metric')} totals {_format_value(metric.get('total'))}, "
                f"averages {_format_value(metric.get('average'))}, peaks at "
                f"{_format_value(maximum.get('value'))} in {maximum.get('where')}, "
                f"and bottoms at {_format_value(minimum.get('value'))} in {minimum.get('where')}."
            )
        if trend:
            narrative += (
                f" Over time, {trend.get('metric')} moved from "
                f"{_format_value(trend.get('first_value'))} in {trend.get('first_period')} "
                f"to {_format_value(trend.get('last_value'))} in {trend.get('last_period')}, "
                f"with a peak of {_format_value(trend.get('peak_value'))} in {trend.get('peak_period')}."
            )
    else:
        narrative = (
            f"Le dataset contient {quality.get('row_count', 0)} lignes et "
            f"{quality.get('column_count', 0)} colonnes. La metrique principale est "
            f"{dataset.get('primary_metric') or 'non detectee'}."
        )
        if metric:
            maximum = metric.get("maximum") or {}
            minimum = metric.get("minimum") or {}
            narrative += (
                f" {metric.get('metric')} totalise {_format_value(metric.get('total'))}, "
                f"avec une moyenne de {_format_value(metric.get('average'))}, un maximum de "
                f"{_format_value(maximum.get('value'))} a {maximum.get('where')} "
                f"et un minimum de {_format_value(minimum.get('value'))} a {minimum.get('where')}."
            )
        if trend:
            narrative += (
                f" Dans le temps, {trend.get('metric')} passe de "
                f"{_format_value(trend.get('first_value'))} en {trend.get('first_period')} "
                f"a {_format_value(trend.get('last_value'))} en {trend.get('last_period')}, "
                f"avec un pic de {_format_value(trend.get('peak_value'))} en {trend.get('peak_period')}."
            )
    return WhatHappened(narrative=narrative)


def _fallback_why_it_happened(fact_pack: Dict[str, Any], language: str) -> WhyItHappened:
    dim_facts = fact_pack.get("dimension_facts") or []
    relationships = fact_pack.get("relationships") or []
    parts: List[str] = []
    sources: List[ReportSource] = []

    if dim_facts:
        dim = dim_facts[0]
        top_metric = dim.get("top_values_by_metric_total") or []
        top_count = dim.get("top_values_by_count") or []
        if top_metric:
            leader = top_metric[0]
            if language == "English":
                parts.append(
                    f"The main visible driver is {dim.get('dimension')}: "
                    f"{leader.get('value')} leads the metric total with "
                    f"{_format_value(leader.get('metric_total'))}."
                )
            else:
                parts.append(
                    f"Le principal facteur visible est {dim.get('dimension')} : "
                    f"{leader.get('value')} domine le total avec "
                    f"{_format_value(leader.get('metric_total'))}."
                )
        elif top_count:
            leader = top_count[0]
            if language == "English":
                parts.append(
                    f"The dataset is concentrated around {leader.get('value')} "
                    f"for {dim.get('dimension')} ({leader.get('count')} rows)."
                )
            else:
                parts.append(
                    f"Le dataset est concentre autour de {leader.get('value')} "
                    f"pour {dim.get('dimension')} ({leader.get('count')} lignes)."
                )
        sources.append(ReportSource(type="internal", title=f"Dimension: {dim.get('dimension')}"))

    if relationships:
        rel = relationships[0]
        cols = " <-> ".join(rel.get("columns") or [])
        if cols:
            if language == "English":
                parts.append(
                    f"The strongest detected relationship is {cols}, "
                    f"with strength {_format_value(rel.get('strength'))}."
                )
            else:
                parts.append(
                    f"La relation la plus forte detectee est {cols}, "
                    f"avec une force de {_format_value(rel.get('strength'))}."
                )
            sources.append(ReportSource(type="internal", title=f"Relationship: {cols}"))

    if not parts:
        parts.append(
            "The current dataset is mainly descriptive; the strongest explanation comes from the ranked metric and quality facts."
            if language == "English"
            else "Le dataset est surtout descriptif ; l'explication la plus fiable vient des classements, tendances et controles qualite."
        )
        sources.append(ReportSource(type="internal", title="Computed fact pack"))

    return WhyItHappened(narrative=" ".join(parts), sources=sources)


def _fallback_recommendations(
    fact_pack: Dict[str, Any],
    language: str,
) -> tuple[List[ActionableRecommendation], List[str]]:
    dataset = fact_pack.get("dataset") or {}
    metric = dataset.get("primary_metric") or "primary metric"
    trend = fact_pack.get("trend") or {}
    quality = fact_pack.get("quality") or {}

    if language == "English":
        recs = [
            ActionableRecommendation(
                priority=1,
                action=f"Build the report headline around {metric} and its peak/low periods.",
                expected_outcome="Readers immediately see the main movement instead of a generic summary.",
            ),
            ActionableRecommendation(
                priority=2,
                action="Add a ranked breakdown for the top dimension values by count and metric total.",
                expected_outcome="The report shows which segments matter most and supports better dashboard filters.",
            ),
            ActionableRecommendation(
                priority=3,
                action=f"Track data quality before every report run: {quality.get('missing_cells', 0)} missing cells and {quality.get('duplicate_rows', 0)} duplicate rows were found here.",
                expected_outcome="The user can trust the report and spot weak input data early.",
            ),
        ]
        avoids = ["Avoid broad recommendations that do not name a metric, segment, or period."]
    else:
        recs = [
            ActionableRecommendation(
                priority=1,
                action=f"Construire le message principal autour de {metric} et de ses periodes de pic/faible niveau.",
                expected_outcome="Le lecteur voit tout de suite le mouvement important au lieu d'un resume generique.",
            ),
            ActionableRecommendation(
                priority=2,
                action="Ajouter un classement des principales dimensions par volume et par total de la metrique.",
                expected_outcome="Le rapport montre quels segments comptent vraiment et aide a definir les filtres du dashboard.",
            ),
            ActionableRecommendation(
                priority=3,
                action=f"Controler la qualite avant chaque generation : {quality.get('missing_cells', 0)} cellules manquantes et {quality.get('duplicate_rows', 0)} doublons ici.",
                expected_outcome="Le rapport devient plus fiable et les donnees faibles sont detectees plus tot.",
            ),
        ]
        avoids = ["Eviter les recommandations larges qui ne citent ni metrique, ni segment, ni periode."]

    if trend:
        recs[0].action += (
            f" Peak: {trend.get('peak_period')} ({_format_value(trend.get('peak_value'))})."
        )
    return recs, avoids


async def _invoke_json_section(messages: List[Any], section_name: str) -> Any:
    """
    Call Gemini via LangChain (non-streaming) and parse the JSON response.
    Returns the parsed Python object (list or dict).
    """
    llm = _get_llm(streaming=False)
    raw_text = ""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, llm.invoke, messages
        )
        raw_text = _message_content_to_str(
            response.content if hasattr(response, "content") else str(response)
        )
        return parse_llm_json_response(raw_text)
    except Exception as exc:
        preview = (raw_text[:800] + "…") if len(raw_text) > 800 else raw_text
        logger.error(
            "report_service: JSON section '%s' parse failed: %s | preview=%r",
            section_name,
            exc,
            preview,
        )
        return None


# ---------------------------------------------------------------------------
# Section parsers — convert raw LLM dicts into Pydantic models
# ---------------------------------------------------------------------------

def _parse_what_happened(raw: Any) -> WhatHappened:
    if not isinstance(raw, dict):
        return WhatHappened(narrative=str(raw) if raw else "")
    return WhatHappened(narrative=raw.get("narrative", ""))


def _parse_why_it_happened(raw: Any) -> WhyItHappened:
    if not isinstance(raw, dict):
        return WhyItHappened(narrative=str(raw) if raw else "")
    sources = []
    for s in raw.get("sources", []):
        if isinstance(s, dict):
            sources.append(ReportSource(**{k: v for k, v in s.items() if k in ReportSource.model_fields}))
    return WhyItHappened(
        narrative=raw.get("narrative", ""),
        sources=sources,
    )


def _parse_what_to_do(raw: Any) -> tuple[List[ActionableRecommendation], List[str]]:
    if not isinstance(raw, dict):
        return [], []
    recs = []
    for item in raw.get("what_to_do", []):
        if not isinstance(item, dict):
            continue
        try:
            recs.append(ActionableRecommendation(**{k: v for k, v in item.items() if k in ActionableRecommendation.model_fields}))
        except Exception:
            pass
    
    avoids = []
    for item in raw.get("what_to_avoid", []):
        if isinstance(item, str):
            avoids.append(item)
            
    return sorted(recs, key=lambda r: r.priority), avoids


# ---------------------------------------------------------------------------
# Main streaming pipeline
# ---------------------------------------------------------------------------

async def generate_report_stream(
    db: AsyncSession,
    dataset_id: int,
    filters: Optional[Dict[str, Any]] = None,
    include_web_context: bool = True,
    conversation_id: Optional[int] = None,
    language: str = "fr",
) -> AsyncGenerator[str, None]:
    """
    4-step pipeline. Yields raw SSE frame strings.
    """
    t0 = time.monotonic()
    report_id = str(uuid4())
    filters = filters or {}
    language_name = _language_label(language)

    # ── STEP 1: Load internal context via LangChain RAG + Conversation Context ───────────────────
    yield _step(ReportStep.loading_context, "Loading dataset analysis and conversation context…")

    try:
        internal_ctx, chunks_by_topic = await build_internal_context(db, dataset_id)
    except ValueError as exc:
        yield _sse(ReportSseEventType.report_error, ReportErrorEvent(error=str(exc)))
        return
    except Exception as exc:
        logger.error("report_service: internal context failed: %s", exc)
        yield _sse(ReportSseEventType.report_error, ReportErrorEvent(error="Failed to load dataset context."))
        return

    dataset_name    = internal_ctx["dataset_name"]
    primary_metric  = internal_ctx["primary_metric"] or "performance"
    metrics         = internal_ctx["metrics"]
    dimensions      = internal_ctx["dimensions"]
    fact_pack       = internal_ctx.get("fact_pack") or {}
    fact_pack_text  = internal_ctx.get("fact_pack_text") or "No computed facts available."

    # Fetch conversation context if available
    conversation_ctx_text = "No prior conversation."
    if conversation_id:
        messages = await ConversationRepository.get_messages(db, conversation_id, limit=10)
        if messages:
            # Reverse to make it chronological
            messages = list(reversed(messages))
            lines = []
            for m in messages:
                role = "user" if m.role.value == "user" else "assistant"
                lines.append(f"{role}: {m.content}")
            conversation_ctx_text = "\\n".join(lines)
            logger.info("report_service: loaded %d messages for conversation context", len(messages))

    # Emit start now that we have the dataset name
    yield _sse(ReportSseEventType.report_start, ReportStartEvent(
        report_id=report_id,
        dataset_name=dataset_name,
    ))

    yield _step(ReportStep.rag_retrieval, f"Retrieved context topics from vector store…")

    # ── STEP 2: External Serper context via LangChain ─────────────────────
    serper_results = []
    external_ctx_text = "No external web context requested."

    if include_web_context:
        yield _step(ReportStep.web_search, "Fetching external industry context via Serper…")
        try:
            queries = build_report_search_queries(primary_metric, dimensions, dataset_name)
            serper_results = await fetch_external_context(queries)
            external_ctx_text = format_serper_context(serper_results)
            yield _step(
                ReportStep.web_search,
                f"Fetched {len(serper_results)} external sources for industry context.",
            )
        except Exception as exc:
            logger.warning("report_service: Serper failed (non-fatal): %s", exc)
            yield _step(ReportStep.web_search, "Web search unavailable — continuing with internal data only.")

    # ── STEP 3: Merge context + generate sections ─────────────────────────
    yield _step(ReportStep.merging_context, "Merging internal RAG + external context…")
    yield _step(ReportStep.generating, "Generating report sections with Gemini…")

    # -- 3a: What Happened (streaming narrative) -----------------------
    wh_messages = build_what_happened_prompt(
        internal_ctx  = (
            chunks_by_topic.get("executive", "")
            + "\n\nKPI_CONTEXT:\n"
            + json.dumps(internal_ctx.get("kpi_block"), default=str)
        ),
        fact_pack = fact_pack_text,
        conversation_ctx = conversation_ctx_text,
        dataset_name  = dataset_name,
        primary_metric= primary_metric,
        language=language_name,
    )
    wh_raw = await _invoke_json_section(wh_messages, "what_happened")
    what_happened = _parse_what_happened(wh_raw)
    if not what_happened.narrative.strip():
        what_happened = _fallback_what_happened(fact_pack, language_name)
    
    # Stream the narrative token-by-token from the paragraph
    wh_paragraph = what_happened.narrative or "No data available to explain what happened."
    word_buffer = ""
    for word in wh_paragraph.split():
        word_buffer += word + " "
        if len(word_buffer) >= 8:          # emit in ~8-char bursts
            yield _token("what_happened", word_buffer)
            word_buffer = ""
            await asyncio.sleep(0.01)       # tiny sleep for SSE flush
    if word_buffer:
        yield _token("what_happened", word_buffer)

    yield _section("what_happened", what_happened.model_dump())

    # -- 3b: Why It Happened (streaming narrative) --------------------------
    why_messages = build_why_it_happened_prompt(
        internal_ctx   = (
            chunks_by_topic.get("relationships", "")
            + "\n\n"
            + chunks_by_topic.get("dimensions", "")
            + "\n\nGUARDRAILS:\n"
            + chunks_by_topic.get("guardrails", "")
        ),
        fact_pack = fact_pack_text,
        conversation_ctx = conversation_ctx_text,
        external_ctx   = external_ctx_text,
        language=language_name,
    )
    why_raw = await _invoke_json_section(why_messages, "why_it_happened")
    why_it_happened = _parse_why_it_happened(why_raw)
    if not why_it_happened.narrative.strip():
        why_it_happened = _fallback_why_it_happened(fact_pack, language_name)
    
    why_narrative = why_it_happened.narrative or "No data available to explain why."
    for word in why_narrative.split():
        yield _token("why_it_happened", word + " ")
        await asyncio.sleep(0.01)

    yield _section("why_it_happened", why_it_happened.model_dump())

    # -- 3c: What To Do (streaming narrative) ------------------------
    wtd_messages = build_what_to_do_prompt(
        what_happened = what_happened.narrative,
        why_it_happened = why_it_happened.narrative,
        fact_pack = fact_pack_text,
        conversation_ctx = conversation_ctx_text,
        language=language_name,
    )
    wtd_raw = await _invoke_json_section(wtd_messages, "what_to_do")
    what_to_do, what_to_avoid = _parse_what_to_do(wtd_raw)
    if not what_to_do:
        what_to_do, what_to_avoid = _fallback_recommendations(fact_pack, language_name)
    
    for rec in what_to_do:
        action_text = f"[{rec.priority}] {rec.action} → {rec.expected_outcome} "
        for word in action_text.split():
            yield _token("what_to_do", word + " ")
            await asyncio.sleep(0.01)
            
    if what_to_avoid:
        for avoid in what_to_avoid:
            avoid_text = f"AVOID: {avoid} "
            for word in avoid_text.split():
                yield _token("what_to_avoid", word + " ")
                await asyncio.sleep(0.01)

    yield _section("what_to_do", [r.model_dump() for r in what_to_do])
    if what_to_avoid:
        yield _section("what_to_avoid", what_to_avoid)

    # ── STEP 4: Assemble & emit final report ─────────────────────────────
    yield _step(ReportStep.finalizing, "Assembling final report…")

    report = ReportDocument(
        report_id              = report_id,
        dataset_name           = dataset_name,
        filters_applied        = filters,
        what_happened          = what_happened,
        why_it_happened        = why_it_happened,
        what_to_do             = what_to_do,
        what_to_avoid          = what_to_avoid,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "report_service: done dataset=%s report_id=%s latency=%dms",
        dataset_id, report_id, latency_ms,
    )

    yield _sse(
        ReportSseEventType.report_done,
        ReportDoneEvent(report=report, latency_ms=latency_ms),
    )
