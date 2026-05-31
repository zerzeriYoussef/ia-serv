"""Redis cache for generated report documents (follow-up Q&A context)."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.api.v1.schemas.report_schema import ReportDocument
from app.core.cache import get_redis
from app.core.config import settings

logger = logging.getLogger(__name__)

_SECTION_LABELS = {
    "what_happened": "Ce qui s'est passé",
    "why": "Pourquoi",
    "what_to_do": "Quoi faire",
    "sources": "Sources",
}


def _cache_key(dataset_id: int, report_id: str) -> str:
    return f"report:{dataset_id}:{report_id}"


async def cache_report(dataset_id: int, report: ReportDocument) -> None:
    try:
        redis = await get_redis()
        if redis is None:
            logger.warning("report_cache: Redis unavailable, report not cached")
            return
        await redis.set(
            _cache_key(dataset_id, report.report_id),
            report.model_dump_json(),
            ex=settings.REPORT_CACHE_TTL,
        )
        logger.info(
            "report_cache: stored report=%s dataset=%s ttl=%ss",
            report.report_id,
            dataset_id,
            settings.REPORT_CACHE_TTL,
        )
    except Exception as exc:
        logger.warning("report_cache: failed to store report: %s", exc)


async def get_cached_report(dataset_id: int, report_id: str) -> Optional[ReportDocument]:
    try:
        redis = await get_redis()
        if redis is None:
            return None
        raw = await redis.get(_cache_key(dataset_id, report_id))
        if not raw:
            return None
        return ReportDocument.model_validate_json(raw)
    except Exception as exc:
        logger.warning("report_cache: failed to load report: %s", exc)
        return None


def format_report_context(
    report: ReportDocument,
    *,
    active_section: Optional[str] = None,
) -> str:
    """Serialize report sections for LLM follow-up prompts."""
    lines: list[str] = [
        f"Report ID: {report.report_id}",
        f"Dataset: {report.dataset_name}",
    ]

    if report.filters_applied:
        lines.append(f"Filters: {json.dumps(report.filters_applied, default=str)}")

    lines.append("\n## Ce qui s'est passé")
    lines.append(report.what_happened.narrative or "(empty)")

    lines.append("\n## Pourquoi")
    lines.append(report.why_it_happened.narrative or "(empty)")

    sources = report.why_it_happened.sources or []
    if sources:
        lines.append("\n### Sources")
        for src in sources:
            bit = f"- [{src.type}] {src.title}"
            if src.url:
                bit += f" ({src.url})"
            lines.append(bit)

    lines.append("\n## Quoi faire (recommendations)")
    if report.what_to_do:
        for idx, rec in enumerate(report.what_to_do, start=1):
            lines.append(
                f"Reco #{idx} (priority {rec.priority}): {rec.action}\n"
                f"  Expected outcome: {rec.expected_outcome}"
            )
    else:
        lines.append("(none)")

    if report.what_to_avoid:
        lines.append("\n## À éviter")
        for item in report.what_to_avoid:
            lines.append(f"- {item}")

    if active_section:
        label = _SECTION_LABELS.get(active_section, active_section)
        lines.append(f"\nUser is currently viewing report section: {label} ({active_section})")

    return "\n".join(lines)
