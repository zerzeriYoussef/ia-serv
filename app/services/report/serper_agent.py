"""
Serper Web-Search Agent — LangChain-powered external context fetcher.

Uses LangChain's GoogleSerperAPIWrapper to retrieve real-time business
context (benchmarks, market trends, competitor insights) that enriches
the AI report with data beyond the user's local dataset.

Flow:
  1. build_report_search_queries()  → derive 3-5 targeted queries
  2. fetch_external_context()       → run queries via Serper, return results
  3. format_serper_context()        → produce a clean text block for the prompt
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query builder — derives smart Serper queries from dataset metadata
# ---------------------------------------------------------------------------

_METRIC_INDUSTRY_HINTS: Dict[str, str] = {
    "revenue":      "SaaS business",
    "sales":        "retail sales",
    "churn":        "SaaS subscription",
    "mrr":          "SaaS monthly recurring revenue",
    "arr":          "annual recurring revenue",
    "goals":        "sports performance",
    "score":        "performance analytics",
    "spend":        "marketing spend",
    "cost":         "operational cost",
    "profit":       "profit margin",
    "temperature":  "climate environment",
    "population":   "demographic",
}


def _guess_industry(primary_metric: str, dimensions: List[str]) -> str:
    """Best-effort industry label from metric/dimension names."""
    combined = (primary_metric or "").lower() + " ".join(d.lower() for d in dimensions)
    for keyword, industry in _METRIC_INDUSTRY_HINTS.items():
        if keyword in combined:
            return industry
    return "business"


def build_report_search_queries(
    primary_metric: str,
    dimensions: List[str],
    dataset_name: str,
    year: int = 2024,
) -> List[str]:
    """
    Produce 4 targeted search queries for external business context.

    Args:
        primary_metric: e.g. "revenue", "Total_Goals"
        dimensions:     e.g. ["region", "category"]
        dataset_name:   original filename hint
        year:           benchmark year for recency

    Returns:
        List of search query strings ready for Serper.
    """
    industry = _guess_industry(primary_metric, dimensions)
    metric_clean = primary_metric.replace("_", " ").lower()

    queries = [
        f"{metric_clean} industry benchmark {year} {industry}",
        f"how to improve {metric_clean} best practices {industry}",
        f"{industry} performance KPI trends {year}",
    ]

    # Add dimension-specific query if we have a geographic/category dimension
    geo_hints = {"region", "country", "continent", "market", "territory", "area"}
    for dim in dimensions:
        if any(h in dim.lower() for h in geo_hints):
            queries.append(f"{industry} {metric_clean} by {dim.lower()} regional analysis {year}")
            break

    logger.info("serper_agent: generated %d queries for metric='%s'", len(queries), primary_metric)
    return queries[:5]  # cap at 5 to stay within Serper free tier


# ---------------------------------------------------------------------------
# Serper fetcher — LangChain GoogleSerperAPIWrapper
# ---------------------------------------------------------------------------

def _get_serper_wrapper(num_results: int = 4):
    """Lazy-init the LangChain Serper wrapper."""
    from langchain_community.utilities import GoogleSerperAPIWrapper
    import os
    # LangChain reads SERPER_API_KEY from env automatically
    os.environ.setdefault("SERPER_API_KEY", settings.SERPER_API_KEY or "")
    return GoogleSerperAPIWrapper(k=num_results)


async def fetch_external_context(
    queries: List[str],
    *,
    num_results: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run multiple Serper queries concurrently via LangChain and return
    a flat list of result dicts: {query, title, snippet, url}.

    Gracefully returns [] if SERPER_API_KEY is missing.
    """
    if not settings.SERPER_API_KEY:
        logger.warning("serper_agent: SERPER_API_KEY not set — skipping web search")
        return []

    try:
        wrapper = _get_serper_wrapper(num_results)
    except Exception as exc:
        logger.error("serper_agent: failed to init wrapper: %s", exc)
        return []

    async def _run_one(query: str) -> List[Dict[str, Any]]:
        """Run a single Serper query in a thread (wrapper is sync)."""
        try:
            raw: str = await asyncio.get_event_loop().run_in_executor(
                None, wrapper.run, query
            )
            # wrapper.run() returns a text summary; use results() for structured data
            structured = await asyncio.get_event_loop().run_in_executor(
                None, wrapper.results, query
            )
            organic = structured.get("organic", [])
            results = []
            for item in organic[:num_results]:
                results.append({
                    "query":   query,
                    "title":   item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url":     item.get("link", ""),
                })
            return results
        except Exception as exc:
            logger.warning("serper_agent: query '%s' failed: %s", query[:60], exc)
            return []

    # Run all queries concurrently
    tasks = [_run_one(q) for q in queries]
    nested = await asyncio.gather(*tasks, return_exceptions=False)

    # Flatten + deduplicate by URL
    seen_urls: set[str] = set()
    flat: List[Dict[str, Any]] = []
    for batch in nested:
        for item in batch:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                flat.append(item)

    logger.info("serper_agent: fetched %d unique results across %d queries", len(flat), len(queries))
    return flat


# ---------------------------------------------------------------------------
# Format for LLM prompt
# ---------------------------------------------------------------------------

def format_serper_context(results: List[Dict[str, Any]]) -> str:
    """
    Convert Serper results into a clean text block for the report prompt.

    Lines are tagged [WEB-n] so report prompts can require hybrid answers that
    blend dataset facts (INTERNAL_CONTEXT) with these snippets (EXTERNAL_CONTEXT).

    Example output:
        [WEB-1] Industry Revenue Benchmark 2024 (https://...)
        Average SaaS revenue growth in 2024 was 18% YoY across mid-market...

        [WEB-2] ...
    """
    if not results:
        return "No external web context available."

    lines: List[str] = []
    for i, item in enumerate(results, start=1):
        title   = item.get("title", "")
        snippet = item.get("snippet", "")
        url     = item.get("url", "")
        lines.append(f"[WEB-{i}] {title} ({url})\n{snippet}")

    return "\n\n".join(lines)
