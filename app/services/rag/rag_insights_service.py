"""
Orchestrate: compact context for the LLM + RAG retrieval + Gemini executive JSON.
"""



from __future__ import annotations



import json

import logging

from typing import Any, Dict, List, Optional, Tuple



from sqlalchemy.ext.asyncio import AsyncSession



from app.models.dashboard import Dashboard

from app.repositories.analysis_repository import AnalysisRepository

from app.repositories.dashboard_repository import DashboardRepository

from app.repositories.dataset_repository import DatasetRepository

from app.services.analysis.kpi_service import build_kpi_block

from app.services.data.parser_service import ParserService

from app.services.rag.gemini_client import generate_json_response

from app.services.rag.knowledge_retriever import retrieve_for_query



logger = logging.getLogger(__name__)



SYSTEM_PROMPT = """You are a senior analytics product lead. Output valid JSON ONLY.
Audience: executives and dashboard builders — no dumping internal analysis structures.
Use ONLY column names that appear in DATASET_CONTEXT or KPI_HINTS.
Downrank trivial links (IDs, emails, UUIDs → names). Prefer primary metric and real business dimensions.
Ground pandas_grouping and chart_type in RETRIEVED_KNOWLEDGE when it fits.
"""





def _slim_relationships(relationships: List[Dict[str, Any]], limit: int = 24) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []

    for rel in (relationships or [])[:limit]:

        out.append(

            {

                "columns": rel.get("columns"),

                "type": rel.get("type"),

                "strength": rel.get("strength"),

                "method": rel.get("method"),

            }

        )

    return out





def _kpi_hints(kpi: Optional[Dict[str, Any]]) -> str:

    """Short text for the model — not returned to the client."""

    if not kpi:

        return "No primary metric KPI could be computed (missing numeric primary column or all-null)."

    lines: List[str] = []

    prim = kpi.get("primary") or {}

    lines.append(

        f"Primary metric: {prim.get('name')} → total={prim.get('value')} (aggregation={prim.get('aggregation')})."

    )

    trend = kpi.get("trend")

    if trend and trend.get("series"):

        s = trend["series"]

        lines.append(

            f"Time trend ({trend.get('time_column')}, {trend.get('granularity')}): "

            f"{len(s)} periods; last={s[-1] if s else 'n/a'}."

        )

    for block in (kpi.get("by_dimension") or [])[:3]:

        dim = block.get("dimension")

        top = (block.get("top_n") or [])[:3]

        tops = ", ".join(f"{t.get('key')}:{t.get('value')}" for t in top if isinstance(t, dict))

        lines.append(f"By {dim} (sample top): {tops}.")

    row_card = next(

        (c for c in (kpi.get("cards") or []) if c.get("name") == "row_count"),

        None,

    )

    if isinstance(row_card, dict):

        lines.append(f"Row count: {row_card.get('value')}.")

    return "\n".join(lines)





async def run_rag_insights(

    db: AsyncSession,

    dataset_id: int,

) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    """
    Returns (llm_executive_json, meta) where meta has rag debug fields.
    """

    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)

    if not analysis_row:

        raise ValueError("no_analysis")



    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise ValueError("no_dataset")



    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)



    kpi = build_kpi_block(

        df,

        analysis_row.primary_metric,

        analysis_row.temporal,

        analysis_row.dimensions,

    )



    analysis_context: Dict[str, Any] = {

        "dataset_id": dataset_id,

        "row_count": len(df),

        "primary_metric": analysis_row.primary_metric,

        "dashboard_columns": (analysis_row.dashboard_columns or [])[:12],

        "column_categories": {

            "metrics": analysis_row.metrics,

            "dimensions": analysis_row.dimensions,

            "temporal": analysis_row.temporal,

            "geographic": analysis_row.geographic,

            "identifiers": analysis_row.identifiers,

            "other": (analysis_row.other or [])[:15],

        },

        "relationships": _slim_relationships(analysis_row.relationships or []),

    }



    query = " ".join(

        str(x)

        for x in (

            analysis_row.metrics or []

        )

        + (analysis_row.dimensions or [])

        + (analysis_row.temporal or [])

        + [analysis_row.primary_metric or ""]

    )

    for rel in (analysis_row.relationships or [])[:20]:

        query += " " + " ".join(str(c) for c in (rel.get("columns") or []))

        query += " " + str(rel.get("type") or "")



    chunks = retrieve_for_query(query, top_k=6)

    kb_text = "\n\n---\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)



    kpi_hints = _kpi_hints(kpi)



    user_prompt = f"""DATASET_CONTEXT (for column names and structure only — do not echo this object in your answer):
{json.dumps(analysis_context, default=str)}

KPI_HINTS (one-line facts computed in our backend — use to inspire KPI names and copy, do NOT paste this block into the response):
{kpi_hints}

RETRIEVED_KNOWLEDGE:
{kb_text}

Return a SINGLE JSON object with EXACTLY these two top-level keys and nothing else:

{
  "executive_summary_kpis": [
    {
      "kpi_name": "Short professional name",
      "target_column": "exact column name from context",
      "pandas_function": "e.g. .sum() or groupby pattern label",
      "logic_hint": "df['col'].mean() or one-line pandas",
      "description": "Why this KPI matters (non-technical)",
      "icon": "material icon name e.g. trending_up, payments, groups"
    }
  ],
  "dashboard_charts": [
    {
      "rank": 1,
      "title": "Chart title for a dashboard tile",
      "relationship": "Column A ↔ Column B",
      "strength_score": 0.0,
      "x_axis_column": "column or null",
      "y_axis_column": "column or null",
      "pandas_grouping": "df.groupby('x')['y'].sum() style one-liner",
      "chart_type": "e.g. Grouped Bar, Line, Scatter, Heatmap, Box Plot",
      "business_insight": "What this visual proves for a business reader"
    }
  ]
}

Rules:
- 4–8 items in executive_summary_kpis (diverse: total, average, trend, top dimension slice when applicable).
- 6–10 items in dashboard_charts, sorted by rank; strength_score between 0 and 1 (use relationship strength from context when available).
- Override the KPI count above: return 8-12 executive_summary_kpis when enough valid columns exist.
- Override the chart count above: return 8-12 dashboard_charts when enough valid columns exist.
- KPI logic_hint may use: df['col'].sum(), mean(), median(), min(), max(), count(), nunique(), std(), var(), value_counts(), len(df), df.shape[0], or df.groupby('dimension')['metric'].sum().idxmax().
- Chart pandas_grouping should prefer supported safe forms: df.groupby('dimension')['metric'].sum(), df.groupby('dimension')['metric'].mean(), df['category'].value_counts().head(10), df[['metric_a', 'metric_b']].sum(), pd.crosstab(df['category_a'], df['category_b']), df.plot.scatter(x='metric_a', y='metric_b').
- Titles and descriptions must sound executive-ready, not like debug logs.
"""



    llm_payload = await generate_json_response(SYSTEM_PROMPT, user_prompt)



    meta = {

        "rag": {

            "retrieved_chunk_ids": [c.chunk_id for c in chunks],

            "retrieval_query_preview": query[:500],

        }

    }



    return llm_payload, meta





async def run_rag_insights_and_save(

    db: AsyncSession,

    dataset_id: int,

    user_id: Optional[str] = None,

    debug: bool = False,

) -> Dashboard:

    """
    Run the full RAG pipeline and persist the result as a Dashboard row.

    Returns the newly created Dashboard ORM object.
    """

    llm_payload, meta = await run_rag_insights(db, dataset_id)





    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)

    analysis_id = analysis_row.id if analysis_row else None



    dashboard = await DashboardRepository.create(

        db,

        user_id=user_id,

        dataset_id=dataset_id,

        analysis_id=analysis_id,

        executive_summary_kpis=llm_payload.get("executive_summary_kpis", []),

        dashboard_charts=llm_payload.get("dashboard_charts", []),

        rag_meta=meta.get("rag") if debug else None,

    )

    await db.commit()

    await db.refresh(dashboard)

    return dashboard

