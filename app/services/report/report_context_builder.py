"""
Report Context Builder — LangChain RAG over the dataset's Chroma collection.

Performs 5 targeted semantic retrievals (one per report section topic) using
LangChain's Chroma vectorstore interface + the existing Gemini embeddings.

Each retrieval is narrowed to the most relevant chunk_types so we get:
  • KPI / metric definitions  → key_metrics section
  • Relationships             → relationships_discovered section
  • Column facts              → insights_by_dimension section
  • Guardrails               → prompt safety
  • Full broad sweep          → executive_summary
"""



from __future__ import annotations



import logging

import math

from typing import Any, Dict, List, Optional, Tuple



import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession



from app.core.config import settings

from app.repositories.analysis_repository import AnalysisRepository

from app.repositories.dataset_repository import DatasetRepository

from app.services.analysis.kpi_service import build_kpi_block

from app.services.data.parser_service import ParserService

from app.services.rag.dataset_indexer import embed_query, _get_chroma_client, _collection_name



logger = logging.getLogger(__name__)





def _clean_value(value: Any) -> Any:

    if value is None:

        return None

    try:

        if pd.isna(value):

            return None

    except Exception:

        pass

    if hasattr(value, "item"):

        try:

            value = value.item()

        except Exception:

            pass

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):

            return None

        return round(value, 4)

    return value





def _fmt_num(value: Any) -> str:

    value = _clean_value(value)

    if value is None:

        return "n/a"

    if isinstance(value, (int, float)):

        if abs(value) >= 1000:

            return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"

        return f"{value:.2f}".rstrip("0").rstrip(".")

    return str(value)





def _series_numeric(df: pd.DataFrame, column: str) -> pd.Series:

    if column not in df.columns:

        return pd.Series(dtype="float64")

    return pd.to_numeric(df[column], errors="coerce").dropna()





def _pick_temporal_column(df: pd.DataFrame, temporal_columns: Optional[List[str]]) -> Optional[str]:

    candidates = [c for c in (temporal_columns or []) if c in df.columns]

    candidates += [

        c for c in df.columns

        if c not in candidates and any(token in c.lower() for token in ("date", "year", "month", "time"))

    ]

    for col in candidates:

        numeric = pd.to_numeric(df[col], errors="coerce")

        if numeric.notna().mean() >= 0.8 and numeric.between(1000, 3000).mean() >= 0.8:

            return col

        parsed = pd.to_datetime(df[col], errors="coerce")

        if parsed.notna().mean() >= 0.7:

            return col

    return None





def _sort_by_temporal(df: pd.DataFrame, temporal_col: str) -> pd.DataFrame:

    numeric = pd.to_numeric(df[temporal_col], errors="coerce")

    if numeric.notna().mean() >= 0.8:

        return df.assign(_report_time=numeric).dropna(subset=["_report_time"]).sort_values("_report_time")

    parsed = pd.to_datetime(df[temporal_col], errors="coerce")

    return df.assign(_report_time=parsed).dropna(subset=["_report_time"]).sort_values("_report_time")





def _safe_row_label(row: pd.Series, temporal_col: Optional[str], dimension_columns: List[str]) -> str:

    if temporal_col and temporal_col in row:

        val = _clean_value(row[temporal_col])

        if val is not None:

            return f"{temporal_col}={val}"

    for col in dimension_columns:

        if col in row:

            val = _clean_value(row[col])

            if val is not None:

                return f"{col}={val}"

    return "selected row"





def _build_metric_facts(

    df: pd.DataFrame,

    metrics: List[str],

    primary_metric: Optional[str],

    temporal_col: Optional[str],

    dimension_columns: List[str],

) -> List[Dict[str, Any]]:

    ordered_metrics = []

    if primary_metric:

        ordered_metrics.append(primary_metric)

    ordered_metrics.extend([m for m in metrics if m not in ordered_metrics])



    facts: List[Dict[str, Any]] = []

    for metric in ordered_metrics[:5]:

        values = _series_numeric(df, metric)

        if values.empty:

            continue

        max_idx = values.idxmax()

        min_idx = values.idxmin()

        max_row = df.loc[max_idx]

        min_row = df.loc[min_idx]

        fact = {

            "metric": metric,

            "total": _clean_value(values.sum()),

            "average": _clean_value(values.mean()),

            "minimum": {

                "value": _clean_value(values.min()),

                "where": _safe_row_label(min_row, temporal_col, dimension_columns),

            },

            "maximum": {

                "value": _clean_value(values.max()),

                "where": _safe_row_label(max_row, temporal_col, dimension_columns),

            },

            "non_null_rows": int(values.count()),

        }

        facts.append(fact)

    return facts





def _build_trend_fact(df: pd.DataFrame, metric: Optional[str], temporal_col: Optional[str]) -> Optional[Dict[str, Any]]:

    if not metric or metric not in df.columns or not temporal_col or temporal_col not in df.columns:

        return None

    sorted_df = _sort_by_temporal(df[[temporal_col, metric]].copy(), temporal_col)

    sorted_df[metric] = pd.to_numeric(sorted_df[metric], errors="coerce")

    sorted_df = sorted_df.dropna(subset=[metric])

    if len(sorted_df) < 2:

        return None

    first = sorted_df.iloc[0]

    last = sorted_df.iloc[-1]

    change = float(last[metric] - first[metric])

    pct_change = (change / float(first[metric]) * 100) if float(first[metric]) else None

    max_row = sorted_df.loc[sorted_df[metric].idxmax()]

    min_row = sorted_df.loc[sorted_df[metric].idxmin()]

    return {

        "time_column": temporal_col,

        "metric": metric,

        "first_period": _clean_value(first[temporal_col]),

        "first_value": _clean_value(first[metric]),

        "last_period": _clean_value(last[temporal_col]),

        "last_value": _clean_value(last[metric]),

        "absolute_change": _clean_value(change),

        "percent_change": _clean_value(pct_change),

        "peak_period": _clean_value(max_row[temporal_col]),

        "peak_value": _clean_value(max_row[metric]),

        "low_period": _clean_value(min_row[temporal_col]),

        "low_value": _clean_value(min_row[metric]),

    }





def _build_dimension_facts(

    df: pd.DataFrame,

    metric: Optional[str],

    dimension_columns: List[str],

    top_n: int = 5,

) -> List[Dict[str, Any]]:

    facts: List[Dict[str, Any]] = []

    for dim in dimension_columns[:5]:

        if dim not in df.columns:

            continue

        entry: Dict[str, Any] = {"dimension": dim}

        counts = df[dim].dropna().astype(str).value_counts().head(top_n)

        if not counts.empty:

            entry["top_values_by_count"] = [

                {"value": key, "count": int(value)} for key, value in counts.items()

            ]

        if metric and metric in df.columns:

            tmp = pd.DataFrame({

                "dim": df[dim].astype(str),

                "metric": pd.to_numeric(df[metric], errors="coerce"),

            }).dropna()

            if not tmp.empty:

                grouped = tmp.groupby("dim", observed=False)["metric"].sum().sort_values(ascending=False).head(top_n)

                entry["top_values_by_metric_total"] = [

                    {"value": key, "metric_total": _clean_value(value)}

                    for key, value in grouped.items()

                ]

        if len(entry) > 1:

            facts.append(entry)

    return facts





def _build_quality_facts(df: pd.DataFrame) -> Dict[str, Any]:

    missing_by_col = df.isna().sum().sort_values(ascending=False)

    missing_total = int(missing_by_col.sum())

    top_missing = [

        {"column": col, "missing": int(count)}

        for col, count in missing_by_col.head(5).items()

        if int(count) > 0

    ]

    return {

        "row_count": int(len(df)),

        "column_count": int(len(df.columns)),

        "duplicate_rows": int(df.duplicated().sum()),

        "missing_cells": missing_total,

        "top_missing_columns": top_missing,

    }





def _relationship_facts(relationships: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:

    def strength(rel: Dict[str, Any]) -> float:

        raw = rel.get("strength", rel.get("strength_score", 0))

        try:

            return abs(float(raw))

        except Exception:

            return 0.0



    sorted_rels = sorted(relationships or [], key=strength, reverse=True)

    facts = []

    for rel in sorted_rels[:limit]:

        cols = rel.get("columns") or []

        facts.append({

            "columns": [str(c) for c in cols],

            "type": rel.get("type", "relationship"),

            "strength": _clean_value(rel.get("strength", rel.get("strength_score"))),

            "direction": rel.get("direction"),

            "method": rel.get("method"),

            "insight": rel.get("insight"),

        })

    return facts





def _build_fact_pack(

    df: pd.DataFrame,

    dataset_name: str,

    primary_metric: Optional[str],

    metrics: List[str],

    dimensions: List[str],

    temporal_columns: List[str],

    relationships: List[Dict[str, Any]],

) -> Dict[str, Any]:

    temporal_col = _pick_temporal_column(df, temporal_columns)

    return {

        "dataset": {

            "name": dataset_name,

            "rows": int(len(df)),

            "columns": int(len(df.columns)),

            "column_names": [str(c) for c in df.columns],

            "primary_metric": primary_metric,

        },

        "quality": _build_quality_facts(df),

        "metric_facts": _build_metric_facts(df, metrics, primary_metric, temporal_col, dimensions),

        "trend": _build_trend_fact(df, primary_metric, temporal_col),

        "dimension_facts": _build_dimension_facts(df, primary_metric, dimensions),

        "relationships": _relationship_facts(relationships),

    }





def format_fact_pack(fact_pack: Dict[str, Any]) -> str:

    lines: List[str] = []

    dataset = fact_pack.get("dataset") or {}

    quality = fact_pack.get("quality") or {}

    lines.append(

        "Dataset facts: "

        f"{dataset.get('name', 'dataset')} has {quality.get('row_count', 0)} rows "

        f"and {quality.get('column_count', 0)} columns. "

        f"Primary metric: {dataset.get('primary_metric') or 'not detected'}."

    )

    lines.append(

        f"Data quality: {quality.get('duplicate_rows', 0)} duplicate rows, "

        f"{quality.get('missing_cells', 0)} missing cells."

    )



    for item in fact_pack.get("metric_facts") or []:

        lines.append(

            f"Metric {item['metric']}: total {_fmt_num(item.get('total'))}, "

            f"average {_fmt_num(item.get('average'))}, "

            f"min {_fmt_num((item.get('minimum') or {}).get('value'))} at {(item.get('minimum') or {}).get('where')}, "

            f"max {_fmt_num((item.get('maximum') or {}).get('value'))} at {(item.get('maximum') or {}).get('where')}."

        )



    trend = fact_pack.get("trend")

    if trend:

        lines.append(

            f"Trend for {trend.get('metric')} by {trend.get('time_column')}: "

            f"{trend.get('first_period')} was {_fmt_num(trend.get('first_value'))}, "

            f"{trend.get('last_period')} was {_fmt_num(trend.get('last_value'))}, "

            f"change {_fmt_num(trend.get('absolute_change'))} "

            f"({_fmt_num(trend.get('percent_change'))}%). "

            f"Peak: {trend.get('peak_period')} at {_fmt_num(trend.get('peak_value'))}; "

            f"low: {trend.get('low_period')} at {_fmt_num(trend.get('low_value'))}."

        )



    for dim in fact_pack.get("dimension_facts") or []:

        count_bits = [

            f"{item['value']} ({item['count']})"

            for item in dim.get("top_values_by_count", [])[:3]

        ]

        metric_bits = [

            f"{item['value']} ({_fmt_num(item['metric_total'])})"

            for item in dim.get("top_values_by_metric_total", [])[:3]

        ]

        if count_bits:

            lines.append(f"Top {dim['dimension']} values by count: {', '.join(count_bits)}.")

        if metric_bits:

            lines.append(f"Top {dim['dimension']} values by metric total: {', '.join(metric_bits)}.")



    for rel in fact_pack.get("relationships") or []:

        cols = " <-> ".join(rel.get("columns") or [])

        if cols:

            lines.append(

                f"Relationship {cols}: type {rel.get('type')}, "

                f"strength {_fmt_num(rel.get('strength'))}, method {rel.get('method') or 'n/a'}."

            )

    return "\n".join(lines)











def _get_lc_vectorstore(dataset_id: int):

    """
    Return a LangChain Chroma vectorstore that wraps the existing
    PersistentClient collection — no re-indexing needed.
    """

    from langchain_chroma import Chroma

    from langchain_google_genai import GoogleGenerativeAIEmbeddings



    embeddings = GoogleGenerativeAIEmbeddings(

        model=settings.GEMINI_EMBEDDING_MODEL,

        google_api_key=settings.GEMINI_API_KEY,

        task_type="retrieval_query",

    )





    chroma_client = _get_chroma_client()



    vectorstore = Chroma(

        client=chroma_client,

        collection_name=_collection_name(dataset_id),

        embedding_function=embeddings,

    )

    return vectorstore





async def _retrieve_by_topic(

    dataset_id: int,

    query: str,

    chunk_type_filter: Optional[str],

    top_k: int = 5,

) -> List[Dict[str, Any]]:

    """
    Semantic search over Chroma with optional metadata filtering.
    Falls back to raw Chroma client if LangChain vectorstore fails.
    """

    try:

        vectorstore = _get_lc_vectorstore(dataset_id)

        where_filter = (

            {"chunk_type": {"$eq": chunk_type_filter}} if chunk_type_filter else None

        )

        retriever = vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs={

                "k": top_k,

                **({"filter": where_filter} if where_filter else {}),

            },

        )



        import asyncio

        docs = await asyncio.get_event_loop().run_in_executor(

            None, retriever.invoke, query

        )

        return [

            {

                "chunk_id": doc.metadata.get("chunk_type", "unknown"),

                "text": doc.page_content,

                "metadata": doc.metadata,

            }

            for doc in docs

        ]

    except Exception as exc:

        logger.warning(

            "report_context: LangChain retrieval failed (%s) — falling back to raw Chroma", exc

        )



        from app.services.rag.dataset_indexer import retrieve_chunks

        return await retrieve_chunks(dataset_id, query, top_k=top_k)













async def build_internal_context(

    db: AsyncSession,

    dataset_id: int,

) -> Tuple[Dict[str, Any], Dict[str, str]]:

    """
    Gather all internal context needed for report generation.

    Returns:
        (context_dict, retrieval_chunks_by_topic)

    context_dict keys:
        dataset_name, row_count, primary_metric, metrics, dimensions,
        temporal, geographic, relationships, kpi_block,
        analysis_id, column_types

    retrieval_chunks_by_topic:
        {topic_name: formatted_text_block}
    """



    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)

    if not analysis_row:

        raise ValueError(f"No analysis found for dataset {dataset_id}. Run /analyze first.")



    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise ValueError(f"Dataset {dataset_id} not found.")





    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)

    relationships = analysis_row.relationships or []

    kpi_block = build_kpi_block(

        df,

        analysis_row.primary_metric,

        analysis_row.temporal,

        analysis_row.dimensions,

    )

    fact_pack = _build_fact_pack(

        df=df,

        dataset_name=dataset.original_filename,

        primary_metric=analysis_row.primary_metric,

        metrics=analysis_row.metrics or [],

        dimensions=analysis_row.dimensions or [],

        temporal_columns=analysis_row.temporal or [],

        relationships=relationships,

    )

    fact_pack_text = format_fact_pack(fact_pack)





    primary = analysis_row.primary_metric or "performance"

    dims    = (analysis_row.dimensions or [])[:3]

    metrics = (analysis_row.metrics or [])[:5]



    retrieval_tasks = {

        "executive":      (_retrieve_by_topic, dataset_id, f"overall {primary} performance summary", None, 6),

        "kpi_metrics":    (_retrieve_by_topic, dataset_id, f"{primary} KPI definitions aggregations", "metric_definition", 4),

        "relationships":  (_retrieve_by_topic, dataset_id, f"correlation relationship between {' '.join(metrics)}", "relationship", 6),

        "dimensions":     (_retrieve_by_topic, dataset_id, f"dimension analysis {' '.join(dims)}", "column_fact", 5),

        "guardrails":     (_retrieve_by_topic, dataset_id, "data quality guardrails methodology", "guardrail", 2),

    }



    import asyncio

    results: Dict[str, List[Dict]] = {}

    for topic, (fn, *args) in retrieval_tasks.items():

        try:

            results[topic] = await fn(*args)

        except Exception as exc:

            logger.warning("report_context: retrieval '%s' failed: %s", topic, exc)

            results[topic] = []



    def _fmt(chunks: List[Dict]) -> str:

        if not chunks:

            return "No relevant context retrieved."

        return "\n\n---\n\n".join(

            f"[{c.get('chunk_id', 'chunk')}]\n{c['text']}" for c in chunks

        )



    chunks_by_topic = {topic: _fmt(chunks) for topic, chunks in results.items()}





    context: Dict[str, Any] = {

        "dataset_name":     dataset.original_filename,

        "dataset_id":       dataset_id,

        "row_count":        dataset.row_count or len(df),

        "primary_metric":   analysis_row.primary_metric,

        "metrics":          analysis_row.metrics or [],

        "dimensions":       analysis_row.dimensions or [],

        "temporal":         analysis_row.temporal or [],

        "geographic":       analysis_row.geographic or [],

        "relationships":    relationships[:20],

        "column_types":     dataset.column_types or {},

        "kpi_block":        kpi_block,

        "fact_pack":        fact_pack,

        "fact_pack_text":   fact_pack_text,

        "analysis_id":      analysis_row.id,

    }



    logger.info(

        "report_context: loaded context for dataset=%s | "

        "metrics=%d dims=%d rels=%d chunks_topics=%d",

        dataset_id,

        len(context["metrics"]),

        len(context["dimensions"]),

        len(context["relationships"]),

        len(chunks_by_topic),

    )



    return context, chunks_by_topic

