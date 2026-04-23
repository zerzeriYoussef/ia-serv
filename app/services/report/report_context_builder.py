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
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.dataset_repository import DatasetRepository
from app.services.analysis.kpi_service import build_kpi_block
from app.services.data.parser_service import ParserService
from app.services.rag.dataset_indexer import embed_query, _get_chroma_client, _collection_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain Chroma retriever (wraps the existing Chroma collection)
# ---------------------------------------------------------------------------

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

    # Point at the same PersistentClient already used by the indexer
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
        # run_in_executor because LangChain retrievers are sync
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
        # Fallback: use the existing raw retrieve_chunks function
        from app.services.rag.dataset_indexer import retrieve_chunks
        return await retrieve_chunks(dataset_id, query, top_k=top_k)


# ---------------------------------------------------------------------------
# Public: build the full internal context dict
# ---------------------------------------------------------------------------

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
    # ── Load DB rows ────────────────────────────────────────────────────────
    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)
    if not analysis_row:
        raise ValueError(f"No analysis found for dataset {dataset_id}. Run /analyze first.")

    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found.")

    # ── Load DataFrame + KPI block ──────────────────────────────────────────
    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
    kpi_block = build_kpi_block(
        df,
        analysis_row.primary_metric,
        analysis_row.temporal,
        analysis_row.dimensions,
    )

    # ── 5 targeted LangChain RAG retrievals ─────────────────────────────────
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

    # ── Assemble context dict ────────────────────────────────────────────────
    context: Dict[str, Any] = {
        "dataset_name":     dataset.original_filename,
        "dataset_id":       dataset_id,
        "row_count":        dataset.row_count or len(df),
        "primary_metric":   analysis_row.primary_metric,
        "metrics":          analysis_row.metrics or [],
        "dimensions":       analysis_row.dimensions or [],
        "temporal":         analysis_row.temporal or [],
        "geographic":       analysis_row.geographic or [],
        "relationships":    (analysis_row.relationships or [])[:20],
        "column_types":     dataset.column_types or {},
        "kpi_block":        kpi_block,
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
