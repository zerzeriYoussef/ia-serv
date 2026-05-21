"""
Dataset Indexer — builds and maintains the Chroma vector store for a dataset.

Called automatically after analysis completes (background task) and manually
via POST /datasets/{dataset_id}/index (admin).

Chunk types:
  column_fact        – one chunk per column (name, dtype, null_rate, cardinality hint)
  relationship       – one chunk per detected relationship
  metric_definition  – summary of primary metric + KPIs
  guardrail          – static rules the model must follow
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.dataset_repository import DatasetRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chroma client (lazy singleton)
# ---------------------------------------------------------------------------

_chroma_client = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    return _chroma_client


def _collection_name(dataset_id: int) -> str:
    return f"dataset_{dataset_id}"


# ---------------------------------------------------------------------------
# Gemini embedding helper
# ---------------------------------------------------------------------------

async def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed texts using Gemini embedContent REST endpoint."""
    key = settings.GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured — cannot embed chunks")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{settings.GEMINI_EMBEDDING_MODEL}:batchEmbedContents"
    )

    model_name = settings.GEMINI_EMBEDDING_MODEL
    requests_payload = [
        {
            "model": model_name,
            "content": {"parts": [{"text": t}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        }
        for t in texts
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            params={"key": key},
            json={"requests": requests_payload},
        )
        resp.raise_for_status()

    data = resp.json()
    return [item["values"] for item in data.get("embeddings", [])]


async def embed_query(query: str) -> List[float]:
    """Embed a single query string for retrieval."""
    key = settings.GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"{settings.GEMINI_EMBEDDING_MODEL}:embedContent"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            params={"key": key},
            json={
                "content": {"parts": [{"text": query}]},
                "taskType": "RETRIEVAL_QUERY",
            },
        )
        resp.raise_for_status()

    return resp.json()["embedding"]["values"]


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def _make_source_version(analysis_id: int, updated_at: Any) -> str:
    return hashlib.sha1(f"{analysis_id}:{updated_at}".encode()).hexdigest()[:12]


def _column_fact_chunks(
    dataset_id: int,
    source_version: str,
    columns: List[str],
    column_types: Optional[Dict[str, str]],
    summary_stats: Optional[Dict[str, Any]],
) -> tuple[List[str], List[str], List[Dict]]:
    """One chunk per column describing name, dtype, and basic statistics."""
    ids, texts, metas = [], [], []
    col_types = column_types or {}
    stats = summary_stats or {}

    for col in columns:
        dtype = col_types.get(col, "unknown")
        col_stats = stats.get(col, {})
        null_rate = col_stats.get("null_pct", col_stats.get("missing_pct", "?"))
        unique_count = col_stats.get("unique", col_stats.get("nunique", "?"))

        text = (
            f"Column: {col}\n"
            f"Type: {dtype}\n"
            f"Null rate: {null_rate}\n"
            f"Unique values: {unique_count}"
        )
        chunk_id = f"ds{dataset_id}_col_{col}_{source_version}"
        ids.append(chunk_id)
        texts.append(text)
        metas.append({
            "dataset_id": dataset_id,
            "chunk_type": "column_fact",
            "source_version": source_version,
            "column": col,
        })
    return ids, texts, metas


def _relationship_chunks(
    dataset_id: int,
    source_version: str,
    relationships: List[Dict[str, Any]],
) -> tuple[List[str], List[str], List[Dict]]:
    ids, texts, metas = [], [], []
    for i, rel in enumerate(relationships or []):
        cols = rel.get("columns") or []
        cols_str = " ↔ ".join(str(c) for c in cols)
        text = (
            f"Relationship: {cols_str}\n"
            f"Type: {rel.get('type', 'unknown')}\n"
            f"Strength: {rel.get('strength', '?')}\n"
            f"Method: {rel.get('method', 'unknown')}"
        )
        chunk_id = f"ds{dataset_id}_rel_{i}_{source_version}"
        ids.append(chunk_id)
        texts.append(text)
        metas.append({
            "dataset_id": dataset_id,
            "chunk_type": "relationship",
            "source_version": source_version,
        })
    return ids, texts, metas


def _metric_definition_chunks(
    dataset_id: int,
    source_version: str,
    analysis_row: Any,
) -> tuple[List[str], List[str], List[Dict]]:
    ids, texts, metas = [], [], []

    metrics = analysis_row.metrics or []
    dimensions = analysis_row.dimensions or []
    temporal = analysis_row.temporal or []
    primary = analysis_row.primary_metric or "N/A"

    text = (
        f"Primary metric: {primary}\n"
        f"All metrics: {', '.join(metrics)}\n"
        f"Dimensions: {', '.join(dimensions)}\n"
        f"Temporal columns: {', '.join(temporal)}"
    )
    chunk_id = f"ds{dataset_id}_metrics_{source_version}"
    ids.append(chunk_id)
    texts.append(text)
    metas.append({
        "dataset_id": dataset_id,
        "chunk_type": "metric_definition",
        "source_version": source_version,
    })
    return ids, texts, metas


def _guardrail_chunks(dataset_id: int, source_version: str) -> tuple[List[str], List[str], List[Dict]]:
    text = (
        "Guardrails:\n"
        "- Do not infer causation from correlation.\n"
        "- Ignore ID, email, and UUID columns in analysis.\n"
        "- Do not execute arbitrary code — only approved tool operations.\n"
        "- Do not reveal raw PII values from the dataset.\n"
        "- Cite only columns and facts present in context or tool output."
    )
    chunk_id = f"ds{dataset_id}_guardrails_{source_version}"
    return [chunk_id], [text], [{
        "dataset_id": dataset_id,
        "chunk_type": "guardrail",
        "source_version": source_version,
    }]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def delete_dataset_index(dataset_id: int) -> None:
    """Remove the Chroma collection for a dataset (best-effort, non-fatal)."""
    try:
        client = _get_chroma_client()
        client.delete_collection(_collection_name(dataset_id))
        logger.info("Deleted Chroma collection for dataset %s", dataset_id)
    except Exception as exc:
        logger.warning(
            "Chroma collection delete for dataset %s (non-fatal): %s",
            dataset_id,
            exc,
        )


async def index_dataset(db: AsyncSession, dataset_id: int) -> int:
    """
    Build or refresh the Chroma collection for a dataset.

    Returns the total number of chunks upserted.
    Raises ValueError if no analysis exists for the dataset.
    """
    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)
    if not analysis_row:
        raise ValueError(f"No analysis found for dataset {dataset_id}. Run /analyze first.")

    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found.")

    source_version = _make_source_version(analysis_row.id, analysis_row.updated_at or analysis_row.created_at)
    columns: List[str] = dataset.columns or []
    column_types: Dict = dataset.column_types or {}
    summary_stats: Dict = dataset.summary_stats or {}
    relationships: List = analysis_row.relationships or []

    # Build all chunks
    all_ids, all_texts, all_metas = [], [], []

    ids, txts, mts = _column_fact_chunks(dataset_id, source_version, columns, column_types, summary_stats)
    all_ids += ids; all_texts += txts; all_metas += mts

    ids, txts, mts = _relationship_chunks(dataset_id, source_version, relationships[:50])
    all_ids += ids; all_texts += txts; all_metas += mts

    ids, txts, mts = _metric_definition_chunks(dataset_id, source_version, analysis_row)
    all_ids += ids; all_texts += txts; all_metas += mts

    ids, txts, mts = _guardrail_chunks(dataset_id, source_version)
    all_ids += ids; all_texts += txts; all_metas += mts

    if not all_ids:
        logger.warning("dataset_indexer: no chunks to index for dataset %s", dataset_id)
        return 0

    # Embed in a single batch call
    logger.info("dataset_indexer: embedding %d chunks for dataset %s", len(all_texts), dataset_id)
    embeddings = await _embed_texts(all_texts)

    # Upsert into Chroma
    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=_collection_name(dataset_id),
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metas,
    )

    logger.info(
        "dataset_indexer: upserted %d chunks into collection '%s'",
        len(all_ids),
        _collection_name(dataset_id),
    )
    return len(all_ids)


async def retrieve_chunks(
    dataset_id: int,
    query: str,
    *,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Semantic retrieval from Chroma for a dataset-scoped query.

    Returns list of dicts: {chunk_id, text, metadata, distance}
    Falls back to empty list if collection doesn't exist yet.
    """
    client = _get_chroma_client()
    try:
        collection = client.get_collection(_collection_name(dataset_id))
    except Exception:
        logger.warning("retrieve_chunks: no Chroma collection for dataset %s", dataset_id)
        return []

    query_embedding = await embed_query(query)

    # Fetch more than top_k for diversity-based deduplication
    fetch_k = min(top_k * 2, 30)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(fetch_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0] if results["ids"] else []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    # MMR-style diversity: at most 2 chunks per chunk_type
    seen_types: Dict[str, int] = {}
    diverse: List[Dict[str, Any]] = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        ctype = meta.get("chunk_type", "other")
        if seen_types.get(ctype, 0) >= 2:
            continue
        seen_types[ctype] = seen_types.get(ctype, 0) + 1
        diverse.append({"chunk_id": chunk_id, "text": doc, "metadata": meta, "distance": dist})
        if len(diverse) >= top_k:
            break

    return diverse
