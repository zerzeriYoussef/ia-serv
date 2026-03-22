"""
API routes for column analysis
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
import logging

from app.core.database import get_db
from app.services.analysis.column_analyzer import ColumnAnalyzer
from app.services.analysis.relationship_detector import RelationshipDetector
from app.services.data.parser_service import ParserService
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.dataset_repository import DatasetRepository
from app.api.v1.schemas.analysis_schema import (
    AnalysisResultSchema,
    SimpleRelationshipResponse,
    AnalysisRequest,
    StatisticalAnalyticsRequest,
    StatisticalAnalyticsResponseSchema,
)
from app.services.analysis.statistical_relationship_service import (
    run_multi_dataset_analytics,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analysis/statistical-relationships",
    response_model=StatisticalAnalyticsResponseSchema,
    summary="Statistical relationships for one or more datasets",
)
async def statistical_relationships_analytics(
    body: StatisticalAnalyticsRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run analytics similar to the Spearman / Kruskal-Wallis+η² / Chi-square+Cramér's V script:

    - For each dataset: detect numeric↔numeric, categorical→numeric, categorical↔categorical
      relationships with multiple-testing-aware alpha and junk-column filtering.
    - If several datasets are provided: attempt sequential **outer** merge on **all**
      common column names, then run the same detection on the merged table.
    """
    seen: set[int] = set()
    ordered_ids: List[int] = []
    for raw_id in body.dataset_ids:
        if raw_id not in seen:
            seen.add(raw_id)
            ordered_ids.append(raw_id)

    loaded: List[tuple[int, str, Any]] = []
    for dataset_id in ordered_ids:
        dataset = await DatasetRepository.get_by_id(db, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found",
            )
        df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
        label = dataset.original_filename or f"dataset_{dataset_id}"
        loaded.append((dataset_id, label, df))

    result = run_multi_dataset_analytics(loaded)
    return result


def _normalize_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each relationship has a 'method' field.

    This is important for older analyses that were stored before
    the 'method' attribute was added to the schema.
    """
    normalized: List[Dict[str, Any]] = []

    for rel in relationships or []:
        rel_copy = dict(rel)  # avoid mutating stored objects
        if "method" not in rel_copy or rel_copy["method"] is None:
            rel_type = rel_copy.get("type")
            if rel_type == "hierarchy":
                rel_copy["method"] = "hierarchy_detector"
            elif rel_type == "predictive":
                rel_copy["method"] = "custom_pps"
            elif rel_type == "correlation":
                rel_copy["method"] = "pearson"
            else:
                rel_copy["method"] = "unknown"
        normalized.append(rel_copy)

    return normalized


@router.post("/datasets/{dataset_id}/analyze", response_model=AnalysisResultSchema)
async def analyze_dataset(
    dataset_id: int,
    request: AnalysisRequest = AnalysisRequest(),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze dataset columns and detect relationships
    
    Uses:
    - CustomPPS for relationship detection (modern, no conflicts!)
    - Pandas for correlations
    - Smart heuristics for categorization
    
    Returns complete analysis including:
    - Column categorization
    - Detected relationships
    - Dashboard recommendations
    - Primary metric
    """
    
    # Get dataset
    dataset = await DatasetRepository.get_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    
    # Check if recent analysis exists
    if not request.force_reanalyze:
        existing = await AnalysisRepository.get_by_dataset(db, dataset_id)
        if existing:
            logger.info(f"Returning existing analysis for dataset {dataset_id}")
            return {
                "dataset_id": existing.dataset_id,
                "relationships": _normalize_relationships(existing.relationships),
                "dashboard_columns": existing.dashboard_columns,
                "column_categories": {
                    "metrics": existing.metrics,
                    "dimensions": existing.dimensions,
                    "identifiers": existing.identifiers,
                    "temporal": existing.temporal,
                    "geographic": existing.geographic,
                    "other": existing.other or []
                },
                "primary_metric": existing.primary_metric,
                "confidence_score": existing.confidence_score,
                "total_relationships": existing.total_relationships,
                "created_at": existing.created_at
            }
    
    # Load data
    logger.info(f"Loading dataset {dataset_id} for analysis...")
    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)
    
    # Run analysis
    analyzer = ColumnAnalyzer(df)
    results = analyzer.analyze()
    
    # Detect additional patterns
    hierarchies = RelationshipDetector.detect_hierarchies(df)
    results["relationships"].extend(hierarchies)
    results["relationships"] = _normalize_relationships(results["relationships"])
    
    # Save to database
    analysis = await AnalysisRepository.create(
        db,
        dataset_id=dataset_id,
        metrics=results["column_categories"]["metrics"],
        dimensions=results["column_categories"]["dimensions"],
        identifiers=results["column_categories"]["identifiers"],
        temporal=results["column_categories"]["temporal"],
        geographic=results["column_categories"]["geographic"],
        other=results["column_categories"]["other"],
        relationships=results["relationships"],
        dashboard_columns=results["dashboard_columns"],
        primary_metric=results["primary_metric"],
        confidence_score=results["confidence_score"],
        total_relationships=len(results["relationships"]),
        analysis_method="custom_pps"
    )
    
    await db.commit()
    
    logger.info(f"Analysis saved for dataset {dataset_id}: {len(results['relationships'])} relationships")
    
    return {
        "dataset_id": dataset_id,
        "relationships": _normalize_relationships(results["relationships"]),
        "dashboard_columns": results["dashboard_columns"],
        "column_categories": results["column_categories"],
        "primary_metric": results["primary_metric"],
        "confidence_score": results["confidence_score"],
        "total_relationships": len(results["relationships"]),
        "created_at": analysis.created_at
    }


@router.get("/datasets/{dataset_id}/analysis", response_model=AnalysisResultSchema)
async def get_analysis(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get existing analysis for dataset"""
    
    analysis = await AnalysisRepository.get_by_dataset(db, dataset_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found for dataset {dataset_id}. Run POST /datasets/{dataset_id}/analyze first."
        )
    
    return {
        "dataset_id": analysis.dataset_id,
        "relationships": _normalize_relationships(analysis.relationships),
        "dashboard_columns": analysis.dashboard_columns,
        "column_categories": {
            "metrics": analysis.metrics,
            "dimensions": analysis.dimensions,
            "identifiers": analysis.identifiers,
            "temporal": analysis.temporal,
            "geographic": analysis.geographic,
            "other": analysis.other or []
        },
        "primary_metric": analysis.primary_metric,
        "confidence_score": analysis.confidence_score,
        "total_relationships": analysis.total_relationships,
        "created_at": analysis.created_at
    }


@router.get("/datasets/{dataset_id}/relationships", response_model=SimpleRelationshipResponse)
async def get_relationships_simple(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get relationships in simple format for chart generation (Features 6-7)
    
    Returns EXACTLY the format you requested:
    {
      "relationships": [["ads", "sales"], ["region", "sales"]],
      "dashboard_columns": ["sales", "product", "region"]
    }
    """
    
    analysis = await AnalysisRepository.get_by_dataset(db, dataset_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found. Run POST /datasets/{dataset_id}/analyze first."
        )
    
    # Simplify to just column pairs
    simple_relationships = [
        rel["columns"] for rel in analysis.relationships
    ]
    
    return {
        "relationships": simple_relationships,
        "dashboard_columns": analysis.dashboard_columns
    }


@router.get("/datasets/{dataset_id}/column-categories")
async def get_column_categories(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get detailed column categorization"""
    
    analysis = await AnalysisRepository.get_by_dataset(db, dataset_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found. Run POST /datasets/{dataset_id}/analyze first."
        )
    
    return {
        "metrics": analysis.metrics,
        "dimensions": analysis.dimensions,
        "identifiers": analysis.identifiers,
        "temporal": analysis.temporal,
        "geographic": analysis.geographic,
        "other": analysis.other or [],
        "primary_metric": analysis.primary_metric
    }


@router.delete("/datasets/{dataset_id}/analysis", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    dataset_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete analysis for a dataset"""
    
    deleted = await AnalysisRepository.delete(db, dataset_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found for dataset {dataset_id}"
        )