from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

from sqlalchemy.ext.asyncio import AsyncSession

from typing import List, Optional

import pandas as pd

import numpy as np

import logging

import os

import shutil

from datetime import datetime, timezone



from app.core.database import get_db

from app.services.data.cleaning_service import CleaningService

from app.services.data.validation_service import ValidationService

from app.services.data.parser_service import ParserService

from app.services.data.scan_service import ScanService

from app.repositories.cleaning_repository import (

    CleaningProfileRepository,

    CleaningLogRepository

)

from app.repositories.dataset_repository import DatasetRepository

from app.api.v1.schemas.cleaning_schema import (

    CleaningProfileCreate,

    CleaningProfileResponse,

    CleanDatasetRequest,

    CleaningReportResponse,

    DataQualityResponse,

    OutlierApplyRequest,

)

from app.api.dependencies.auth import CurrentUser, require_auth, require_admin



logger = logging.getLogger(__name__)



router = APIRouter()





def _to_native_types(obj):

    """Recursively convert NumPy / pandas scalar types to native Python types."""

    if isinstance(obj, np.generic):

        return obj.item()

    if isinstance(obj, dict):

        return {k: _to_native_types(v) for k, v in obj.items()}

    if isinstance(obj, list):

        return [_to_native_types(v) for v in obj]

    if isinstance(obj, tuple):

        return tuple(_to_native_types(v) for v in obj)

    return obj









@router.post("/cleaning-profiles", response_model=CleaningProfileResponse)

async def create_cleaning_profile(

    profile: CleaningProfileCreate,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    """Create a new cleaning profile"""



    new_profile = await CleaningProfileRepository.create(

        db,

        **profile.dict()

    )



    logger.info(f"Created cleaning profile: {new_profile.name}")



    return new_profile





@router.get("/cleaning-profiles", response_model=List[CleaningProfileResponse])

async def list_cleaning_profiles(

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    """List all cleaning profiles"""

    profiles = await CleaningProfileRepository.get_all(db)

    return profiles





@router.get("/cleaning-profiles/{profile_id}", response_model=CleaningProfileResponse)

async def get_cleaning_profile(

    profile_id: int,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    """Get cleaning profile by ID"""

    profile = await CleaningProfileRepository.get_by_id(db, profile_id)



    if not profile:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=f"Profil de nettoyage {profile_id} introuvable"

        )



    return profile





@router.delete("/cleaning-profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)

async def delete_cleaning_profile(

    profile_id: int,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    """Delete cleaning profile"""

    deleted = await CleaningProfileRepository.delete(db, profile_id)



    if not deleted:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=f"Profil de nettoyage {profile_id} introuvable"

        )









@router.post("/datasets/{dataset_id}/clean", response_model=CleaningReportResponse)

async def clean_dataset(

    dataset_id: int,

    request: CleanDatasetRequest,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    profile_id = request.profile_id

    save_as_new = request.save_as_new

    """
    Clean a dataset using a cleaning profile

    If no profile_id is provided, uses the default profile.
    If ``save_as_new`` is True (default), the cleaned data is written to a new
    dataset and the original file is left untouched. If False, the original
    file is overwritten in-place but a timestamped backup is created next to
    it and recorded on the cleaning log.
    """





    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=f"Jeu de données {dataset_id} introuvable"

        )





    if not current_user.is_admin and dataset.user_id != current_user.user_id:

        logger.warning(

            "Forbidden: user_id=%s tried to clean dataset %s owned by user_id=%s",

            current_user.user_id, dataset_id, dataset.user_id

        )

        raise HTTPException(

            status_code=status.HTTP_403_FORBIDDEN,

            detail="Vous n'avez pas accès à ce jeu de données"

        )





    if profile_id:

        profile = await CleaningProfileRepository.get_by_id(db, profile_id)

        if not profile:

            raise HTTPException(

                status_code=status.HTTP_404_NOT_FOUND,

                detail=f"Profil de nettoyage {profile_id} introuvable"

            )

    else:

        profile = await CleaningProfileRepository.get_default(db)

        if not profile:

            profile = await CleaningProfileRepository.create(

                db,

                name="Profil de nettoyage par défaut",

                description="Profil par défaut créé automatiquement",

                is_default=True,

                handle_missing="fill",

                missing_fill_strategy="auto",

                remove_duplicates=True,

                duplicate_subset=None,

                duplicate_keep="best",

                dedup_normalize_text=True,

                fix_data_types=True,

                numeric_coerce_min_rate=0.8,

                detect_outliers=True,

                outlier_method="auto",

                outlier_threshold=1.5,

                outlier_action="cap",

                outlier_max_rows=500_000,

                add_flag_columns=False,

                save_outliers_metadata=True,

                target_columns=None,

                strip_whitespace=True,

                standardize_text=True,

                standardize_dates=True,

                date_format="%Y-%m-%d",

                column_rules=None,

            )

        else:





            needs_upgrade = (

                getattr(profile, "is_default", False)

                and (

                    (

                        getattr(profile, "name", "") == "Default Cleaning Profile"

                        and getattr(profile, "description", "") == "Auto-created default profile"

                    )

                    or getattr(profile, "handle_missing", None) == "drop"

                    or getattr(profile, "missing_fill_strategy", None) == "mean"

                    or getattr(profile, "outlier_method", None) == "iqr"

                    or getattr(profile, "outlier_action", None) != "cap"

                    or getattr(profile, "duplicate_keep", None) is None

                    or getattr(profile, "add_flag_columns", None) is True

                    or getattr(profile, "save_outliers_metadata", None) is False

                )

            )

            if needs_upgrade:

                profile = await CleaningProfileRepository.update(

                    db,

                    profile.id,

                    handle_missing="fill",

                    missing_fill_strategy="auto",

                    strip_whitespace=True,

                    standardize_text=True,

                    standardize_dates=True,

                    date_format="%Y-%m-%d",

                    remove_duplicates=True,

                    duplicate_subset=None,

                    duplicate_keep="best",

                    dedup_normalize_text=True,

                    fix_data_types=True,

                    numeric_coerce_min_rate=0.8,

                    detect_outliers=True,

                    outlier_method="auto",

                    outlier_threshold=1.5,

                    outlier_action="cap",

                    outlier_max_rows=500_000,

                    add_flag_columns=False,

                    save_outliers_metadata=True,

                    target_columns=None,

                )





    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)



    profile_dict = {

        "handle_missing": profile.handle_missing,

        "missing_fill_strategy": profile.missing_fill_strategy,

        "missing_fill_value": profile.missing_fill_value,

        "remove_duplicates": profile.remove_duplicates,

        "duplicate_subset": profile.duplicate_subset,

        "duplicate_keep": getattr(profile, "duplicate_keep", "best") or "best",

        "dedup_normalize_text": (

            True if getattr(profile, "dedup_normalize_text", None) is None

            else bool(profile.dedup_normalize_text)

        ),

        "fix_data_types": profile.fix_data_types,

        "numeric_coerce_min_rate": (

            float(getattr(profile, "numeric_coerce_min_rate", 0.8) or 0.8)

        ),

        "detect_outliers": profile.detect_outliers,

        "outlier_method": profile.outlier_method,

        "outlier_threshold": profile.outlier_threshold,

        "outlier_action": profile.outlier_action,

        "outlier_max_rows": int(

            getattr(profile, "outlier_max_rows", 500_000) or 500_000

        ),

        "add_flag_columns": bool(getattr(profile, "add_flag_columns", False)),

        "save_outliers_metadata": bool(getattr(profile, "save_outliers_metadata", True)),

        "target_columns": getattr(profile, "target_columns", None),

        "strip_whitespace": profile.strip_whitespace,

        "standardize_text": profile.standardize_text,

        "standardize_dates": profile.standardize_dates,

        "date_format": profile.date_format,

        "column_rules": getattr(profile, "column_rules", None) or {},

    }





    from app.models.column_analysis import ColumnAnalysis

    from sqlalchemy import select

    stmt = select(ColumnAnalysis).where(ColumnAnalysis.dataset_id == dataset_id)

    analysis_result = await db.execute(stmt)

    analysis = analysis_result.scalar_one_or_none()



    if analysis and analysis.identifiers:

        identifier_action = request.missing_identifier_action or getattr(profile, "missing_identifier_action", "drop")

        column_rules = profile_dict.get("column_rules") or {}



        for col in analysis.identifiers:

            if col not in df.columns:

                continue

            if col not in column_rules:

                column_rules[col] = {}

            if identifier_action == "drop":

                column_rules[col]["impute"] = "drop"

            elif identifier_action == "fill_unknown":

                column_rules[col]["impute"] = "constant"

                column_rules[col]["fill_value"] = "Inconnu"



        profile_dict["column_rules"] = column_rules



    df_clean, cleaning_report = CleaningService.clean_dataframe(df, profile_dict)

    cleaning_report = _to_native_types(cleaning_report)



    backup_path: Optional[str] = None



    if save_as_new:

        new_filename = f"cleaned_{dataset.filename}"

        new_path = dataset.file_path.replace(dataset.filename, new_filename)



        if dataset.file_type == 'csv':

            df_clean.to_csv(new_path, index=False)

        elif dataset.file_type in ['xlsx', 'xls']:

            df_clean.to_excel(new_path, index=False)



        new_dataset = await DatasetRepository.create(

            db,

            filename=new_filename,

            original_filename=f"cleaned_{dataset.original_filename}",

            file_path=new_path,

            file_size=os.path.getsize(new_path),

            file_type=dataset.file_type,

            row_count=len(df_clean),

            column_count=len(df_clean.columns),

            columns=df_clean.columns.tolist(),

            user_id=current_user.user_id,

            status="processed"

        )



        target_dataset_id = new_dataset.id

    else:



        try:

            if os.path.exists(dataset.file_path):

                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

                backup_path = f"{dataset.file_path}.bak.{ts}"

                shutil.copy2(dataset.file_path, backup_path)

        except OSError as exc:

            logger.warning(

                "Could not back up dataset %s before overwrite: %s",

                dataset_id, exc,

            )

            backup_path = None



        if dataset.file_type == 'csv':

            df_clean.to_csv(dataset.file_path, index=False)

        elif dataset.file_type in ['xlsx', 'xls']:

            df_clean.to_excel(dataset.file_path, index=False)





        summary_stats = dict(dataset.summary_stats) if dataset.summary_stats else {}

        if "scan_result" in summary_stats:

            del summary_stats["scan_result"]



        await DatasetRepository.update_metadata(

            db,

            dataset_id,

            {

                "row_count": len(df_clean),

                "column_count": len(df_clean.columns),

                "summary_stats": summary_stats

            }

        )



        target_dataset_id = dataset_id



    log = await CleaningLogRepository.create(

        db,

        dataset_id=target_dataset_id,

        profile_id=profile.id,

        rows_before=int(cleaning_report["rows_before"]),

        rows_after=int(cleaning_report["rows_after"]),

        duplicates_removed=int(cleaning_report["changes"].get("duplicates", {}).get("duplicates_removed", 0)),

        missing_values_handled=int(cleaning_report["changes"].get("missing_values", {}).get("total_filled", 0)),

        outliers_detected=int(cleaning_report["changes"].get("outliers", {}).get("total_outliers", 0)),

        data_types_fixed=int(cleaning_report["changes"].get("data_types", {}).get("types_changed", 0)),

        operations_performed=cleaning_report["operations"],

        cleaning_report=cleaning_report,

        backup_path=backup_path,

    )



    logger.info("Cleaned dataset %s, created log %s", dataset_id, log.id)



    return log





@router.get("/datasets/{dataset_id}/quality", response_model=DataQualityResponse)

async def get_data_quality(

    dataset_id: int,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    """Get data quality report for a dataset"""





    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=f"Jeu de données {dataset_id} introuvable"

        )





    if not current_user.is_admin and dataset.user_id != current_user.user_id:

        raise HTTPException(

            status_code=status.HTTP_403_FORBIDDEN,

            detail="Vous n'avez pas accès à ce jeu de données"

        )





    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)



    quality_report = CleaningService.get_data_quality_report(df)

    validation_report = ValidationService.validate_dataframe(df)







    missing_pct = float(quality_report["missing_values"]["percentage"] or 0.0)

    duplicate_pct = float(quality_report["duplicates"]["percentage"] or 0.0)



    outlier_flag_cols = [c for c in df.columns if str(c).endswith("_outlier")]

    outlier_total = 0

    for c in outlier_flag_cols:

        try:

            outlier_total += int(df[c].fillna(False).astype(bool).sum())

        except Exception:

            continue

    outlier_pct = (outlier_total / len(df) * 100.0) if len(df) else 0.0



    missing_penalty = min(missing_pct, 100.0) * 0.40

    duplicate_penalty = min(duplicate_pct, 100.0) * 0.20

    outlier_penalty = min(outlier_pct, 100.0) * 0.20



    validation_penalty = 0.0

    if not validation_report["valid"]:

        validation_penalty += 15.0

    warnings_count = len(validation_report.get("warnings") or [])

    if warnings_count:

        validation_penalty += min(warnings_count * 2.0, 5.0)

    validation_penalty = min(validation_penalty, 20.0)



    quality_score = max(

        0.0,

        100.0 - missing_penalty - duplicate_penalty - outlier_penalty - validation_penalty,

    )



    return {

        "dataset_id": dataset_id,

        "quality_score": round(quality_score, 2),

        "total_rows": quality_report["total_rows"],

        "total_columns": quality_report["total_columns"],

        "missing_values": quality_report["missing_values"],

        "duplicates": quality_report["duplicates"],

        "data_types": quality_report["data_types"],

        "numeric_summary": quality_report["numeric_summary"],

        "categorical_summary": quality_report["categorical_summary"],

        "by_column": quality_report.get("by_column", {}),

        "validation_errors": validation_report["errors"],

        "validation_warnings": validation_report["warnings"],

    }





@router.get("/datasets/{dataset_id}/cleaning-history", response_model=List[CleaningReportResponse])

async def get_cleaning_history(

    dataset_id: int,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    """Get cleaning history for a dataset"""





    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=f"Jeu de données {dataset_id} introuvable"

        )





    if not current_user.is_admin and dataset.user_id != current_user.user_id:

        raise HTTPException(

            status_code=status.HTTP_403_FORBIDDEN,

            detail="Vous n'avez pas accès à ce jeu de données"

        )



    logs = await CleaningLogRepository.get_by_dataset(db, dataset_id)

    return logs





@router.get("/datasets/{dataset_id}/outliers")

async def get_outliers_metadata(

    dataset_id: int,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth),

):

    """Get outliers metadata from the latest cleaning run for this dataset."""

    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Jeu de données {dataset_id} introuvable")

    if not current_user.is_admin and dataset.user_id != current_user.user_id:

        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Vous n'avez pas accès à ce jeu de données")



    logs = await CleaningLogRepository.get_by_dataset(db, dataset_id)

    log = next(

        (

            item for item in logs

            if (item.cleaning_report or {}).get("changes", {}).get("outliers", {}).get("metadata")

        ),

        None,

    )

    if not log:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail="Aucune exécution de nettoyage avec métadonnées de valeurs aberrantes pour ce jeu de données",

        )



    outliers = (log.cleaning_report or {}).get("changes", {}).get("outliers", {})

    metadata = outliers.get("metadata")

    if not metadata:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail="Aucune métadonnée de valeurs aberrantes sauvegardée pour cette exécution de nettoyage"

        )



    return {

        "dataset_id": dataset_id,

        "cleaning_log_id": log.id,

        "method": outliers.get("method"),

        "threshold": outliers.get("threshold"),

        "outliers_by_column": outliers.get("outliers_by_column", {}),

        "total_outliers": outliers.get("total_outliers", 0),

        "metadata": metadata,

    }





@router.post("/datasets/{dataset_id}/outliers/apply")

async def apply_outliers_action(

    dataset_id: int,

    request: OutlierApplyRequest,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth),

):

    """Apply a user-approved outlier action (cap/remove) using latest metadata."""

    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Jeu de données {dataset_id} introuvable")

    if not current_user.is_admin and dataset.user_id != current_user.user_id:

        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Vous n'avez pas accès à ce jeu de données")



    if request.action == "flag":

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Utilisez 'cap' ou 'remove' comme action à appliquer")

    if not request.force:

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="force doit être true pour appliquer l'action sur les valeurs aberrantes")



    logs = await CleaningLogRepository.get_by_dataset(db, dataset_id)

    log = next(

        (

            item for item in logs

            if (item.cleaning_report or {}).get("changes", {}).get("outliers", {}).get("metadata")

        ),

        None,

    )

    if not log:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail="Aucune exécution de nettoyage avec métadonnées de valeurs aberrantes pour ce jeu de données",

        )



    outliers = (log.cleaning_report or {}).get("changes", {}).get("outliers", {})

    metadata = outliers.get("metadata")

    if not metadata:

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Aucune métadonnée de valeurs aberrantes sauvegardée pour cette exécution de nettoyage")



    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)

    rows_before = len(df)



    requested_rows = [int(r) for r in (request.selected_rows or [])]

    selected_rows = set()

    if requested_rows:

        requested_rows_set = set(requested_rows)





        selected_rows.update(r for r in requested_rows_set if r in df.index)





        selected_rows.update((r - 1) for r in requested_rows_set if (r - 1) in df.index and r >= 1)





        if "id" in df.columns:

            id_series = pd.to_numeric(df["id"], errors="coerce")

            matched_idx = df.index[id_series.isin(requested_rows_set)].tolist()

            selected_rows.update(int(i) for i in matched_idx)



        if not selected_rows:

            raise HTTPException(

                status_code=status.HTTP_400_BAD_REQUEST,

                detail=(

                    "selected_rows ne correspond à aucune ligne. "

                    "Vous pouvez passer des index de dataframe, des numéros de ligne (base 1), ou des valeurs de la colonne 'id'."

                ),

            )



    if request.target_columns:

        target_cols = {c for c in request.target_columns if c in metadata and c in df.columns}

        if not target_cols:

            raise HTTPException(

                status_code=status.HTTP_400_BAD_REQUEST,

                detail=(

                    "Aucune target_columns valide fournie. "

                    f"Colonnes disponibles avec métadonnées de valeurs aberrantes : {sorted([c for c in metadata.keys() if c in df.columns])}"

                ),

            )

    else:

        target_cols = {c for c in metadata.keys() if c in df.columns}



    removed_rows = set()

    capped_cells = 0

    columns_applied = []



    for col, col_meta in metadata.items():

        if col not in target_cols or col not in df.columns:

            continue

        lb = col_meta.get("lower_bound")

        ub = col_meta.get("upper_bound")

        rows = [int(r) for r in col_meta.get("rows", [])]







        if not selected_rows:

            if lb is None or ub is None:

                continue

            violating_mask = (df[col] < lb) | (df[col] > ub)

            violating_mask = violating_mask.fillna(False)

            valid_rows = [int(r) for r in df.index[violating_mask].tolist()]

        else:

            rows = [r for r in rows if r in selected_rows]

            if not rows:

                continue

            valid_rows = [r for r in rows if r in df.index]

            if not valid_rows:

                continue



        if request.action == "remove":

            removed_rows.update(valid_rows)

            columns_applied.append(col)

            continue



        if lb is None or ub is None:

            continue



        for r in valid_rows:

            if pd.isna(df.at[r, col]):

                continue

            original = df.at[r, col]

            if original < lb:

                df.at[r, col] = lb

                capped_cells += 1

            elif original > ub:

                df.at[r, col] = ub

                capped_cells += 1

        columns_applied.append(col)



    if request.action == "remove" and removed_rows:

        df = df.drop(index=list(removed_rows))



    if dataset.file_type == 'csv':

        df.to_csv(dataset.file_path, index=False)

    elif dataset.file_type in ['xlsx', 'xls']:

        df.to_excel(dataset.file_path, index=False)



    await DatasetRepository.update_metadata(

        db,

        dataset_id,

        {

            "row_count": len(df),

            "column_count": len(df.columns)

        }

    )



    apply_report = {

        "source_cleaning_log_id": log.id,

        "action": request.action,

        "force": request.force,

        "target_columns": sorted(list(target_cols)),

        "selected_rows": sorted(list(selected_rows)) if selected_rows else None,

        "selected_rows_input": requested_rows if requested_rows else None,

        "columns_applied": sorted(list(set(columns_applied))),

        "rows_before": rows_before,

        "rows_after": len(df),

        "rows_removed": len(removed_rows),

        "capped_cells": capped_cells,

    }



    apply_log = await CleaningLogRepository.create(

        db,

        dataset_id=dataset_id,

        profile_id=log.profile_id,

        rows_before=rows_before,

        rows_after=len(df),

        duplicates_removed=0,

        missing_values_handled=0,

        outliers_detected=len(removed_rows) if request.action == "remove" else capped_cells,

        data_types_fixed=0,

        operations_performed=["outlier_apply"],

        cleaning_report={"operations": ["outlier_apply"], "changes": {"outlier_apply": apply_report}},

        backup_path=None,

    )



    return {

        "dataset_id": dataset_id,

        "cleaning_log_id": apply_log.id,

        "outlier_apply": apply_report,

    }









@router.get("/datasets/{dataset_id}/scan")

async def scan_dataset(

    dataset_id: int,

    force: bool = True,

    method: str = "auto",

    threshold: float = 3.0,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth),

):

    """
    Return pre-computed scan result (cached at upload time).
    Falls back to a live scan if the background task hasn't finished yet.
    Pass ?force=true to bypass the cache and run a fresh scan.

    Response shape:
    {
      dataset_id, cached, total_issues,
      summary: { missing, missing_token, outlier, duplicate },
      issues: [ { row_number, column, issue_type, description, current_value, severity } ]
    }
    """

    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,

                            detail=f"Jeu de données {dataset_id} introuvable")

    if not current_user.is_admin and dataset.user_id != current_user.user_id:

        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,

                            detail="Vous n'avez pas accès à ce jeu de données")





    cached_scan = (dataset.summary_stats or {}).get("scan_result")

    if cached_scan and not force:

        return {"dataset_id": dataset_id, "cached": True, **cached_scan}





    logger.info(

        "Running live scan for dataset %s (force=%s, method=%s, threshold=%s)",

        dataset_id, force, method, threshold,

    )

    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)

    scan_result = ScanService.scan_dataframe(

        df,

        outlier_method=method,

        outlier_threshold=threshold,

    )





    try:

        summary_stats = dict(dataset.summary_stats) if dataset.summary_stats else {}

        summary_stats["scan_result"] = scan_result

        await DatasetRepository.update_metadata(db, dataset_id, {"summary_stats": summary_stats})

    except Exception as exc:

        logger.warning("Could not persist scan cache for dataset %s: %s", dataset_id, exc)



    return {"dataset_id": dataset_id, "cached": False, **scan_result}









@router.post("/datasets/{dataset_id}/validate")

async def validate_dataset(

    dataset_id: int,

    db: AsyncSession = Depends(get_db),

    current_user: CurrentUser = Depends(require_auth)

):

    """Validate dataset against default rules"""





    dataset = await DatasetRepository.get_by_id(db, dataset_id)

    if not dataset:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=f"Jeu de données {dataset_id} introuvable"

        )





    if not current_user.is_admin and dataset.user_id != current_user.user_id:

        raise HTTPException(

            status_code=status.HTTP_403_FORBIDDEN,

            detail="Vous n'avez pas accès à ce jeu de données"

        )





    df, _ = await ParserService.parse_file(dataset.file_path, dataset.file_type)





    validation_report = ValidationService.validate_dataframe(df)



    return {

        "dataset_id": dataset_id,

        "valid": validation_report["valid"],

        "errors": validation_report["errors"],

        "warnings": validation_report["warnings"],

        "checks_performed": validation_report["checks_performed"],

        "total_checks": validation_report["total_checks"]

    }

