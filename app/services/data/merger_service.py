"""
Merger Service
--------------

Merges multiple datasets on auto-detected common columns using outer joins.
"""

import logging
from typing import List, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class MergerService:
    """Service for merging multiple datasets on common columns."""

    @staticmethod
    async def merge_multiple_datasets(
        datasets: List,  # List of Dataset model instances
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Merge multiple datasets using auto-detected join keys.

        Returns:
            merged_df   – Combined DataFrame
            merge_report – Details about every merge operation
        """

        if not datasets:
            raise ValueError("No datasets provided")

        # ── single dataset: no merge needed ─────────────────────────────────
        if len(datasets) == 1:
            from app.services.data.parser_service import ParserService

            df, _ = await ParserService.parse_file(
                datasets[0].file_path,
                datasets[0].file_type,
            )
            return df, {
                "datasets_merged": 1,
                "merge_operations": [],
                "merge_strategy": "single_dataset",
                "total_rows_before": len(df),
                "total_rows_after": len(df),
                "total_columns": len(df.columns),
                "datasets": [datasets[0].filename],
            }

        # ── load all dataframes ──────────────────────────────────────────────
        dfs: List[Tuple[str, pd.DataFrame]] = []
        total_rows_before = 0

        for dataset in datasets:
            from app.services.data.parser_service import ParserService

            df, _ = await ParserService.parse_file(
                dataset.file_path,
                dataset.file_type,
            )
            dfs.append((dataset.filename, df))
            total_rows_before += len(df)
            logger.info(
                f"Loaded {dataset.filename}: {len(df)} rows, "
                f"{len(df.columns)} columns"
            )

        # ── initialise merge report ──────────────────────────────────────────
        merge_report: Dict = {
            "datasets_merged": len(dfs),
            "merge_operations": [],
            "total_rows_before": total_rows_before,
            "total_rows_after": 0,
            "total_columns": 0,
            "datasets": [name for name, _ in dfs],
            "merge_strategy": "auto_detect_keys",
        }

        # ── start with first dataset ─────────────────────────────────────────
        merged_df = dfs[0][1].copy()
        logger.info(f"Starting merge with {dfs[0][0]}: {len(merged_df)} rows")

        # ── merge subsequent datasets ────────────────────────────────────────
        for i, (filename, df) in enumerate(dfs[1:], 1):
            common_cols = list(set(merged_df.columns) & set(df.columns))

            if common_cols:
                before_rows = len(merged_df)
                before_cols = len(merged_df.columns)

                try:
                    merged_df = pd.merge(
                        merged_df,
                        df,
                        on=common_cols,
                        how="outer",
                        suffixes=("", f"_{filename}"),
                    )

                    after_rows = len(merged_df)
                    after_cols = len(merged_df.columns)

                    merge_report["merge_operations"].append({
                        "step": i,
                        "dataset": filename,
                        "join_keys": common_cols,
                        "join_type": "outer",
                        "rows_before": before_rows,
                        "rows_after": after_rows,
                        "columns_before": before_cols,
                        "columns_after": after_cols,
                        "status": "success",
                    })

                    logger.info(
                        f"Merged {filename} on {common_cols}: "
                        f"{before_rows} → {after_rows} rows"
                    )

                except Exception as e:
                    logger.error(f"Failed to merge {filename}: {e}")
                    merge_report["merge_operations"].append({
                        "step": i,
                        "dataset": filename,
                        "status": "failed",
                        "reason": str(e),
                    })

            else:
                logger.warning(f"Skipping {filename}: no common columns found")
                merge_report["merge_operations"].append({
                    "step": i,
                    "dataset": filename,
                    "status": "skipped",
                    "reason": "no_common_columns",
                })

        merge_report["total_rows_after"] = len(merged_df)
        merge_report["total_columns"] = len(merged_df.columns)

        logger.info(
            f"Merge complete: {merge_report['total_rows_after']} rows, "
            f"{merge_report['total_columns']} columns"
        )

        return merged_df, merge_report
