"""
"""



import pandas as pd

import numpy as np

from typing import Dict, List, Optional

from collections import Counter



from app.services.data.cleaning_service import CleaningService, _DEFAULT_MISSING_TOKENS

import logging

logger = logging.getLogger(__name__)





class ScanService:











    _OUTLIER_EXCLUDE_PATTERNS = {

        "id", "_id", "code", "_code", "key", "_key",

        "ref", "num", "number", "no", "index", "idx",

        "sku", "barcode", "upc", "isbn", "ean",

        "invoice", "order", "transaction", "receipt",

        "stock", "product", "item", "ticket",

        "uuid", "guid", "hash", "token", "session",

        "email", "phone", "mobile", "username", "login",

        "zip", "postal",

    }



    @staticmethod

    def _get_identifier_columns(df: pd.DataFrame) -> List[str]:

        """
        Return columns that should be excluded from outlier detection.

        Uses TWO checks (either one is enough to exclude):
        1. RelationshipDetector._is_identifier() — strict, multi-heuristic
        2. Name-based pattern matching — broad, catches StockCode, InvoiceNo, etc.
           even when uniqueness is low (same product ordered many times).

        This is intentionally more lenient than the relationship detector because
        a column named 'StockCode' should NEVER be treated as a numeric metric,
        regardless of its cardinality.
        """

        excluded = set()



        for col in df.columns:







            if CleaningService._is_outlier_identifier_column(df, col):

                excluded.add(col)

                continue



        return list(excluded)



    @staticmethod

    def scan_dataframe(

        df: pd.DataFrame,

        outlier_method: str = "auto",

        outlier_threshold: float = 3.0,

    ) -> Dict:

        """
        Read-only scan of a dataframe.
        Returns structured issues with row_number, column, issue_type, description.
        Never modifies df — always works on df.copy() for internal operations.

        Args:
            outlier_method:    "auto", "iqr", "mad", "log_iqr", "zscore",
                               "isolation_forest", or "lof"
            outlier_threshold: multiplier/cutoff. Default 3.0 favors extreme
                               outliers only in the read-only scan.
        """

        issues: List[Dict] = []





        for col in df.columns:

            missing_idx = df.index[df[col].isnull()].tolist()

            for row_idx in missing_idx:

                issues.append({

                    "row_index"    : int(row_idx),

                    "row_number"   : int(row_idx) + 1,

                    "column"       : str(col),

                    "issue_type"   : "missing",

                    "description"  : f"Valeur manquante dans la colonne « {col} »",

                    "current_value": None,

                    "severity"     : "error",

                })





        for col in df.select_dtypes(include="object").columns:

            try:

                stripped = df[col].astype("string").str.strip().str.lower()

                mask = stripped.isin(_DEFAULT_MISSING_TOKENS) & df[col].notna()

                for row_idx in df.index[mask].tolist():

                    raw_val = df.at[row_idx, col]

                    issues.append({

                        "row_index"    : int(row_idx),

                        "row_number"   : int(row_idx) + 1,

                        "column"       : str(col),

                        "issue_type"   : "missing_token",

                        "description"  : f"Token nul « {raw_val} » dans la colonne « {col} » (traité comme manquant)",

                        "current_value": str(raw_val),

                        "severity"     : "error",

                    })

            except Exception as exc:

                logger.debug("missing_token scan failed for col %s: %s", col, exc)









        try:

            id_cols = ScanService._get_identifier_columns(df)

            if id_cols:

                logger.info("Outlier scan: skipping identifier columns %s", id_cols)



            df_for_outliers = df.drop(columns=id_cols, errors="ignore").copy()



            _, outlier_report = CleaningService._handle_outliers(

                df_for_outliers,

                method=outlier_method,

                threshold=outlier_threshold,

                action="flag",

                add_flag_columns=False,

                save_metadata=True,

            )

            for col, meta in outlier_report.get("metadata", {}).items():

                lb = meta.get("lower_bound")

                ub = meta.get("upper_bound")

                effective_method = meta.get("effective_method", meta.get("method"))

                severity = meta.get("severity", "warning")

                reason = "regle de domaine invalide"

                bound_str = f"[{lb:.2f} – {ub:.2f}]" if lb is not None and ub is not None else ""

                if bound_str:

                    reason = f"hors borne normale {bound_str}"

                for row_idx, val in zip(meta.get("rows", []), meta.get("values", [])):

                    issues.append({

                        "row_index"    : int(row_idx),

                        "row_number"   : int(row_idx) + 1,

                        "column"       : str(col),

                        "issue_type"   : "outlier",

                        "description"  : (

                            f"Valeur aberrante {val} dans « {col} » "

                            f"— {reason}"

                        ),

                        "current_value": val,

                        "severity"     : severity,

                        "metadata"     : {

                            "method": meta.get("method"),

                            "effective_method": effective_method,

                            "column_role": meta.get("column_role"),

                            "role_confidence": meta.get("role_confidence"),

                            "role_reason": meta.get("role_reason"),

                            "value_profile": meta.get("value_profile"),

                            "suggested_action": meta.get("suggested_action", "review"),

                            "lower_bound": lb,

                            "upper_bound": ub,

                        },

                    })

        except Exception as exc:

            logger.warning("Outlier scan failed: %s", exc)





        try:

            dup_subset = CleaningService._default_duplicate_subset(df)

            key_cols   = dup_subset if dup_subset else [c for c in df.columns]

            dup_mask   = df.duplicated(subset=key_cols, keep=False)



            if dup_mask.any():

                first_seen: Dict[str, int] = {}

                for row_idx in df.index[dup_mask].tolist():

                    try:

                        key = str(df.loc[row_idx, key_cols].values.tolist())

                    except Exception:

                        key = str(row_idx)



                    if key not in first_seen:

                        first_seen[key] = int(row_idx)

                    else:

                        orig_row = first_seen[key] + 1

                        issues.append({

                            "row_index"    : int(row_idx),

                            "row_number"   : int(row_idx) + 1,

                            "column"       : None,

                            "issue_type"   : "duplicate",

                            "description"  : f"Ligne dupliquée — identique à la ligne {orig_row}",

                            "current_value": None,

                            "severity"     : "warning",

                        })

        except Exception as exc:

            logger.warning("Duplicate scan failed: %s", exc)





        summary = dict(Counter(i["issue_type"] for i in issues))



        result = {

            "total_issues": len(issues),

            "summary"     : summary,

            "issues"      : sorted(

                issues,

                key=lambda x: (x["row_index"], x["column"] or "")

            ),

        }



        logger.info(

            "Scan complete — %s issues found: %s",

            result["total_issues"], summary

        )

        return result

