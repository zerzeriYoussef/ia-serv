import re
import pandas as pd
import numpy as np
import unicodedata
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# Default tokens treated as missing values when found in object columns.
# Matched case-insensitively, against values that are already whitespace-stripped.
_DEFAULT_MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "-",
    "--",
    "?",
}


class CleaningService:
    """Service for cleaning and preprocessing data"""

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

    _QUANTITY_PATTERNS = (
        "qty", "quantity", "count", "units", "volume",
        "qte", "quantite", "nombre", "nb", "unite", "unites",
    )
    _PRICE_PATTERNS = (
        "price", "cost", "amount", "sales", "revenue", "total",
        "prix", "cout", "montant", "vente", "ventes", "revenu",
        "revenus", "tarif", "chiffre_affaires", "ca",
    )
    _PERCENT_PATTERNS = (
        "percent", "percentage", "pct", "rate", "ratio",
        "pourcentage", "taux",
    )

    @staticmethod
    def _normalize_column_name(col: str) -> str:
        """Normalize names so French accents do not break semantic matching."""
        text = unicodedata.normalize("NFKD", str(col))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    @staticmethod
    def _is_outlier_identifier_column(df: pd.DataFrame, col: str) -> bool:
        """Return True for numeric identifiers that should not be outlier-scanned."""
        col_lower = CleaningService._normalize_column_name(col)
        metric_patterns = (
            CleaningService._QUANTITY_PATTERNS
            + CleaningService._PRICE_PATTERNS
            + CleaningService._PERCENT_PATTERNS
        )
        if any(pattern in col_lower for pattern in metric_patterns):
            return False

        value_profile = CleaningService._numeric_value_profile(df[col])
        if (
            value_profile["count_like_score"] >= 0.75
            and value_profile["unique_ratio"] <= 0.70
        ):
            return False

        try:
            from app.services.analysis.relationship_detector import RelationshipDetector

            if RelationshipDetector._is_identifier(df, col):
                return True
        except Exception:
            pass

        return pd.api.types.is_numeric_dtype(df[col]) and any(
            pattern in col_lower for pattern in CleaningService._OUTLIER_EXCLUDE_PATTERNS
        )

    @staticmethod
    def _numeric_value_profile(series: pd.Series) -> Dict[str, float]:
        """Summarize numeric shape for value-based role inference."""
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        n = len(numeric)
        if n == 0:
            return {
                "n": 0.0,
                "integer_rate": 0.0,
                "non_negative_rate": 0.0,
                "positive_rate": 0.0,
                "unique_ratio": 0.0,
                "skew": 0.0,
                "count_like_score": 0.0,
                "percentage_0_100_rate": 0.0,
            }

        integer_rate = float(np.isclose(numeric, np.round(numeric)).mean())
        non_negative_rate = float((numeric >= 0).mean())
        positive_rate = float((numeric > 0).mean())
        unique_ratio = float(numeric.nunique(dropna=True) / max(n, 1))
        skew = float(numeric.skew()) if n >= 3 else 0.0
        percentage_0_100_rate = float(((numeric >= 0) & (numeric <= 100)).mean())

        repeated_score = max(0.0, min(1.0, 1.0 - unique_ratio))
        count_like_score = (
            0.45 * integer_rate
            + 0.35 * non_negative_rate
            + 0.20 * repeated_score
        )

        return {
            "n": float(n),
            "integer_rate": integer_rate,
            "non_negative_rate": non_negative_rate,
            "positive_rate": positive_rate,
            "unique_ratio": unique_ratio,
            "skew": skew,
            "count_like_score": float(count_like_score),
            "percentage_0_100_rate": percentage_0_100_rate,
        }

    @staticmethod
    def _infer_numeric_role_info(col: str, series: pd.Series) -> Dict[str, object]:
        """Infer a semantic numeric role using name hints and value shape."""
        col_lower = CleaningService._normalize_column_name(col)
        profile = CleaningService._numeric_value_profile(series)

        if any(pattern in col_lower for pattern in CleaningService._PERCENT_PATTERNS):
            return {
                "role": "percentage",
                "confidence": 0.95,
                "reason": "column name looks like a percentage/rate",
                "profile": profile,
            }
        if any(pattern in col_lower for pattern in CleaningService._QUANTITY_PATTERNS):
            return {
                "role": "quantity",
                "confidence": 0.95,
                "reason": "column name looks like a quantity/count",
                "profile": profile,
            }
        if any(pattern in col_lower for pattern in CleaningService._PRICE_PATTERNS):
            return {
                "role": "money",
                "confidence": 0.95,
                "reason": "column name looks like a price/money metric",
                "profile": profile,
            }

        if profile["count_like_score"] >= 0.75 and profile["unique_ratio"] <= 0.70:
            return {
                "role": "count_like",
                "confidence": round(float(profile["count_like_score"]), 2),
                "reason": "values are mostly integer, non-negative, and repeated",
                "profile": profile,
            }

        if (
            profile["percentage_0_100_rate"] >= 0.98
            and profile["unique_ratio"] > 0.10
            and profile["n"] >= 10
        ):
            return {
                "role": "percentage_like",
                "confidence": 0.70,
                "reason": "values mostly fall in the 0-100 range",
                "profile": profile,
            }

        non_na = pd.to_numeric(series, errors="coerce").dropna()
        if len(non_na) and (non_na >= 0).all() and profile["skew"] > 1.0:
            return {
                "role": "positive_skewed",
                "confidence": 0.65,
                "reason": "positive numeric column with right-skewed distribution",
                "profile": profile,
            }

        return {
            "role": "numeric",
            "confidence": 0.50,
            "reason": "generic numeric column",
            "profile": profile,
        }

    @staticmethod
    def _infer_numeric_role(col: str, series: pd.Series) -> str:
        """Compatibility helper returning only the role name."""
        return str(CleaningService._infer_numeric_role_info(col, series)["role"])

    @staticmethod
    def _auto_outlier_method(col: str, series: pd.Series, requested: str) -> str:
        """Choose the concrete detector for a numeric column."""
        if requested != "auto":
            return requested

        non_na = series.dropna()
        if len(non_na) < 8 or non_na.nunique(dropna=True) <= 2:
            return "skip"

        role = CleaningService._infer_numeric_role(col, series)
        if role in {"quantity", "count_like", "money", "positive_skewed"}:
            return "log_iqr"
        return "mad"

    @staticmethod
    def _default_duplicate_subset(df: pd.DataFrame) -> Optional[List[str]]:
        """
        Choose a safer default subset for duplicate detection.

        If an `id` column exists, we exclude it so records that are identical
        except for their identifier can still be flagged as duplicates.
        """
        cols = list(df.columns)
        if "id" in cols and len(cols) > 1:
            subset = [c for c in cols if c != "id"]
            return subset or None
        return None

    @staticmethod
    def clean_dataframe(
        df: pd.DataFrame,
        profile: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean dataframe based on cleaning profile.

        Pipeline order (designed so each step's stats reflect observed data):
            text -> missing-tokens -> types -> outliers -> missing-values
            -> duplicates -> dates
        """
        original_columns = list(df.columns)
        report: Dict = {
            "rows_before": len(df),
            "operations": [],
            "changes": {},
            "summary": {},
        }

        df_clean = df.copy()
        missing_before_total = int(df_clean.isnull().sum().sum())#counts nulls

        if profile.get("strip_whitespace", False):
            CleaningService._clean_text_inplace(
                df_clean,
                standardize=profile.get("standardize_text", False),
            )
            report["operations"].append("text_cleaning")

        column_rules = profile.get("column_rules") or {}

        df_clean, missing_tokens_report = CleaningService._normalize_missing_tokens(
            df_clean,
            column_rules=column_rules,
        )
        if missing_tokens_report.get("tokens_normalized", 0) > 0:
            report["operations"].append("missing_tokens")
            report["changes"]["missing_tokens"] = missing_tokens_report

        if profile.get("fix_data_types", False):
            df_clean, type_report = CleaningService._fix_data_types(
                df_clean,
                min_rate=float(profile.get("numeric_coerce_min_rate") or 0.8),
                explicit_date_format=profile.get("date_format"),
            )
            report["operations"].append("data_types")
            report["changes"]["data_types"] = type_report

        if profile.get("detect_outliers", False):
            outlier_max_rows = int(profile.get("outlier_max_rows") or 500_000)
            if len(df_clean) > outlier_max_rows:
                logger.warning(
                    "Skipping outlier detection: %s rows > limit %s",
                    len(df_clean), outlier_max_rows,
                )
                report["operations"].append("outliers")
                report["changes"]["outliers"] = {
                    "method": profile.get("outlier_method", "iqr"),
                    "threshold": float(profile.get("outlier_threshold") or 1.5),
                    "action": profile.get("outlier_action", "flag"),
                    "outliers_by_column": {},
                    "total_outliers": 0,
                    "skipped": True,
                    "reason": "dataset_too_large",
                    "max_rows": outlier_max_rows,
                }
            else:
                df_clean, outlier_report = CleaningService._handle_outliers(
                    df_clean,
                    method=profile.get("outlier_method", "iqr"),
                    threshold=float(profile.get("outlier_threshold") or 1.5),
                    action=profile.get("outlier_action", "flag"),
                    add_flag_columns=bool(profile.get("add_flag_columns", True)),
                    save_metadata=bool(profile.get("save_outliers_metadata", False)),
                    target_columns=profile.get("target_columns"),
                )
                report["operations"].append("outliers")
                report["changes"]["outliers"] = outlier_report

        if profile.get("handle_missing"):
            df_clean, missing_report = CleaningService._handle_missing(
                df_clean,
                strategy=profile.get("handle_missing", "drop"),
                fill_strategy=profile.get("missing_fill_strategy", "auto"),
                fill_value=profile.get("missing_fill_value"),
                column_rules=column_rules,
            )
            report["operations"].append("missing_values")
            report["changes"]["missing_values"] = missing_report

        if profile.get("remove_duplicates", False):
            df_clean, dup_report = CleaningService._remove_duplicates(
                df_clean,
                subset=profile.get("duplicate_subset"),
                keep=profile.get("duplicate_keep", "best"),
                normalize_text=bool(profile.get("dedup_normalize_text", True)),
            )
            report["operations"].append("duplicates")
            report["changes"]["duplicates"] = dup_report

        if profile.get("standardize_dates", False):
            df_clean, date_report = CleaningService._standardize_dates(
                df_clean,
                target_format=profile.get("date_format"),
            )
            report["operations"].append("date_standardization")
            report["changes"]["dates"] = date_report

        report["rows_after"] = len(df_clean)
        report["rows_removed"] = report["rows_before"] - report["rows_after"]

        cells_filled = int(
            report["changes"].get("missing_values", {}).get("total_filled", 0)
        )
        added_columns = [c for c in df_clean.columns if c not in original_columns]
        report["summary"] = {
            "rows_before": report["rows_before"],
            "rows_after": report["rows_after"],
            "rows_removed": report["rows_removed"],
            "cells_filled": cells_filled,
            "missing_before": missing_before_total,
            "missing_after": int(df_clean.isnull().sum().sum()),
            "duplicates_removed": int(
                report["changes"].get("duplicates", {}).get("duplicates_removed", 0)
            ),
            "outliers_total": int(
                report["changes"].get("outliers", {}).get("total_outliers", 0)
            ),
            "types_changed": int(
                report["changes"].get("data_types", {}).get("types_changed", 0)
            ),
            "added_columns": added_columns,
            "operations": list(report["operations"]),
        }

        return df_clean, report

    @staticmethod
    def _normalize_missing_tokens(
        df: pd.DataFrame,
        column_rules: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Replace common 'missing' textual placeholders with NaN on object columns.

        Recognized tokens (case-insensitive, after strip):
            "", "na", "n/a", "nan", "null", "none", "-", "--", "?"

        Per-column extra tokens may be supplied in `column_rules[col]["missing_tokens"]`.
        """
        column_rules = column_rules or {}
        normalized_total = 0
        per_column_counts: Dict[str, int] = {}

        for col in df.columns:
            if df[col].dtype != "object":
                continue
            extra_tokens = set()
            rule = column_rules.get(col, {})
            for tok in rule.get("missing_tokens", []) or []:
                if tok is None:
                    continue
                extra_tokens.add(str(tok).strip().lower())

            tokens = _DEFAULT_MISSING_TOKENS | extra_tokens

            series = df[col]
            stripped = series.astype("string").str.strip()
            lowered = stripped.str.lower()
            mask = lowered.isin(tokens) & series.notna()
            count = int(mask.sum())
            if count:
                df.loc[mask, col] = np.nan
                per_column_counts[col] = count
                normalized_total += count

        return df, {
            "tokens_normalized": int(normalized_total),
            "by_column": per_column_counts,
            "tokens": sorted(_DEFAULT_MISSING_TOKENS),
        }

    @staticmethod
    def _handle_missing(
        df: pd.DataFrame,
        strategy: str = "drop",
        fill_strategy: str = "auto",
        fill_value: Optional[str] = None,
        column_rules: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values.

        Imputation stats are computed *excluding rows flagged as outliers in the
        same column* (i.e. rows where `{col}_outlier == True`) so a single
        extreme value can't pull the mean/median.
        """
        column_rules = column_rules or {}
        missing_before = df.isnull().sum().to_dict()
        total_missing = int(df.isnull().sum().sum())
        rows_before = len(df)
        rows_dropped = 0
        total_filled = 0

        if strategy == "drop":
            df = df.dropna()
            rows_dropped = rows_before - len(df)

        elif strategy == "fill":
            for col in df.columns:
                if str(col).endswith("_outlier"):
                    continue
                if not df[col].isnull().any():
                    continue

                rule = column_rules.get(col, {})
                col_strategy = rule.get("impute") or fill_strategy
                col_fill_value = rule.get("fill_value")
                if col_fill_value is None:
                    col_fill_value = fill_value

                if col_strategy == "drop":
                    df = df.dropna(subset=[col])
                    continue

                outlier_col = f"{col}_outlier"
                if outlier_col in df.columns:
                    clean_mask = ~df[outlier_col].fillna(False).astype(bool)
                    stats_series = df.loc[clean_mask, col]
                else:
                    stats_series = df[col]

                fill = CleaningService._compute_fill_value(
                    df[col], stats_series, col_strategy, col_fill_value
                )
                if fill is None:
                    continue

                df[col] = df[col].fillna(fill)

            total_filled = total_missing - int(df.isnull().sum().sum())

        elif strategy == "interpolate":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].interpolate()
            df = df.ffill().bfill()
            total_filled = total_missing - int(df.isnull().sum().sum())

        missing_after = df.isnull().sum().to_dict()
        report = {
            "strategy": strategy,
            "fill_strategy": fill_strategy,
            "missing_before": {k: int(v) for k, v in missing_before.items() if v > 0},
            "missing_after": {k: int(v) for k, v in missing_after.items() if v > 0},
            "total_filled": int(total_filled),
            "rows_dropped": int(rows_dropped),
        }
        logger.info("Handled %s missing values using %s", total_filled, strategy)
        return df, report

    @staticmethod
    def _compute_fill_value(
        full_series: pd.Series,
        stats_series: pd.Series,
        strategy: str,
        fill_value,
    ):
        """Return a single fill value for a series given a strategy."""
        non_na = stats_series.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(full_series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(full_series)

        if strategy == "constant":
            if fill_value is None:
                return 0 if is_numeric else "Unknown"
            if is_numeric:
                try:
                    return float(fill_value)
                except (TypeError, ValueError):
                    return 0
            if is_datetime:
                try:
                    return pd.to_datetime(fill_value)
                except (TypeError, ValueError):
                    return None
            return fill_value

        if non_na.empty:
            non_na = full_series.dropna()
        if non_na.empty:
            return None

        if is_numeric:
            if strategy == "mean":
                return float(non_na.mean())
            if strategy == "median":
                return float(non_na.median())
            if strategy == "mode":
                mode = non_na.mode()
                return float(mode.iloc[0]) if len(mode) else float(non_na.median())
            return float(non_na.median())

        if is_datetime:
            mode = non_na.mode()
            return mode.iloc[0] if len(mode) else None

        mode = non_na.mode()
        return mode.iloc[0] if len(mode) else "Unknown"

    @staticmethod
    def _remove_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "best",
        normalize_text: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove duplicate rows.

        Args:
            keep: "first", "last" or "best". "best" keeps the row with the most
                non-null values within each duplicate group.
            normalize_text: when True, build a temporary normalized key
                (strip + lower + collapse whitespace) on object columns for
                matching only. Original values are preserved.
        """
        effective_subset = subset if subset else CleaningService._default_duplicate_subset(df)
        if effective_subset:
            effective_subset = [
                c for c in effective_subset
                if c in df.columns and not str(c).endswith("_outlier")
            ] or None

        if effective_subset is None:
            key_cols = [c for c in df.columns if not str(c).endswith("_outlier")]
        else:
            key_cols = effective_subset

        if not key_cols or len(df) == 0:
            return df, {
                "duplicates_found": 0,
                "duplicates_removed": 0,
                "subset": subset,
                "effective_subset": effective_subset,
                "keep": keep,
                "normalize_text": normalize_text,
            }

        if normalize_text:
            key_df = df[key_cols].copy()
            for kc in key_cols:
                if key_df[kc].dtype == "object":
                    key_df[kc] = (
                        key_df[kc].astype("string")
                        .str.strip()
                        .str.lower()
                        .str.replace(r"\s+", " ", regex=True)
                    )
        else:
            key_df = df[key_cols]

        duplicates_before = int(key_df.duplicated().sum())

        if keep == "best" and duplicates_before > 0:
            non_null_count = df.notna().sum(axis=1)
            order = non_null_count.sort_values(ascending=False, kind="stable").index
            key_sorted = key_df.loc[order]
            keep_mask_sorted = ~key_sorted.duplicated(keep="first")
            kept_idx = order[keep_mask_sorted]
            df_clean = df.loc[kept_idx].sort_index()
        else:
            pandas_keep = "first" if keep == "best" else keep
            df_clean = df.loc[~key_df.duplicated(keep=pandas_keep)]

        duplicates_removed = len(df) - len(df_clean)
        report = {
            "duplicates_found": int(duplicates_before),
            "duplicates_removed": int(duplicates_removed),
            "subset": subset,
            "effective_subset": effective_subset,
            "keep": keep,
            "normalize_text": normalize_text,
        }
        logger.info("Removed %s duplicate rows (keep=%s)", duplicates_removed, keep)
        return df_clean, report

    @staticmethod
    def _fix_data_types(
        df: pd.DataFrame,
        min_rate: float = 0.8,
        explicit_date_format: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Auto-detect and fix data types by coercing gracefully.

        - Numeric coercion only fires when at least `min_rate` of non-null values
          parse as numbers (default 0.8) to avoid silently nuking text columns.
        - Date coercion: if no explicit format is given and parsing with
          `dayfirst=True` and `dayfirst=False` disagrees on >5% of values, the
          column is flagged as ambiguous and *not* coerced.
        """
        type_changes: Dict[str, Dict[str, str]] = {}
        ambiguous_dates: List[str] = []

        for col in df.columns:
            if str(col).endswith("_outlier"):
                continue
            original_type = str(df[col].dtype)
            if df[col].dtype != "object":
                continue

            non_na = df[col].dropna()
            if len(non_na) == 0:
                continue

            num_coerced = pd.to_numeric(non_na, errors="coerce")
            num_success_rate = float(num_coerced.notna().mean())
            if num_success_rate >= min_rate:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                type_changes[col] = {
                    "from": original_type,
                    "to": str(df[col].dtype),
                }
                continue

            try:
                date_default = pd.to_datetime(
                    non_na, errors="coerce", format="mixed", dayfirst=False
                )
            except TypeError:
                date_default = pd.to_datetime(non_na, errors="coerce")
            date_success_rate = float(date_default.notna().mean())

            if date_success_rate < 0.3:
                continue

            if not explicit_date_format:
                try:
                    date_dayfirst = pd.to_datetime(
                        non_na, errors="coerce", format="mixed", dayfirst=True
                    )
                except TypeError:
                    date_dayfirst = pd.to_datetime(
                        non_na, errors="coerce", dayfirst=True
                    )
                both_ok = date_default.notna() & date_dayfirst.notna()
                if both_ok.any():
                    diff_rate = float(
                        (date_default[both_ok] != date_dayfirst[both_ok]).mean()
                    )
                else:
                    diff_rate = 0.0
                if diff_rate > 0.05:
                    ambiguous_dates.append(str(col))
                    continue

            try:
                df[col] = pd.to_datetime(
                    df[col], errors="coerce", format="mixed", dayfirst=False
                )
            except TypeError:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            type_changes[col] = {
                "from": original_type,
                "to": "datetime64[ns]",
            }

        report: Dict = {
            "types_changed": len(type_changes),
            "changes": type_changes,
            "min_rate": min_rate,
        }
        if ambiguous_dates:
            report["ambiguous_dates"] = ambiguous_dates
        logger.info("Fixed %s column data types", len(type_changes))
        return df, report

    @staticmethod
    def _handle_outliers(
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
        action: str = "flag",
        add_flag_columns: bool = True,
        save_metadata: bool = False,
        target_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers (computed on the column's *observed* values)."""
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns
            if (
                not str(c).endswith("_outlier")
                and not CleaningService._is_outlier_identifier_column(df, c)
            )
        ]
        if target_columns:
            target_set = set(target_columns)
            numeric_cols = [c for c in numeric_cols if c in target_set]

        if len(df) < 5:
            return df, {
                "method": method,
                "threshold": threshold,
                "action": action,
                "outliers_by_column": {},
                "total_outliers": 0,
                "skipped": True,
                "reason": "too_few_rows",
                "min_rows": 5,
            }

        outliers_detected: Dict[str, int] = {}
        total_outliers = 0
        outlier_metadata: Dict[str, Dict] = {}

        for col in numeric_cols:
            non_na = df[col].dropna()
            outlier_mask = pd.Series(False, index=df.index)
            lower_bound: Optional[float] = None
            upper_bound: Optional[float] = None
            effective_method = CleaningService._auto_outlier_method(col, df[col], method)
            role_info = CleaningService._infer_numeric_role_info(col, df[col])
            role = str(role_info["role"])
            domain_mask = pd.Series(False, index=df.index)
            # non_negative_role: values must be >= 0 (quantity, count, money, age-like)
            non_negative_role = role in {"quantity", "count_like", "money"}
            if non_negative_role:
                domain_mask = (df[col] < 0).fillna(False)
            elif role in {"percentage", "percentage_like"}:
                domain_mask = ((df[col] < 0) | (df[col] > 100)).fillna(False)
            severity = "warning"

            if effective_method in {"iqr", "log_iqr", "mad"} and len(non_na) < 4:
                effective_method = "skip"
            elif effective_method == "log_iqr" and (non_na < 0).any():
                effective_method = "skip"
            elif effective_method == "zscore" and len(non_na) < 2:
                effective_method = "skip"
            elif effective_method in {"isolation_forest", "lof"} and len(non_na) < 10:
                effective_method = "skip"

            if effective_method == "skip" and not bool(domain_mask.any()):
                continue

            if effective_method == "iqr":
                Q1 = float(non_na.quantile(0.25))
                Q3 = float(non_na.quantile(0.75))
                IQR = Q3 - Q1
                if IQR != 0:
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    if non_negative_role and lower_bound < 0:
                        lower_bound = 0.0
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_mask = outlier_mask.fillna(False)

            elif effective_method == "log_iqr":
                transformed = np.log1p(non_na)
                Q1 = float(transformed.quantile(0.25))
                Q3 = float(transformed.quantile(0.75))
                IQR = Q3 - Q1
                if IQR != 0:
                    log_lower = Q1 - threshold * IQR
                    log_upper = Q3 + threshold * IQR
                    lower_bound = float(np.expm1(log_lower))
                    upper_bound = float(np.expm1(log_upper))
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_mask = outlier_mask.fillna(False)

            elif effective_method == "mad":
                median = float(non_na.median())
                mad = float((non_na - median).abs().median())
                if mad != 0:
                    modified_z = 0.6745 * (non_na - median).abs() / mad
                    cutoff = threshold if threshold and threshold > 3.0 else 3.5
                    outlier_mask.loc[non_na.index] = modified_z > cutoff
                    lower_bound = float(median - (cutoff * mad / 0.6745))
                    upper_bound = float(median + (cutoff * mad / 0.6745))
                    # Clamp lower_bound for non-negative roles (e.g. age, price)
                    # so we never cap -10 to a value like -3.72
                    if non_negative_role and lower_bound < 0:
                        lower_bound = 0.0

            elif effective_method == "zscore":
                z_array = np.abs(stats.zscore(non_na))
                z_scores = pd.Series(z_array, index=non_na.index)
                outlier_mask.loc[non_na.index] = z_scores > threshold

            elif effective_method == "isolation_forest":
                model = IsolationForest(contamination="auto", random_state=42)
                labels = model.fit_predict(non_na.to_numpy().reshape(-1, 1))
                outlier_mask.loc[non_na.index] = labels == -1

            elif effective_method == "lof":
                n_neighbors = min(20, max(2, len(non_na) - 1))
                model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
                labels = model.fit_predict(non_na.to_numpy().reshape(-1, 1))
                outlier_mask.loc[non_na.index] = labels == -1

            outlier_mask = (outlier_mask | domain_mask).fillna(False)

            num_outliers = int(outlier_mask.sum())
            if num_outliers <= 0:
                continue

            if bool((outlier_mask & domain_mask).any()):
                severity = "critical"

            outliers_detected[col] = num_outliers
            total_outliers += num_outliers

            if save_metadata:
                rows = df.index[outlier_mask].tolist()
                values = [None if pd.isna(v) else float(v) for v in df.loc[outlier_mask, col].tolist()]
                meta = {
                    "rows": [int(r) for r in rows],
                    "values": values,
                    "method": method,
                    "effective_method": effective_method,
                    "threshold": threshold,
                    "severity": severity,
                    "column_role": role,
                    "role_confidence": role_info.get("confidence"),
                    "role_reason": role_info.get("reason"),
                    "value_profile": role_info.get("profile"),
                    "suggested_action": "review" if severity == "warning" else action,
                }
                if lower_bound is not None and upper_bound is not None:
                    meta["lower_bound"] = float(lower_bound)
                    meta["upper_bound"] = float(upper_bound)
                outlier_metadata[col] = meta

            if action == "remove":
                df = df.loc[~outlier_mask]
            elif action == "cap":
                # Handle statistical bounds
                if effective_method in {"iqr", "log_iqr", "mad"} and lower_bound is not None and upper_bound is not None:
                    df.loc[outlier_mask & (df[col] < lower_bound), col] = lower_bound
                    df.loc[outlier_mask & (df[col] > upper_bound), col] = upper_bound
                elif effective_method == "zscore":
                    mean_val = float(non_na.mean())
                    std_val = float(non_na.std())
                    lb = max(mean_val - threshold * std_val, 0.0) if non_negative_role else mean_val - threshold * std_val
                    ub = mean_val + threshold * std_val
                    df.loc[outlier_mask & (df[col] < lb), col] = lb
                    df.loc[outlier_mask & (df[col] > ub), col] = ub
                elif effective_method in {"isolation_forest", "lof", "skip"}:
                    # Cannot cap statistically without bounds; fallback to NaN so they can be imputed
                    # Only do this for purely statistical outliers (not domain outliers, which are handled below)
                    df.loc[outlier_mask & ~domain_mask, col] = np.nan

                # --- Domain floor / ceiling (applied AFTER statistical cap) ---
                # Empirical non-negative check: if ALL non-outlier values are >= 0,
                # the column is logically non-negative (e.g. AGE, SCORE, PRICE).
                # This catches columns whose unique_ratio is too high for count_like
                # classification but are still clearly non-negative in practice.
                non_outlier_vals = df.loc[~outlier_mask, col].dropna()
                col_is_non_negative = (
                    non_negative_role
                    or (len(non_outlier_vals) > 0 and (non_outlier_vals >= 0).mean() >= 0.90)
                )
                if col_is_non_negative:
                    df.loc[(df[col] < 0).fillna(False), col] = 0
                elif role in {"percentage", "percentage_like"}:
                    df.loc[(df[col] < 0).fillna(False), col] = 0
                    df.loc[(df[col] > 100).fillna(False), col] = 100
            elif action == "flag" and add_flag_columns:
                df[f"{col}_outlier"] = outlier_mask

        report = {
            "method": method,
            "threshold": threshold,
            "action": action,
            "add_flag_columns": add_flag_columns,
            "save_outliers_metadata": save_metadata,
            "target_columns": target_columns,
            "outliers_by_column": outliers_detected,
            "total_outliers": int(total_outliers),
        }
        if save_metadata:
            report["metadata"] = outlier_metadata
        logger.info("Detected %s outliers using %s method", total_outliers, method)
        return df, report

    @staticmethod
    def _clean_text_inplace(df: pd.DataFrame, standardize: bool = False) -> None:
        """Strip whitespace (and optionally standardize) text columns in place."""
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            df[col] = df[col].str.strip()
            if standardize:
                df[col] = df[col].str.lower()
                df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    @staticmethod
    def _clean_text(df: pd.DataFrame, standardize: bool = False) -> pd.DataFrame:
        """Compatibility wrapper kept for callers expecting a copy."""
        out = df.copy()
        CleaningService._clean_text_inplace(out, standardize=standardize)
        return out

    @staticmethod
    def _standardize_dates(
        df: pd.DataFrame,
        target_format: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Standardize date columns.

        If `target_format` is supplied, datetime columns are formatted as strings
        using that format (e.g. `%Y-%m-%d`). Otherwise the columns are left as
        `datetime64[ns]` so downstream tools keep a strong type.
        """
        date_cols = df.select_dtypes(include=["datetime64"]).columns
        changes: Dict[str, str] = {}
        for col in date_cols:
            if target_format:
                df[col] = df[col].dt.strftime(target_format)
                changes[col] = f"Formatted to {target_format}"
        return df, {
            "date_columns": list(date_cols),
            "format": target_format,
            "changes": changes,
        }

    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report."""
        if len(df) == 0:
            return {
                "total_rows": 0,
                "total_columns": len(df.columns),
                "memory_usage_mb": 0,
                "missing_values": {"total": 0, "by_column": {}, "percentage": 0},
                "duplicates": {"count": 0, "percentage": 0, "effective_subset": None},
                "data_types": df.dtypes.astype(str).to_dict(),
                "numeric_summary": {},
                "categorical_summary": {},
                "by_column": {},
            }

        dup_subset = CleaningService._default_duplicate_subset(df)
        dup_count = int(df.duplicated(subset=dup_subset).sum())

        total_cells = len(df) * len(df.columns)
        missing_total = int(df.isnull().sum().sum())

        report: Dict = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "missing_values": {
                "total": missing_total,
                "by_column": {
                    col: int(count) for col, count in df.isnull().sum().items() if count > 0
                },
                "percentage": round(missing_total / total_cells * 100, 2) if total_cells else 0,
            },
            "duplicates": {
                "count": dup_count,
                "percentage": round((dup_count / len(df)) * 100, 2),
                "effective_subset": dup_subset,
            },
            "data_types": df.dtypes.astype(str).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {},
            "by_column": {},
        }

        outlier_flag_cols = {
            c[: -len("_outlier")]: c for c in df.columns if str(c).endswith("_outlier")
        }

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if str(col).endswith("_outlier"):
                continue
            report["numeric_summary"][col] = {
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                "median": float(df[col].median()) if not df[col].isnull().all() else None,
                "std": float(df[col].std()) if not df[col].isnull().all() else None,
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None,
                "missing": int(df[col].isnull().sum()),
            }

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            report["categorical_summary"][col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                "missing": int(df[col].isnull().sum()),
            }

        for col in df.columns:
            if str(col).endswith("_outlier"):
                continue
            missing = int(df[col].isnull().sum())
            missing_pct = round(missing / len(df) * 100, 2)
            unique = int(df[col].nunique(dropna=True))
            unique_pct = round(unique / len(df) * 100, 2)
            outlier_count = 0
            flag_col = outlier_flag_cols.get(col)
            if flag_col:
                try:
                    outlier_count = int(df[flag_col].fillna(False).astype(bool).sum())
                except Exception:
                    outlier_count = 0
            completeness = round(100 - missing_pct, 2)
            report["by_column"][col] = {
                "dtype": str(df[col].dtype),
                "missing": missing,
                "missing_pct": missing_pct,
                "unique": unique,
                "unique_pct": unique_pct,
                "outlier_count": outlier_count,
                "completeness_score": completeness,
            }

        return report
