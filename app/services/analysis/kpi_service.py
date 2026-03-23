"""
Deterministic KPI computation helpers for analysis responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _safe_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def build_kpi_block(
    df: pd.DataFrame,
    primary_metric: Optional[str],
    temporal_columns: Optional[List[str]],
    dimension_columns: Optional[List[str]],
    top_n: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    Build KPI payload from dataframe + detected analysis columns.

    Returns None when no usable primary metric exists.
    """
    if not primary_metric or primary_metric not in df.columns:
        return None

    metric_series = pd.to_numeric(df[primary_metric], errors="coerce")
    metric_non_null = metric_series.dropna()
    total_value = _safe_float(metric_non_null.sum()) if not metric_non_null.empty else 0.0
    avg_value = _safe_float(metric_non_null.mean()) if not metric_non_null.empty else 0.0

    kpi: Dict[str, Any] = {
        "primary": {
            "name": primary_metric,
            "aggregation": "sum",
            "value": total_value,
        },
        "cards": [
            {"name": f"{primary_metric}_total", "value": total_value},
            {"name": f"{primary_metric}_avg", "value": avg_value},
            {"name": "row_count", "value": _safe_int(len(df))},
        ],
        "trend": None,
        "by_dimension": [],
    }

    # Trend: first temporal column, monthly sum
    temporal_columns = temporal_columns or []
    if temporal_columns:
        time_col = temporal_columns[0]
        if time_col in df.columns:
            dt = pd.to_datetime(df[time_col], errors="coerce")
            trend_df = pd.DataFrame({"_t": dt, "_m": metric_series}).dropna()
            if not trend_df.empty:
                monthly = (
                    trend_df.groupby(trend_df["_t"].dt.to_period("M"))["_m"]
                    .sum()
                    .reset_index()
                    .sort_values("_t")
                )
                kpi["trend"] = {
                    "time_column": time_col,
                    "granularity": "month",
                    "series": [
                        {"period": str(period), "value": _safe_float(value)}
                        for period, value in zip(monthly["_t"], monthly["_m"])
                    ],
                }

    # By dimension: first dimension, top-N sum
    dimension_columns = dimension_columns or []
    if dimension_columns:
        dim_col = dimension_columns[0]
        if dim_col in df.columns:
            dim_df = pd.DataFrame({"_d": df[dim_col], "_m": metric_series}).dropna()
            if not dim_df.empty:
                grouped = (
                    dim_df.groupby("_d", as_index=True)["_m"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(top_n)
                )
                kpi["by_dimension"].append(
                    {
                        "dimension": dim_col,
                        "top_n": [
                            {"key": str(key), "value": _safe_float(value)}
                            for key, value in grouped.items()
                        ],
                    }
                )

    return kpi

