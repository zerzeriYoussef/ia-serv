"""
Deterministic KPI computation helpers for analysis responses.
"""



from __future__ import annotations



from typing import Any, Dict, List, Optional



import pandas as pd





def _to_float(value: Any) -> float:

    if pd.isna(value):

        return 0.0

    return float(value)





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



    metric = pd.to_numeric(df[primary_metric], errors="coerce")

    non_null_metric = metric.dropna()

    if non_null_metric.empty:

        return None



    total = _to_float(non_null_metric.sum())

    avg = _to_float(non_null_metric.mean())



    result: Dict[str, Any] = {

        "primary": {"name": primary_metric, "aggregation": "sum", "value": total},

        "cards": [

            {"name": f"{primary_metric}_total", "value": total},

            {"name": f"{primary_metric}_avg", "value": avg},

            {"name": "row_count", "value": int(len(df))},

        ],

        "trend": None,

        "by_dimension": [],

    }





    temporal_columns = temporal_columns or []

    for time_col in temporal_columns:

        if time_col not in df.columns:

            continue

        parsed_time = pd.to_datetime(df[time_col], errors="coerce")

        valid_ratio = parsed_time.notna().mean()

        if valid_ratio < 0.7:

            continue



        trend_df = pd.DataFrame(

            {"period": parsed_time.dt.to_period("M"), "metric": metric}

        ).dropna()

        if trend_df.empty:

            continue

        grouped = trend_df.groupby("period", as_index=False)["metric"].sum()

        grouped["period"] = grouped["period"].astype(str)

        grouped = grouped.sort_values("period")

        result["trend"] = {

            "time_column": time_col,

            "granularity": "month",

            "series": [

                {"period": row["period"], "value": _to_float(row["metric"])}

                for _, row in grouped.iterrows()

            ],

        }

        break



    dimension_columns = dimension_columns or []

    for dim_col in dimension_columns:

        if dim_col not in df.columns:

            continue

        dim_df = pd.DataFrame({"dim": df[dim_col], "metric": metric}).dropna()

        if dim_df.empty:

            continue

        grouped = (

            dim_df.groupby("dim", as_index=True)["metric"]

            .sum()

            .sort_values(ascending=False)

            .head(top_n)

        )

        result["by_dimension"].append(

            {

                "dimension": dim_col,

                "top_n": [

                    {"key": str(key), "value": _to_float(value)}

                    for key, value in grouped.items()

                ],

            }

        )

        break



    return result

