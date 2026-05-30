"""
Visualization Agent — maps ToolResult + intent → ChartSpec.

Rules:
- Data sourced exclusively from ToolResult.payload — never invented.
- Validated against ChartSpec Pydantic model before returning.
- Pure deterministic logic — no LLM call here.
"""



from __future__ import annotations



import logging

from typing import Any, Dict, Optional



from app.api.v1.schemas.chat_schema import ChartSpec, ChartType, ToolResult



logger = logging.getLogger(__name__)





class VizAgent:

    """
    Converts a ToolResult into a validated ChartSpec.

    Mapping rules:
      series + groupby_agg  → bar (or line if temporal group_col)
      series + value_counts → bar
      frame_preview + corr  → heatmap
      scalar                → no chart (returns None)
    """



    TEMPORAL_HINTS = frozenset(

        {"date", "month", "year", "week", "quarter", "time", "period", "day"}

    )



    def build_spec(

        self,

        tool_result: ToolResult,

        *,

        intent_hint: str = "",

        col_types: Optional[Dict[str, str]] = None,

        title: Optional[str] = None,

    ) -> Optional[ChartSpec]:

        """
        Return a validated ChartSpec or None if the result type doesn't warrant a chart.
        """

        col_types = col_types or {}



        if tool_result.result_type == "scalar":

            return None



        payload = tool_result.payload or {}



        if tool_result.result_type == "series":

            return self._series_to_chart(payload, intent_hint, col_types, title)



        if tool_result.result_type == "frame_preview":

            return self._frame_to_chart(payload, intent_hint, title)



        return None







    def _series_to_chart(

        self,

        payload: Dict[str, Any],

        intent_hint: str,

        col_types: Dict[str, str],

        title: Optional[str],

    ) -> Optional[ChartSpec]:

        x_data = payload.get("axis_x", [])

        y_data = payload.get("axis_y", [])

        x_col = payload.get("group_col") or payload.get("column", "x")

        y_col = payload.get("agg_col", "y")

        agg_func = payload.get("agg_func", "")



        if not x_data or not y_data:

            return None





        chart_type = ChartType.bar

        x_lower = x_col.lower()

        if any(h in x_lower for h in self.TEMPORAL_HINTS) or "trend" in intent_hint.lower():

            chart_type = ChartType.line



        caption = (

            f"{agg_func.capitalize()} of {y_col} grouped by {x_col}."

            if agg_func else f"Distribution of {x_col}."

        )



        return ChartSpec(

            chart_type=chart_type,

            x_col=x_col,

            y_col=y_col,

            agg_func=agg_func or None,

            title=title or f"{y_col} by {x_col}",

            caption=caption,

            data={"axis_x": x_data, "axis_y": y_data},

        )



    def _frame_to_chart(

        self,

        payload: Dict[str, Any],

        intent_hint: str,

        title: Optional[str],

    ) -> Optional[ChartSpec]:



        if "matrix" in payload and "columns" in payload:

            cols = payload["columns"]

            return ChartSpec(

                chart_type=ChartType.heatmap,

                x_col="columns",

                y_col="columns",

                agg_func="pearson_r",

                title=title or "Correlation Matrix",

                caption="Pearson correlation between numeric columns.",

                data={

                    "labels_x": cols,

                    "labels_y": cols,

                    "matrix": payload["matrix"],

                },

            )



        cols = list(payload.keys())

        if len(cols) < 2:

            return None



        x_col = cols[0]

        axis_x = [str(v) for v in payload[x_col]]



        numeric_cols: list[str] = []

        for c in cols[1:]:

            vals = payload.get(c) or []

            if not vals:

                continue

            sample = vals[: min(5, len(vals))]

            if all(v is None or isinstance(v, (int, float)) for v in sample):

                numeric_cols.append(c)



        if not axis_x or not numeric_cols:

            return None



        chart_type = ChartType.bar

        x_lower = str(x_col).lower()

        if any(h in x_lower for h in self.TEMPORAL_HINTS) or "trend" in intent_hint.lower():

            chart_type = ChartType.line



        if len(numeric_cols) == 1:

            y_col = numeric_cols[0]

            axis_y = [

                float(v) if v is not None and isinstance(v, (int, float)) else 0.0

                for v in payload[y_col]

            ]

            return ChartSpec(

                chart_type=chart_type,

                x_col=str(x_col),

                y_col=str(y_col),

                title=title or f"{y_col} by {x_col}",

                caption="Tabular data visualization.",

                data={"axis_x": axis_x, "axis_y": axis_y},

            )





        series = []

        for c in numeric_cols:

            series.append({

                "name": str(c),

                "values": [

                    float(v) if v is not None and isinstance(v, (int, float)) else 0.0

                    for v in payload[c]

                ],

            })

        y_label = " vs ".join(numeric_cols[:3])

        if len(numeric_cols) > 3:

            y_label += "…"

        return ChartSpec(

            chart_type=chart_type,

            x_col=str(x_col),

            y_col=y_label,

            title=title or f"Comparison by {x_col}",

            caption="Multi-series comparison.",

            data={"axis_x": axis_x, "series": series},

        )

