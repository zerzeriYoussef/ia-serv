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
            return None  # Scalars render as KPI cards, not charts

        payload = tool_result.payload or {}

        if tool_result.result_type == "series":
            return self._series_to_chart(payload, intent_hint, col_types, title)

        if tool_result.result_type == "frame_preview":
            return self._frame_to_chart(payload, intent_hint, title)

        return None

    # ------------------------------------------------------------------

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

        # Decide bar vs line: line if x_col looks temporal or user asked for trend
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
        # Correlation matrix → heatmap
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
        return None
