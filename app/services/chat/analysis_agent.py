"""
Analysis Agent — safe Pandas tool executor for the chat system.

Wraps KPIExecutor patterns with:
- Strict op/agg allowlist (AllowedOp / ALLOWED_AGGS)
- Column name validation before any execution
- Result size capping (frame_preview ≤ 20 rows, series ≤ 50 entries)
- 10-second asyncio timeout via thread pool
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from app.api.v1.schemas.chat_schema import AllowedOp, ALLOWED_AGGS, ToolArgs, ToolResult

logger = logging.getLogger(__name__)

_EXECUTOR_TIMEOUT = 10.0  # seconds


class AnalysisAgentError(ValueError):
    """Raised for validation errors before execution begins."""


class AnalysisAgent:
    """
    Executes one ToolArgs call against a DataFrame.

    Design rules:
    - No eval() / exec() anywhere in this file.
    - Every op is an explicit branch — no dynamic dispatch.
    - Column names validated against df.columns before touching data.
    - Results bounded in size before returning.
    """

    MAX_SERIES_ROWS = 50
    MAX_FRAME_ROWS = 20

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, args: ToolArgs) -> ToolResult:
        """
        Validate args, execute the op in a thread pool with timeout, return ToolResult.
        Raises AnalysisAgentError for disallowed ops / missing columns.
        """
        self._validate(args)
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self._execute, args),
                timeout=_EXECUTOR_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise AnalysisAgentError(
                f"Tool execution timed out after {_EXECUTOR_TIMEOUT}s"
            )
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, args: ToolArgs) -> None:
        # Op allowlist
        if args.op not in AllowedOp.__members__.values():
            raise AnalysisAgentError(f"Op '{args.op}' is not in the allowed list")

        # Agg allowlist
        if args.agg_func and args.agg_func not in ALLOWED_AGGS:
            raise AnalysisAgentError(
                f"agg_func '{args.agg_func}' is not allowed. Allowed: {sorted(ALLOWED_AGGS)}"
            )

        # Column existence
        missing = []
        for col in [args.group_col, args.agg_col, args.filter_col]:
            if col and col not in self.df.columns:
                missing.append(col)
        if missing:
            raise AnalysisAgentError(
                f"Column(s) not found in dataset: {missing}. "
                f"Available: {list(self.df.columns[:30])}"
            )

    # ------------------------------------------------------------------
    # Dispatch (sync — called inside thread pool)
    # ------------------------------------------------------------------

    def _execute(self, args: ToolArgs) -> ToolResult:
        op = args.op
        if op == AllowedOp.groupby_agg:
            return self._groupby_agg(args)
        if op == AllowedOp.value_counts:
            return self._value_counts(args)
        if op == AllowedOp.describe:
            return self._describe(args)
        if op == AllowedOp.correlation:
            return self._correlation(args)
        if op == AllowedOp.filter_agg:
            return self._filter_agg(args)
        # Should never reach here due to _validate, but be explicit
        raise AnalysisAgentError(f"Unknown op '{op}'")

    # ------------------------------------------------------------------
    # Op implementations
    # ------------------------------------------------------------------

    def _groupby_agg(self, args: ToolArgs) -> ToolResult:
        group_col = args.group_col
        agg_col = args.agg_col
        agg_func = args.agg_func or "sum"

        if not group_col or not agg_col:
            raise AnalysisAgentError("groupby_agg requires group_col and agg_col")

        tmp = self.df.copy()
        tmp[agg_col] = pd.to_numeric(tmp[agg_col], errors="coerce")
        grouped = tmp.groupby(group_col)[agg_col].agg(agg_func)
        grouped = grouped.sort_values(ascending=False).head(args.top_n)

        payload = {
            "axis_x": [str(k) for k in grouped.index],
            "axis_y": [
                (float(v) if pd.notnull(v) else None) for v in grouped.values
            ],
            "group_col": group_col,
            "agg_col": agg_col,
            "agg_func": agg_func,
        }
        return ToolResult(result_type="series", payload=payload)

    def _value_counts(self, args: ToolArgs) -> ToolResult:
        col = args.group_col or args.agg_col
        if not col:
            raise AnalysisAgentError("value_counts requires group_col")

        vc = self.df[col].astype(str).value_counts().head(args.top_n)
        payload = {
            "axis_x": list(vc.index),
            "axis_y": [int(v) for v in vc.values],
            "column": col,
        }
        return ToolResult(result_type="series", payload=payload)

    def _describe(self, args: ToolArgs) -> ToolResult:
        col = args.agg_col or args.group_col
        if col:
            series = pd.to_numeric(self.df[col], errors="coerce")
            stats = series.describe().to_dict()
            payload = {k: (float(v) if pd.notnull(v) else None) for k, v in stats.items()}
            payload["column"] = col
            return ToolResult(result_type="scalar", payload=payload)

        # Full describe (numeric columns only)
        desc = self.df.select_dtypes(include="number").describe()
        payload = {}
        for c in desc.columns[:self.MAX_FRAME_ROWS]:
            payload[c] = {k: (float(v) if pd.notnull(v) else None)
                          for k, v in desc[c].to_dict().items()}
        return ToolResult(result_type="frame_preview", payload=payload)

    def _correlation(self, args: ToolArgs) -> ToolResult:
        num_df = self.df.select_dtypes(include="number")
        if args.agg_col and args.group_col:
            cols = [c for c in (args.group_col, args.agg_col) if c in num_df.columns]
            if len(cols) == 2:
                corr_val = float(num_df[cols[0]].corr(num_df[cols[1]]))
                return ToolResult(
                    result_type="scalar",
                    payload={"col_a": cols[0], "col_b": cols[1], "pearson_r": corr_val},
                )

        # Top-N correlation matrix (trimmed)
        corr = num_df.corr()
        trimmed = corr.iloc[:self.MAX_FRAME_ROWS, :self.MAX_FRAME_ROWS]
        payload = {
            "columns": list(trimmed.columns),
            "matrix": [
                [float(v) if pd.notnull(v) else None for v in row]
                for _, row in trimmed.iterrows()
            ],
        }
        return ToolResult(result_type="frame_preview", payload=payload)

    def _filter_agg(self, args: ToolArgs) -> ToolResult:
        filter_col = args.filter_col
        filter_val = args.filter_val
        agg_col = args.agg_col
        agg_func = args.agg_func or "sum"

        if not filter_col or filter_val is None or not agg_col:
            raise AnalysisAgentError("filter_agg requires filter_col, filter_val, and agg_col")

        # Safe string/numeric filter — no eval
        col_series = self.df[filter_col]
        if pd.api.types.is_numeric_dtype(col_series):
            try:
                mask = col_series == float(filter_val)
            except (ValueError, TypeError):
                mask = col_series.astype(str) == str(filter_val)
        else:
            mask = col_series.astype(str) == str(filter_val)

        filtered = self.df[mask]
        warnings = []
        if filtered.empty:
            warnings.append(f"No rows match {filter_col} == '{filter_val}'")

        tmp = filtered.copy()
        tmp[agg_col] = pd.to_numeric(tmp[agg_col], errors="coerce")
        agg_result = getattr(tmp[agg_col], agg_func)()
        value = float(agg_result) if pd.notnull(agg_result) else None

        return ToolResult(
            result_type="scalar",
            payload={
                "filter_col": filter_col,
                "filter_val": filter_val,
                "agg_col": agg_col,
                "agg_func": agg_func,
                "value": value,
                "matching_rows": int(mask.sum()),
            },
            warnings=warnings,
        )
