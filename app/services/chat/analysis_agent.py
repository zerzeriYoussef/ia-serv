"""
Analysis Agent — executes LLM-generated Pandas expressions against a DataFrame.

The LLM produces a `code_expr` string (a single Python expression). This agent
evaluates it in a restricted namespace that only exposes `df` and `pd`, so the
LLM can express any query naturally (multi-filter, groupby, pivot, etc.) while
preventing access to the filesystem, network, or OS.

Sandbox rules (enforced via restricted eval namespace):
- Only `df` (the dataset DataFrame) and `pd` (pandas) are available
- `__builtins__` is completely removed
- No import, open, os, exec, eval, or network calls possible
- Results are normalised to a ToolResult with a capped payload
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pandas as pd

from app.api.v1.schemas.chat_schema import ToolArgs, ToolResult

logger = logging.getLogger(__name__)

_EXECUTOR_TIMEOUT = 15.0  # seconds — slightly more headroom for complex queries

# Builtins that are safe to expose in the sandbox
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "range": range,
    "list": list,
    "dict": dict,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

MAX_SERIES_ROWS = 50
MAX_FRAME_ROWS = 20


class AnalysisAgentError(ValueError):
    """Raised when expression evaluation fails."""


class AnalysisAgent:
    """
    Evaluates a free-form Pandas expression produced by the LLM.

    The expression is run in a restricted namespace:
      - `df`  → the full dataset DataFrame
      - `pd`  → pandas
      - `np`  → numpy
      - safe builtins only (no open, import, exec, eval, os, etc.)

    The raw result (scalar, Series, or DataFrame) is normalised into a
    ToolResult with a bounded payload so the Narrator always gets clean data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, args: ToolArgs) -> ToolResult:
        """
        Evaluate `args.code_expr` asynchronously in a thread pool.
        Returns a normalised ToolResult.
        Raises AnalysisAgentError on syntax / runtime errors.
        """
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._execute, args.code_expr
                ),
                timeout=_EXECUTOR_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise AnalysisAgentError(
                f"Expression timed out after {_EXECUTOR_TIMEOUT}s"
            )
        return result

    # ------------------------------------------------------------------
    # Sandboxed execution
    # ------------------------------------------------------------------

    def _execute(self, code_expr: str) -> ToolResult:
        """Evaluate `code_expr` in a restricted namespace and normalise result."""
        namespace = {
            "__builtins__": _SAFE_BUILTINS,
            "df": self.df,
            "pd": pd,
            "np": np,
        }

        try:
            raw = eval(compile(code_expr, "<llm_expr>", "eval"), namespace)
        except SyntaxError as exc:
            raise AnalysisAgentError(f"Syntax error in generated expression: {exc}")
        except Exception as exc:
            raise AnalysisAgentError(f"Runtime error evaluating expression: {exc}")

        return self._normalise(raw)

    # ------------------------------------------------------------------
    # Result normalisation
    # ------------------------------------------------------------------

    def _normalise(self, raw: Any) -> ToolResult:
        """Convert any pandas/python result into a bounded ToolResult."""

        # ── DataFrame ─────────────────────────────────────────────────
        if isinstance(raw, pd.DataFrame):
            # Square correlation matrix → heatmap-friendly payload
            if (
                raw.shape[0] == raw.shape[1]
                and raw.shape[0] >= 2
                and list(raw.index.astype(str)) == list(raw.columns.astype(str))
            ):
                cols = [str(c) for c in raw.columns]
                matrix = [
                    [
                        float(raw.iloc[i, j]) if pd.notnull(raw.iloc[i, j]) else None
                        for j in range(len(cols))
                    ]
                    for i in range(len(cols))
                ]
                return ToolResult(
                    result_type="frame_preview",
                    payload={"matrix": matrix, "columns": cols},
                )

            # Include index if it is meaningful (not a default RangeIndex or flat index without name)
            if not isinstance(raw.index, pd.RangeIndex) or raw.index.name is not None:
                raw = raw.reset_index()

            trimmed = raw.head(MAX_FRAME_ROWS)
            payload: Any = {
                str(col): [
                    (float(v) if isinstance(v, (int, float, np.integer, np.floating)) and pd.notnull(v) else
                     None if (isinstance(v, float) and pd.isnull(v)) else
                     str(v))
                    for v in trimmed[col]
                ]
                for col in trimmed.columns
            }
            return ToolResult(result_type="frame_preview", payload=payload)

        # ── Series ────────────────────────────────────────────────────
        if isinstance(raw, pd.Series):
            s = raw.head(MAX_SERIES_ROWS)
            
            # Extract intuitive column names for plotting
            s_index_names = [str(n) for n in getattr(s.index, "names", [s.index.name]) if n is not None]
            group_col = ", ".join(s_index_names) if s_index_names else "category"
            agg_col = str(s.name) if s.name is not None else "value"
            
            payload = {
                "axis_x": [str(k) for k in s.index],
                "axis_y": [
                    (float(v) if pd.notnull(v) else None) for v in s.values
                ],
                "name": agg_col,
                "group_col": group_col,
                "agg_col": agg_col,
            }
            return ToolResult(result_type="series", payload=payload)

        # ── Scalar (int, float, str, bool, numpy scalar) ──────────────
        if isinstance(raw, (int, float, str, bool, np.integer, np.floating)):
            value = float(raw) if isinstance(raw, (int, float, np.integer, np.floating)) else raw
            return ToolResult(result_type="scalar", payload={"value": value})

        # ── Fallback: stringify whatever came back ─────────────────────
        warnings = ["Result type not natively supported; converted to string."]
        return ToolResult(
            result_type="scalar",
            payload={"value": str(raw)[:500]},
            warnings=warnings,
        )
