"""
KPI Executor Service
Executes pandas logic from dashboard KPI/chart configurations.
SAFE execution — no eval() or exec().
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class KPIExecutor:
    """
    Execute KPI calculations and chart groupings from dashboard configurations.
    All execution is pattern-matched — never uses eval() or exec().
    """

    # Supported aggregation functions mapped to safe lambdas
    SAFE_AGGREGATIONS: Dict[str, Callable[[pd.DataFrame, str], Any]] = {
        "sum": lambda df, col: df[col].sum(),
        "mean": lambda df, col: df[col].mean(),
        "avg": lambda df, col: df[col].mean(),
        "median": lambda df, col: df[col].median(),
        "min": lambda df, col: df[col].min(),
        "max": lambda df, col: df[col].max(),
        "count": lambda df, col: df[col].count(),
        "nunique": lambda df, col: df[col].nunique(),
        "std": lambda df, col: df[col].std(),
        "var": lambda df, col: df[col].var(),
    }

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    # ------------------------------------------------------------------
    # Single KPI
    # ------------------------------------------------------------------

    def execute_kpi(self, kpi_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single KPI configuration.

        Args:
            kpi_config: dict with keys kpi_name, target_column, pandas_function, …

        Returns:
            Result dict with value, formatted_value, execution_success, optional error.
        """
        kpi_name = kpi_config.get("kpi_name", "Unknown KPI")
        target_column = kpi_config.get("target_column")
        pandas_function = kpi_config.get("pandas_function", "")
        logic_hint = kpi_config.get("logic_hint", "")
        combined = f"{pandas_function} {logic_hint}"

        try:
            # --- Row count (len(df), df.shape[0]) ---
            if re.search(r"\.shape\s*\[\s*0\s*\]|len\s*\(\s*df\s*\)", combined, re.I):
                n = len(self.df)
                return {
                    "kpi_name": kpi_name,
                    "target_column": target_column,
                    "value": float(n),
                    "formatted_value": str(n),
                    "function_used": "rowcount",
                    "execution_success": True,
                }

            # --- groupby + agg + idxmax (e.g. top category by total) ---
            idxmax_ok, idxmax_payload = self._try_kpi_groupby_idxmax(
                kpi_name, target_column, combined, require_idxmax=True
            )
            if idxmax_ok:
                return idxmax_payload
            idxmax_ok2, idxmax_payload2 = self._try_kpi_groupby_idxmax(
                kpi_name, target_column, combined, require_idxmax=False
            )
            if idxmax_ok2:
                return idxmax_payload2

            if not target_column or target_column not in self.df.columns:
                return self._kpi_error(
                    kpi_name,
                    target_column,
                    f"Column '{target_column}' not found in dataset",
                )

            series = self.df[target_column]
            function_name = self._parse_function(pandas_function, logic_hint)

            # --- Categorical / text: mode-style (no numeric coercion) ---
            if function_name in ("mode", "top", "most_common") or (
                function_name in self.SAFE_AGGREGATIONS
                and not pd.api.types.is_numeric_dtype(series)
                and function_name in ("count", "nunique")
            ):
                if function_name in ("mode", "top", "most_common"):
                    vc = series.astype(str).value_counts()
                    if vc.empty:
                        return self._kpi_error(kpi_name, target_column, "No values to count")
                    top_val = vc.index[0]
                    top_cnt = int(vc.iloc[0])
                    return {
                        "kpi_name": kpi_name,
                        "target_column": target_column,
                        "value": float(top_cnt),
                        "formatted_value": f"{top_val} (n={top_cnt})",
                        "function_used": "mode",
                        "execution_success": True,
                    }

            # Coerce to numeric — skip rows that can't be converted
            col_numeric = pd.to_numeric(series, errors="coerce")
            if col_numeric.isna().all():
                # Categorical column but LLM used .mean() / .sum() — interpret as "most common"
                if pd.api.types.is_object_dtype(series.dtype) or isinstance(
                    series.dtype, pd.CategoricalDtype
                ):
                    vc = series.astype(str).value_counts()
                    if vc.empty:
                        return self._kpi_error(kpi_name, target_column, "No values to count")
                    top_val = vc.index[0]
                    top_cnt = int(vc.iloc[0])
                    return {
                        "kpi_name": kpi_name,
                        "target_column": target_column,
                        "value": float(top_cnt),
                        "formatted_value": f"{top_val} (n={top_cnt})",
                        "function_used": "mode",
                        "execution_success": True,
                    }
                # Last try: count non-null raw values
                if function_name in ("count", "nunique"):
                    fn = self.SAFE_AGGREGATIONS[function_name]
                    val = fn(self.df, target_column)
                    return {
                        "kpi_name": kpi_name,
                        "target_column": target_column,
                        "value": float(val),
                        "formatted_value": self._format_value(float(val)),
                        "function_used": function_name,
                        "execution_success": True,
                    }
                return self._kpi_error(
                    kpi_name,
                    target_column,
                    f"Column '{target_column}' has no numeric values",
                )

            tmp_df = self.df.copy()
            tmp_df[target_column] = col_numeric

            if function_name in self.SAFE_AGGREGATIONS:
                raw = self.SAFE_AGGREGATIONS[function_name](tmp_df, target_column)
                value = float(raw) if pd.notnull(raw) else float("nan")
                return {
                    "kpi_name": kpi_name,
                    "target_column": target_column,
                    "value": value,
                    "formatted_value": self._format_value(value),
                    "function_used": function_name,
                    "execution_success": True,
                }

            return self._kpi_error(
                kpi_name,
                target_column,
                f"Unsupported function: '{function_name}'. "
                f"Supported: {list(self.SAFE_AGGREGATIONS)} plus rowcount/mode/groupby_idxmax",
            )

        except Exception as exc:
            logger.error("KPI execution failed for '%s': %s", kpi_name, exc)
            return self._kpi_error(kpi_name, target_column, str(exc))

    def _try_kpi_groupby_idxmax(
        self,
        kpi_name: str,
        target_column: Optional[str],
        combined: str,
        *,
        require_idxmax: bool,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Parse df.groupby('G')['V'].sum().idxmax() style hints and compute
        the label with the maximum aggregated value.
        """
        if require_idxmax:
            pattern = re.compile(
                r"groupby\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*"
                r"\[\s*['\"]([^'\"]+)['\"]\s*\]\s*"
                r"\.(\w+)\s*\(\s*\)\s*"
                r"\.idxmax\s*\(\s*\)",
                re.I,
            )
        else:
            # Infer "top dimension by metric" when KPI title suggests it
            if "top" not in kpi_name.lower():
                return False, {}
            pattern = re.compile(
                r"groupby\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*"
                r"\[\s*['\"]([^'\"]+)['\"]\s*\]\s*"
                r"\.(\w+)\s*\(\s*\)",
                re.I,
            )
        m = pattern.search(combined)
        if not m:
            return False, {}

        group_col, value_col, agg_name = m.group(1), m.group(2), m.group(3).lower()
        for c in (group_col, value_col):
            if c not in self.df.columns:
                return True, self._kpi_error(kpi_name, target_column, f"Column '{c}' not found")

        _allowed_gb = {"sum", "mean", "avg", "median", "min", "max", "count", "std", "var", "nunique"}
        if agg_name not in _allowed_gb:
            return True, self._kpi_error(kpi_name, target_column, f"Unsupported agg '{agg_name}' in idxmax KPI")

        gb_agg = "mean" if agg_name == "avg" else agg_name
        tmp = self.df.copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        grouped = tmp.groupby(group_col)[value_col].agg(gb_agg)
        if grouped.empty or grouped.isna().all():
            return True, self._kpi_error(kpi_name, target_column, "Empty groupby result for idxmax KPI")

        winner = grouped.idxmax()
        win_val = float(grouped.max())
        return True, {
            "kpi_name": kpi_name,
            "target_column": group_col,
            "value": win_val,
            "formatted_value": f"{winner} ({self._format_value(win_val)})",
            "function_used": f"groupby_{agg_name}_idxmax",
            "execution_success": True,
        }

    # ------------------------------------------------------------------
    # Single chart grouping
    # ------------------------------------------------------------------

    def execute_chart(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pandas grouping for a single chart specification.

        Args:
            chart_config: dict with keys title, x_axis_column, y_axis_column,
                          pandas_grouping, …

        Returns:
            Result dict with data (dict), execution_success, optional error.
        """
        title = chart_config.get("title", chart_config.get("chart_title", "Chart"))
        pandas_grouping = chart_config.get("pandas_grouping", "")

        x_col = chart_config.get("x_axis_column")
        y_col = chart_config.get("y_axis_column")

        try:
            lowered = pandas_grouping.lower()

            if "crosstab" in lowered:
                return self._execute_chart_crosstab(title, pandas_grouping)

            if "scatter" in lowered:
                return self._execute_chart_scatter(title, pandas_grouping, x_col, y_col)

            normalized = self._strip_trailing_chain(pandas_grouping)
            group_col, agg_col, agg_func = self._parse_grouping(normalized)

            group_col = group_col or x_col
            agg_col = agg_col or y_col

            missing = []
            if group_col and group_col not in self.df.columns:
                missing.append(f"groupby column '{group_col}'")
            if agg_col and agg_col not in self.df.columns:
                missing.append(f"agg column '{agg_col}'")
            if missing:
                raise ValueError(f"Not found in dataset: {', '.join(missing)}")

            if not (group_col and agg_col and agg_func):
                raise ValueError(
                    f"Could not parse grouping pattern from: '{pandas_grouping}'. "
                    "Expected format: df.groupby('X')['Y'].func() (optional .sort_values / .head after)"
                )

            safe_name = agg_func.lower()
            safe_fn = self.SAFE_AGGREGATIONS.get(safe_name)
            if not safe_fn:
                raise ValueError(
                    f"Unsupported aggregation '{agg_func}'. "
                    f"Supported: {list(self.SAFE_AGGREGATIONS)}"
                )

            tmp_df = self.df.copy()
            tmp_df[agg_col] = pd.to_numeric(tmp_df[agg_col], errors="coerce")

            pandas_agg = "mean" if safe_name == "avg" else safe_name
            grouped = tmp_df.groupby(group_col)[agg_col].agg(pandas_agg)
            keys = [str(k) for k in grouped.index]
            vals = [(float(v) if pd.notnull(v) else None) for v in grouped.values]

            return {
                "chart_title": title,
                "x_axis": group_col,
                "y_axis": agg_col,
                "aggregation": safe_name,
                "point_count": len(keys),
                "data": {"axis_x": keys, "axis_y": vals},
                "execution_success": True,
            }

        except Exception as exc:
            logger.error("Chart execution failed for '%s': %s", title, exc)
            return {
                "chart_title": title,
                "x_axis": None,
                "y_axis": None,
                "aggregation": None,
                "data": {},
                "error": str(exc),
                "execution_success": False,
            }

    @staticmethod
    def _strip_trailing_chain(s: str) -> str:
        """Remove .sort_values(...), .head(...), etc. so core groupby chain remains."""
        s = s.strip()
        cut_markers = (
            ".sort_values(",
            ".sort_index(",
            ".head(",
            ".tail(",
            ".reset_index(",
            ".nlargest(",
            ".nsmallest(",
        )
        while True:
            best = -1
            for m in cut_markers:
                idx = s.find(m)
                if idx != -1 and (best == -1 or idx < best):
                    best = idx
            if best == -1:
                break
            s = s[:best].rstrip()
        return s.strip()

    def _execute_chart_scatter(
        self,
        title: str,
        pandas_grouping: str,
        x_fallback: Optional[str],
        y_fallback: Optional[str],
    ) -> Dict[str, Any]:
        x_m = re.search(r"""x\s*=\s*['\"]([^'\"]+)['\"]""", pandas_grouping)
        y_m = re.search(r"""y\s*=\s*['\"]([^'\"]+)['\"]""", pandas_grouping)
        x_col = (x_m.group(1) if x_m else None) or x_fallback
        y_col = (y_m.group(1) if y_m else None) or y_fallback
        if not x_col or not y_col:
            raise ValueError(
                f"Could not parse scatter columns from: '{pandas_grouping}'. "
                "Expected df.plot.scatter(x='X', y='Y')"
            )
        for c in (x_col, y_col):
            if c not in self.df.columns:
                raise ValueError(f"Column '{c}' not found")

        xs = pd.to_numeric(self.df[x_col], errors="coerce")
        ys = pd.to_numeric(self.df[y_col], errors="coerce")
        valid = xs.notna() & ys.notna()
        x_vals = [float(xs.iat[i]) for i in range(len(self.df)) if valid.iat[i]]
        y_vals = [float(ys.iat[i]) for i in range(len(self.df)) if valid.iat[i]]
        return {
            "chart_title": title,
            "x_axis": x_col,
            "y_axis": y_col,
            "aggregation": "scatter",
            "point_count": len(x_vals),
            "data": {"axis_x": x_vals, "axis_y": y_vals},
            "execution_success": True,
        }

    def _execute_chart_crosstab(self, title: str, pandas_grouping: str) -> Dict[str, Any]:
        """
        Parse pd.crosstab(df['A'], df['B'], ...) and return normalized matrix as nested dict.
        """
        cols = re.findall(r"""df\s*\[\s*['\"]([^'\"]+)['\"]\s*\]""", pandas_grouping)
        if len(cols) < 2:
            raise ValueError(
                f"Could not parse crosstab columns from: '{pandas_grouping}'. "
                "Expected pd.crosstab(df['A'], df['B'], ...)"
            )
        a, b = cols[0], cols[1]
        for c in (a, b):
            if c not in self.df.columns:
                raise ValueError(f"Column '{c}' not found")

        normalize = "index" if re.search(r"normalize\s*=\s*['\"]index['\"]", pandas_grouping) else None
        ct = pd.crosstab(self.df[a], self.df[b], normalize=normalize)
        labels_x = [str(idx) for idx in ct.index]
        labels_y = [str(col) for col in ct.columns]
        matrix = [
            [(float(ct.at[idx, col]) if pd.notnull(ct.at[idx, col]) else None) for col in ct.columns]
            for idx in ct.index
        ]
        return {
            "chart_title": title,
            "x_axis": a,
            "y_axis": b,
            "aggregation": "crosstab",
            "point_count": len(labels_x) * len(labels_y),
            "data": {"labels_x": labels_x, "labels_y": labels_y, "matrix": matrix},
            "execution_success": True,
        }

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def execute_all_kpis(self, kpi_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute all KPI configs and return a list of results."""
        return [self.execute_kpi(k) for k in kpi_configs]

    def execute_all_charts(self, chart_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute all chart configs and return a list of results."""
        return [self.execute_chart(c) for c in chart_configs]

    def execute_all_charts_metadata(
        self, chart_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute charts but return metadata only (no data payload)."""
        results = []
        for idx, cfg in enumerate(chart_configs):
            full = self.execute_chart(cfg)
            results.append({
                "chart_index": idx,
                "chart_title": full.get("chart_title"),
                "chart_type": cfg.get("chart_type"),
                "x_axis": full.get("x_axis"),
                "y_axis": full.get("y_axis"),
                "aggregation": full.get("aggregation"),
                "point_count": full.get("point_count", 0),
                "insight": cfg.get("business_insight", ""),
                "execution_success": full.get("execution_success", False),
                "error": full.get("error"),
            })
        return results

    def execute_single_chart(self, chart_configs: List[Dict[str, Any]], chart_index: int) -> Dict[str, Any]:
        """Execute and return full data for a single chart by index."""
        if chart_index < 0 or chart_index >= len(chart_configs):
            raise IndexError(f"chart_index {chart_index} out of range (0..{len(chart_configs) - 1})")
        return self.execute_chart(chart_configs[chart_index])

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_function(pandas_function: str, logic_hint: str = "") -> str:
        """
        Extract the aggregation name from a pandas function string.
        ".mean()" → "mean", "sum" → "sum"
        """
        for src in (pandas_function, logic_hint):
            match = re.search(r"\.(\w+)\s*\(", src)
            if match:
                return match.group(1).lower()
        return (pandas_function or "").strip(".() ").lower()

    @staticmethod
    def _parse_grouping(
        pandas_grouping: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse a groupby one-liner into its components.
        "df.groupby('Company')['Price'].mean()" → ("Company", "Price", "mean")
        """
        group_match = re.search(r"groupby\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", pandas_grouping)
        group_col = group_match.group(1) if group_match else None

        agg_col_matches = re.findall(r"\[\s*['\"]([^'\"]+)['\"]\s*\]", pandas_grouping)
        agg_col = agg_col_matches[-1] if agg_col_matches else None

        func_match = re.search(r"\.(\w+)\s*\(\s*\)\s*$", pandas_grouping.strip())
        agg_func = func_match.group(1).lower() if func_match else None

        return group_col, agg_col, agg_func

    @staticmethod
    def _format_value(value: float) -> str:
        """Format a numeric value for human-readable display."""
        if pd.isna(value):
            return "N/A"
        abs_val = abs(value)
        if abs_val >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs_val >= 1_000:
            return f"{value / 1_000:.2f}K"
        return f"{value:.2f}"

    @staticmethod
    def _kpi_error(kpi_name: str, target_column: Optional[str], error: str) -> Dict[str, Any]:
        return {
            "kpi_name": kpi_name,
            "target_column": target_column,
            "value": None,
            "formatted_value": None,
            "function_used": None,
            "error": error,
            "execution_success": False,
        }
