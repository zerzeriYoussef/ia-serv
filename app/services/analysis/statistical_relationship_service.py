"""
Statistical relationship detection across numeric and categorical columns.

Ports the Spearman / Kruskal-Wallis+η² / Chi-square+Cramér's V pipeline
with Bonferroni-style alpha and junk-column heuristics (no console output).
"""



from __future__ import annotations



import re

import warnings

from typing import Any, Dict, List, Optional, Tuple



import numpy as np

import pandas as pd

import scipy.stats as stats



warnings.filterwarnings("ignore")



ID_PATTERNS = re.compile(

    r"\b(id|uuid|guid|key|hash|token|code|ref|url|link|email|phone|"

    r"address|name|firstname|lastname|fullname|username|login|"

    r"description|comment|note|text|message|subject|body|title|slug)\b",

    re.IGNORECASE,

)





def looks_like_junk(col: str, series: pd.Series) -> bool:

    """Return True if this column is likely an ID, key, free-text, or name field."""

    col_clean = re.sub(r"[_\-\s]", "", col)

    if ID_PATTERNS.search(col_clean):

        return True

    if series.dtype == "O":

        uniqueness_ratio = series.nunique() / max(len(series.dropna()), 1)

        if uniqueness_ratio > 0.9:

            return True

    return False





def cramers_v(x: pd.Series, y: pd.Series) -> float:

    """Cramér's V: effect size for categorical↔categorical (0=none, 1=perfect)."""

    ct = pd.crosstab(x, y)

    chi2 = stats.chi2_contingency(ct, correction=False)[0]

    n = ct.values.sum()

    if n == 0:

        return 0.0

    phi2 = chi2 / n

    r, k = ct.shape

    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))

    r_corr = r - (r - 1) ** 2 / (n - 1)

    k_corr = k - (k - 1) ** 2 / (n - 1)

    denom = min(r_corr - 1, k_corr - 1)

    return 0.0 if denom <= 0 else float(np.sqrt(phi2_corr / denom))





def eta_squared(cat: pd.Series, num: pd.Series) -> float:

    """Eta-squared (η²): proportion of variance in numeric explained by category."""

    grand_mean = num.mean()

    combined = pd.DataFrame({"_cat": cat, "_num": num}).dropna()

    if combined.empty:

        return 0.0

    ss_between = 0.0

    for _, g in combined.groupby("_cat", observed=False):

        vals = g["_num"]

        if len(vals) == 0:

            continue

        ss_between += len(vals) * (float(vals.mean()) - grand_mean) ** 2

    ss_total = float(((num - grand_mean) ** 2).sum())

    return 0.0 if ss_total == 0 else float(ss_between / ss_total)





def strength_label(value: float, thresholds: Tuple[float, float, float]) -> str:

    """Map an effect-size value to a human-readable strength label."""

    weak, moderate, strong = thresholds

    if value >= strong:

        return "very strong"

    if value >= moderate:

        return "strong"

    if value >= weak:

        return "moderate"

    return "weak"





def detect_relationships(df: pd.DataFrame, label: str) -> Dict[str, Any]:

    """
    Run full statistical relationship detection on a single dataframe.

    Returns a JSON-serializable dict (no side effects).
    """

    df = df.copy()

    df = df.dropna(axis=1, how="all")

    df = df.loc[:, df.nunique() > 1]



    if df.empty:

        return {

            "label": label,

            "shape": {"rows": 0, "columns": 0},

            "alpha": None,

            "n_tests": 0,

            "relationships": [],

            "sections": {},

            "total_meaningful": 0,

            "message": "No valid data to analyse.",

        }



    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()



    numeric_cols = [c for c in numeric_cols if not looks_like_junk(c, df[c])]

    categorical_cols = [c for c in categorical_cols if not looks_like_junk(c, df[c])]



    n_rows = len(df)



    if n_rows >= 30:

        categorical_cols = [

            c for c in categorical_cols if 2 <= df[c].nunique() <= 50

        ]

    else:

        categorical_cols = [c for c in categorical_cols if df[c].nunique() >= 2]



    relationships: List[Tuple[float, str, str]] = []



    n_num, n_cat = len(numeric_cols), len(categorical_cols)

    n_tests = (

        (n_num * (n_num - 1)) // 2

        + n_cat * n_num

        + (n_cat * (n_cat - 1)) // 2

    )



    if n_rows < 30:

        alpha = 0.10

    else:

        alpha = 0.05 / max(n_tests, 1)



    spearman_weak, spearman_mod, spearman_strong = 0.25, 0.45, 0.70



    if n_num > 1:

        corr_matrix = df[numeric_cols].corr(method="spearman")

        for i in range(n_num):

            for j in range(i + 1, n_num):

                col1, col2 = numeric_cols[i], numeric_cols[j]

                rho = corr_matrix.loc[col1, col2]

                if not pd.notna(rho):

                    continue

                t_stat = rho * np.sqrt((n_rows - 2) / max(1 - rho**2, 1e-10))

                p_val = 2 * stats.t.sf(abs(t_stat), df=n_rows - 2)

                if p_val < alpha and abs(rho) >= spearman_weak:

                    s = strength_label(

                        abs(rho), (spearman_weak, spearman_mod, spearman_strong)

                    )

                    direction = "positive" if rho > 0 else "negative"

                    desc = (

                        f"{col1} ↔ {col2}  "

                        f"[Spearman ρ = {rho:+.2f}, {s} {direction} correlation]"

                    )

                    relationships.append((abs(rho), "Numeric ↔ Numeric", desc))



    eta_weak, eta_mod, eta_strong = 0.01, 0.06, 0.14



    for cat_col in categorical_cols:

        for num_col in numeric_cols:

            clean = df[[cat_col, num_col]].dropna()

            min_size = int(max(5, n_rows // 10))

            if len(clean) < min_size:

                continue



            groups = [

                g[num_col].values

                for _, g in clean.groupby(cat_col, observed=False)

                if len(g) >= 2

            ]

            if len(groups) < 2:

                continue

            try:

                _, p_val = stats.kruskal(*groups)

            except Exception:

                continue

            if p_val < alpha:

                eta2 = eta_squared(clean[cat_col], clean[num_col])

                if eta2 >= eta_weak:

                    s = strength_label(eta2, (eta_weak, eta_mod, eta_strong))

                    desc = (

                        f"{cat_col} → {num_col}  "

                        f"[Kruskal-Wallis p={p_val:.2e}, η² = {eta2:.3f}, {s} effect]"

                    )

                    relationships.append((eta2, "Categorical → Numeric", desc))



    cramer_weak, cramer_mod, cramer_strong = 0.10, 0.25, 0.40



    for i in range(n_cat):

        for j in range(i + 1, n_cat):

            col1, col2 = categorical_cols[i], categorical_cols[j]

            clean = df[[col1, col2]].dropna()

            min_size = int(max(5, n_rows // 10))

            if len(clean) < min_size:

                continue



            ct = pd.crosstab(clean[col1], clean[col2])

            if ct.shape[0] < 2 or ct.shape[1] < 2:

                continue

            try:

                _, p_val, _, _ = stats.chi2_contingency(ct)

            except Exception:

                continue

            if p_val < alpha:

                v = cramers_v(clean[col1], clean[col2])

                if v >= cramer_weak:

                    s = strength_label(v, (cramer_weak, cramer_mod, cramer_strong))

                    desc = (

                        f"{col1} ↔ {col2}  "

                        f"[Chi-square p={p_val:.2e}, Cramér's V = {v:.3f}, {s} association]"

                    )

                    relationships.append((v, "Categorical ↔ Categorical", desc))



    if not relationships:

        return {

            "label": label,

            "shape": {"rows": n_rows, "columns": int(df.shape[1])},

            "alpha": float(alpha),

            "n_tests": int(n_tests),

            "relationships": [],

            "sections": {},

            "total_meaningful": 0,

            "message": (

                "No meaningful relationships found after filtering and correction."

            ),

        }



    relationships.sort(key=lambda x: x[0], reverse=True)



    sections: Dict[str, List[str]] = {}

    for effect, rtype, desc in relationships:

        sections.setdefault(rtype, []).append(desc)



    rel_objects = [

        {"effect_size": float(effect), "relationship_type": rtype, "description": desc}

        for effect, rtype, desc in relationships

    ]



    return {

        "label": label,

        "shape": {"rows": n_rows, "columns": int(df.shape[1])},

        "alpha": float(alpha),

        "n_tests": int(n_tests),

        "relationships": rel_objects,

        "sections": sections,

        "total_meaningful": len(relationships),

        "message": (

            f"Total: {len(relationships)} meaningful relationship(s) found "

            f"(from {n_tests} tests run)."

        ),

    }





def merge_datasets_on_common_columns(

    loaded: List[Tuple[int, str, pd.DataFrame]],

) -> Tuple[Optional[pd.DataFrame], List[str], List[int]]:

    """
    Sequentially outer-merge dataframes on all common column names (same as script).

    Returns (merged_df or None, log lines, dataset ids that entered the merge chain).
    """

    if len(loaded) < 2:

        return None, [], []



    log: List[str] = []

    merged_df = loaded[0][2].copy()

    merged_ids = [loaded[0][0]]



    for dataset_id, fp, df in loaded[1:]:

        common_cols = list(set(merged_df.columns) & set(df.columns))

        if common_cols:

            merged_df = pd.merge(merged_df, df, on=common_cols, how="outer")

            merged_ids.append(dataset_id)

            log.append(

                f"Merged dataset {dataset_id} ({fp}) on common columns: {common_cols}"

            )

        else:

            log.append(

                f"Skipped dataset {dataset_id} ({fp}) — no common columns to merge on"

            )



    if len(merged_ids) < 2:

        log.append("Could not find any common columns to join multiple datasets.")

        return None, log, merged_ids



    return merged_df, log, merged_ids





def run_multi_dataset_analytics(

    loaded: List[Tuple[int, str, pd.DataFrame]],

) -> Dict[str, Any]:

    """
    Run per-dataset analytics, then merged analytics if possible.

    `loaded` is list of (dataset_id, display_label, dataframe).
    """

    per_dataset: List[Dict[str, Any]] = []

    for dataset_id, label, df in loaded:

        result = detect_relationships(df, label)

        result["dataset_id"] = dataset_id

        per_dataset.append(result)



    merged_block: Optional[Dict[str, Any]] = None

    if len(loaded) > 1:

        merged_df, merge_log, merged_ids = merge_datasets_on_common_columns(loaded)

        if merged_df is not None and len(merged_ids) > 1:

            merged_block = {

                "merge_log": merge_log,

                "dataset_ids": merged_ids,

                "analysis": detect_relationships(

                    merged_df, "MERGED_MULTIPLE_DATASETS"

                ),

            }

        else:

            merged_block = {

                "merge_log": merge_log,

                "dataset_ids": merged_ids,

                "analysis": None,

                "message": "Merged analytics skipped — not enough overlap to merge.",

            }



    return {

        "per_dataset": per_dataset,

        "merged": merged_block,

    }

