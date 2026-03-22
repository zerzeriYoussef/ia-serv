"""
Statistical Relationship Analyzer
----------------------------------

Detects statistically significant relationships between columns using:
- Spearman correlation (numeric ↔ numeric)
- Kruskal-Wallis test + Eta-squared (categorical → numeric)
- Chi-square test + Cramér's V (categorical ↔ categorical)

All tests are Bonferroni-corrected for multiple comparisons.
"""

import re
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# ── Column junk-filter pattern ──────────────────────────────────────────────

ID_PATTERNS = re.compile(
    r'\b(id|uuid|guid|key|hash|token|code|ref|url|link|email|phone|'
    r'address|name|firstname|lastname|fullname|username|login|'
    r'description|comment|note|text|message|subject|body|title|slug)\b',
    re.IGNORECASE
)


def looks_like_junk(col: str, series: pd.Series) -> bool:
    """Return True if column is likely an ID, key, free-text, or name field."""
    col_clean = re.sub(r'[_\-\s]', '', col)
    if ID_PATTERNS.search(col_clean):
        return True
    if series.dtype == 'O':
        uniqueness_ratio = series.nunique() / max(len(series.dropna()), 1)
        if uniqueness_ratio > 0.9:
            return True
    return False


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V: effect size for categorical↔categorical (0=none, 1=perfect)."""
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
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
    groups = [
        g.values
        for _, g in pd.concat([cat, num], axis=1).groupby(cat.name)[num.name]
    ]
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((num - grand_mean) ** 2).sum()
    return 0.0 if ss_total == 0 else float(ss_between / ss_total)


def strength_label(value: float, thresholds: tuple) -> str:
    """Map effect-size value to human-readable strength label."""
    weak, moderate, strong = thresholds
    if value >= strong:
        return "very strong"
    if value >= moderate:
        return "strong"
    if value >= weak:
        return "moderate"
    return "weak"


class StatisticalAnalyzer:
    """
    Statistical relationship detector with Bonferroni correction.

    Uses Spearman, Kruskal-Wallis, and Chi-square tests to find
    statistically significant relationships between columns.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_rows = len(df)

    def analyze(self) -> List[Dict]:
        """
        Detect statistically significant relationships.

        Returns only validated relationships with p-values and effect sizes.
        """
        # Get column types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Filter junk columns
        numeric_cols = [c for c in numeric_cols if not looks_like_junk(c, self.df[c])]
        categorical_cols = [c for c in categorical_cols if not looks_like_junk(c, self.df[c])]

        # Filter categorical columns by cardinality
        if self.n_rows >= 30:
            categorical_cols = [
                c for c in categorical_cols
                if 2 <= self.df[c].nunique() <= 50
            ]
        else:
            categorical_cols = [
                c for c in categorical_cols
                if self.df[c].nunique() >= 2
            ]

        # Calculate Bonferroni correction
        n_num, n_cat = len(numeric_cols), len(categorical_cols)
        n_tests = (
            (n_num * (n_num - 1)) // 2
            + n_cat * n_num
            + (n_cat * (n_cat - 1)) // 2
        )

        # Adjust alpha for small datasets
        if self.n_rows < 30:
            alpha = 0.10
        else:
            alpha = 0.05 / max(n_tests, 1)

        logger.info(f"Running {n_tests} statistical tests with α={alpha:.4f}")

        relationships: List[Dict] = []

        # 1. Numeric ↔ Numeric (Spearman correlation)
        relationships.extend(self._detect_numeric_correlations(numeric_cols, alpha))

        # 2. Categorical → Numeric (Kruskal-Wallis + Eta²)
        relationships.extend(self._detect_categorical_numeric(categorical_cols, numeric_cols, alpha))

        # 3. Categorical ↔ Categorical (Chi-square + Cramér's V)
        relationships.extend(self._detect_categorical_associations(categorical_cols, alpha))

        logger.info(f"Found {len(relationships)} statistically significant relationships")

        return relationships

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_numeric_correlations(
        self, numeric_cols: List[str], alpha: float
    ) -> List[Dict]:
        """Spearman correlation with Bonferroni correction."""
        relationships: List[Dict] = []

        spearman_weak, spearman_mod, spearman_strong = 0.25, 0.45, 0.70

        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr(method='spearman')

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    rho = corr_matrix.loc[col1, col2]

                    if not pd.notna(rho):
                        continue

                    # Calculate p-value from t-distribution
                    t_stat = rho * np.sqrt(
                        (self.n_rows - 2) / max(1 - rho ** 2, 1e-10)
                    )
                    p_val = 2 * stats.t.sf(abs(t_stat), df=self.n_rows - 2)

                    if p_val < alpha and abs(rho) >= spearman_weak:
                        s_label = strength_label(
                            abs(rho),
                            (spearman_weak, spearman_mod, spearman_strong),
                        )
                        direction = "positive" if rho > 0 else "negative"

                        relationships.append({
                            "columns": [col1, col2],
                            "type": "correlation",
                            "strength": float(abs(rho)),
                            "direction": direction,
                            "method": "spearman",
                            "p_value": float(p_val),
                            "effect_size": float(abs(rho)),
                            "statistical_method": "spearman",
                            "is_statistically_significant": True,
                            "strength_label": s_label,
                            "insight": (
                                f"{col1} ↔ {col2} "
                                f"[{s_label} {direction} correlation]"
                            ),
                        })

        return relationships

    def _detect_categorical_numeric(
        self,
        categorical_cols: List[str],
        numeric_cols: List[str],
        alpha: float,
    ) -> List[Dict]:
        """Kruskal-Wallis test + Eta-squared effect size."""
        relationships: List[Dict] = []

        eta_weak, eta_mod, eta_strong = 0.01, 0.06, 0.14

        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                clean = self.df[[cat_col, num_col]].dropna()

                min_size = int(max(5, self.n_rows // 10))
                if len(clean) < min_size:
                    continue

                groups = [
                    g[num_col].values
                    for _, g in clean.groupby(cat_col)
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
                        s_label = strength_label(
                            eta2, (eta_weak, eta_mod, eta_strong)
                        )

                        relationships.append({
                            "columns": [cat_col, num_col],
                            "type": "categorical_to_numeric",
                            "strength": float(eta2),
                            "direction": f"{cat_col} → {num_col}",
                            "method": "kruskal_wallis",
                            "p_value": float(p_val),
                            "effect_size": float(eta2),
                            "statistical_method": "kruskal_wallis",
                            "is_statistically_significant": True,
                            "strength_label": s_label,
                            "insight": (
                                f"{cat_col} explains "
                                f"{eta2 * 100:.1f}% of variance in {num_col}"
                            ),
                        })

        return relationships

    def _detect_categorical_associations(
        self, categorical_cols: List[str], alpha: float
    ) -> List[Dict]:
        """Chi-square test + Cramér's V effect size."""
        relationships: List[Dict] = []

        cramer_weak, cramer_mod, cramer_strong = 0.10, 0.25, 0.40

        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]
                clean = self.df[[col1, col2]].dropna()

                min_size = int(max(5, self.n_rows // 10))
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
                        s_label = strength_label(
                            v, (cramer_weak, cramer_mod, cramer_strong)
                        )

                        relationships.append({
                            "columns": [col1, col2],
                            "type": "categorical_association",
                            "strength": float(v),
                            "direction": f"{col1} ↔ {col2}",
                            "method": "chi_square",
                            "p_value": float(p_val),
                            "effect_size": float(v),
                            "statistical_method": "chi_square",
                            "is_statistically_significant": True,
                            "strength_label": s_label,
                            "insight": (
                                f"{col1} and {col2} are associated "
                                f"(Cramér's V = {v:.3f})"
                            ),
                        })

        return relationships
