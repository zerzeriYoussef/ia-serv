"""
"""



from typing import Dict, List, Optional

import logging



import numpy as np

import pandas as pd

from scipy.stats import chi2_contingency, f_oneway

from statsmodels.stats.anova import anova_lm

import statsmodels.formula.api as smf



from app.services.analysis.column_analyzer import ColumnAnalyzer





logger = logging.getLogger(__name__)





class ImprovedColumnAnalyzer(ColumnAnalyzer):

    """
    Column analyzer with additional statistical relationship detection.

    This class augments the base ``ColumnAnalyzer`` by:
    - Running chi-square tests for categorical–categorical pairs
    - Running one-way ANOVA for numeric–categorical pairs
    - Computing effect sizes:
        - Cramér's V for categorical–categorical
        - Eta-squared for numeric–categorical
    - Filtering relationships by both statistical and practical significance.

    Example
    -------
    >>> analyzer = ImprovedColumnAnalyzer(df)
    >>> result = analyzer.analyze()
    >>> result["relationships"][0]["p_value"]
    0.0012
    >>> result["relationships"][0]["strength"]  # effect size
    0.32
    """





    _P_THRESHOLD: float = 0.05

    _CRAMERS_V_THRESHOLD: float = 0.10

    _ETA_SQUARED_THRESHOLD: float = 0.06



    def analyze(self) -> Dict:

        """
        Run full analysis with improved relationship detection.

        Returns the same structure as ``ColumnAnalyzer.analyze``, but
        relationships additionally include:
        - ``p_value``
        - ``effect_size`` (same as ``strength``)
        - ``statistic``
        - ``significance`` ("*", "**", or "***")
        """



        logger.info(

            f"[Improved] Analyzing {len(self.df.columns)} columns "

            f"from {len(self.df)} rows..."

        )





        pps_relationships = self._detect_relationships()





        correlation_relationships = self._get_correlations()





        cat_cat_relationships = self._detect_categorical_associations()

        num_cat_relationships = self._detect_numeric_categorical()





        self._normalize_relationships(pps_relationships, default_method="custom_pps")

        self._normalize_relationships(

            correlation_relationships, default_method="pearson"

        )





        all_relationships = self._merge_multiple_relationship_sets(

            pps_relationships,

            correlation_relationships,

            cat_cat_relationships,

            num_cat_relationships,

        )





        dashboard_columns = self._select_dashboard_columns()

        primary_metric = self._identify_primary_metric()

        confidence = self._calculate_confidence(all_relationships)



        result = {

            "relationships": all_relationships,

            "dashboard_columns": dashboard_columns,

            "column_categories": self.column_types,

            "primary_metric": primary_metric,

            "confidence_score": confidence,

            "total_relationships": len(all_relationships),

        }



        logger.info(

            f"[Improved] Analysis complete: {len(all_relationships)} relationships found"

        )



        return result











    def _detect_categorical_associations(self) -> List[Dict]:

        """
        Detect categorical–categorical relationships using chi-square tests.

        Uses:
        - ``scipy.stats.chi2_contingency`` for the chi-square test
        - Cramér's V as effect size

        Filtering:
        - p-value < 0.05
        - Cramér's V > 0.10
        """



        dimensions = self.column_types.get("dimensions", [])

        if len(dimensions) < 2:

            return []



        relationships: List[Dict] = []



        for i in range(len(dimensions)):

            for j in range(i + 1, len(dimensions)):

                col1 = dimensions[i]

                col2 = dimensions[j]



                try:

                    data = self.df[[col1, col2]].dropna()

                    if data[col1].nunique() < 2 or data[col2].nunique() < 2:

                        continue



                    contingency = pd.crosstab(data[col1], data[col2])





                    if contingency.size == 0:

                        continue





                    if (contingency.values < 5).any():

                        continue



                    chi2, p_value, _, _ = chi2_contingency(

                        contingency, correction=False

                    )



                    cramers_v = self._calculate_cramers_v(chi2, contingency)

                    if cramers_v is None:

                        continue





                    if p_value >= self._P_THRESHOLD:

                        continue

                    if cramers_v <= self._CRAMERS_V_THRESHOLD:

                        continue



                    relationships.append(

                        {

                            "columns": [col1, col2],

                            "type": "categorical_association",

                            "strength": round(float(cramers_v), 3),

                            "effect_size": round(float(cramers_v), 3),

                            "p_value": float(p_value),

                            "method": "chi_square",

                            "statistic": float(chi2),

                            "significance": self._significance_stars(p_value),

                            "direction": "association",

                        }

                    )

                except Exception as exc:

                    logger.debug(

                        f"Chi-square failed for {col1} ~ {col2}: {exc}", exc_info=False

                    )

                    continue



        logger.info(

            f"Found {len(relationships)} categorical associations (chi-square)"

        )

        return relationships



    def _detect_numeric_categorical(self) -> List[Dict]:

        """
        Detect numeric–categorical relationships using one-way ANOVA.

        Uses:
        - ``scipy.stats.f_oneway`` for F-statistic and p-value
        - ``statsmodels.stats.anova.anova_lm`` + OLS to compute eta-squared

        Filtering:
        - p-value < 0.05
        - eta-squared > 0.06 (medium effect)

        Both "numeric predicts category" and "category predicts numeric" are
        represented by a single relationship with ``direction`` set to
        ``"{category} → {numeric}"``.
        """



        metrics = self.column_types.get("metrics", [])

        dimensions = self.column_types.get("dimensions", [])



        if not metrics or not dimensions:

            return []



        relationships: List[Dict] = []



        for num_col in metrics:

            for cat_col in dimensions:

                try:

                    data = self.df[[num_col, cat_col]].dropna()





                    if data[cat_col].nunique() < 2:

                        continue



                    groups = []

                    for _, group_df in data.groupby(cat_col):

                        values = group_df[num_col].dropna().to_numpy()



                        if values.size < 2:

                            continue

                        groups.append(values)



                    if len(groups) < 2:

                        continue





                    f_stat, p_value = f_oneway(*groups)





                    eta_sq = self._calculate_eta_squared(num_col, cat_col, data)

                    if eta_sq is None:

                        continue





                    if p_value >= self._P_THRESHOLD:

                        continue

                    if eta_sq <= self._ETA_SQUARED_THRESHOLD:

                        continue



                    relationships.append(

                        {

                            "columns": [cat_col, num_col],

                            "type": "numeric_categorical",

                            "strength": round(float(eta_sq), 3),

                            "effect_size": round(float(eta_sq), 3),

                            "p_value": float(p_value),

                            "method": "anova",

                            "statistic": float(f_stat),

                            "significance": self._significance_stars(p_value),

                            "direction": f"{cat_col} → {num_col}",

                        }

                    )

                except Exception as exc:

                    logger.debug(

                        f"ANOVA failed for {cat_col} ~ {num_col}: {exc}", exc_info=False

                    )

                    continue



        logger.info(

            f"Found {len(relationships)} numeric–categorical relationships (ANOVA)"

        )

        return relationships











    def _calculate_cramers_v(

        self, chi2: float, contingency: pd.DataFrame

    ) -> Optional[float]:

        """
        Compute Cramér's V effect size for a contingency table.

        V = sqrt(chi2 / (n * min(r - 1, c - 1)))
        """



        n = contingency.to_numpy().sum()

        if n <= 0:

            return None



        r, c = contingency.shape

        denom = n * float(min(r - 1, c - 1))

        if denom <= 0:

            return None



        v = np.sqrt(chi2 / denom)

        return float(v)



    def _calculate_eta_squared(

        self, numeric_col: str, category_col: str, data: pd.DataFrame

    ) -> Optional[float]:

        """
        Compute eta-squared effect size for one-way ANOVA.

        Uses statsmodels OLS + ``anova_lm`` to obtain:
        eta-squared = SS_between / SS_total
        """



        try:

            model = smf.ols(

                f"{numeric_col} ~ C({category_col})", data=data

            ).fit()

            anova_table = anova_lm(model, typ=2)





            ss_between = float(anova_table["sum_sq"].iloc[0])

            ss_resid = float(anova_table["sum_sq"].iloc[1])

            ss_total = ss_between + ss_resid



            if ss_total <= 0:

                return None



            eta_sq = ss_between / ss_total

            return float(eta_sq)

        except Exception as exc:

            logger.debug(

                f"Failed to compute eta-squared for {category_col} ~ {numeric_col}: {exc}",

                exc_info=False,

            )

            return None



    @staticmethod

    def _significance_stars(p_value: float) -> str:

        """
        Map p-value to significance stars.

        - '***' for p < 0.001
        - '**'  for p < 0.01
        - '*'   for p < 0.05
        - ''    otherwise
        """



        if p_value < 0.001:

            return "***"

        if p_value < 0.01:

            return "**"

        if p_value < 0.05:

            return "*"

        return ""



    def _normalize_relationships(

        self, relationships: List[Dict], default_method: str

    ) -> None:

        """
        Ensure relationships contain p_value/effect_size/statistic fields.

        Existing PPS and correlation relationships don't have statistical
        tests, so we:
        - Set ``effect_size`` = ``strength``
        - Set ``p_value`` and ``statistic`` to None
        - Leave ``significance`` empty
        """



        for rel in relationships:

            rel.setdefault("method", default_method)

            rel.setdefault("effect_size", rel.get("strength"))

            rel.setdefault("p_value", None)

            rel.setdefault("statistic", None)

            rel.setdefault("significance", "")



    def _merge_multiple_relationship_sets(

        self, *rel_lists: List[List[Dict]]

    ) -> List[Dict]:

        """
        Merge multiple relationship lists, deduplicating by column pair.

        Later lists win when the same pair appears multiple times.
        """



        merged: List[Dict] = []

        seen_pairs = {}



        for rel_list in rel_lists:

            for rel in rel_list:

                pair = tuple(sorted(rel.get("columns", [])))

                if not pair:

                    continue



                seen_pairs[pair] = rel





        merged = list(seen_pairs.values())

        merged.sort(key=lambda x: x.get("strength", 0.0), reverse=True)



        return merged



