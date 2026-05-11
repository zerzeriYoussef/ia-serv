"""
Column Analysis Service
Analyzes dataset columns and detects relationships using CustomPPS
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import re

from app.services.analysis.custom_pps import CustomPPS
from app.services.analysis.relationship_detector import RelationshipDetector

logger = logging.getLogger(__name__)


class ColumnAnalyzer:
    HELPER_METRIC_PATTERNS = re.compile(
        r"(outlier|^is_|^has_|flag|indicator|dummy|onehot|encoded)",
        re.IGNORECASE,
    )

    INDEX_LIKE_PATTERNS = re.compile(r"^(unnamed:?\s*\d+|index|row_?id)$", re.IGNORECASE)

    """
    Intelligent column analysis using modern libraries
    
    Features:
    - Column categorization (metrics, dimensions, identifiers, etc.)
    - Relationship detection using CustomPPS
    - Dashboard column recommendations
    - Primary metric identification
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_types = self._categorize_columns()
    
    def analyze(self) -> Dict:
        """
        Main analysis method
        
        Returns:
            {
                "relationships": [...],
                "dashboard_columns": [...],
                "column_categories": {...},
                "primary_metric": "...",
                "confidence_score": 0.85
            }
        """
        
        logger.info(f"Analyzing {len(self.df.columns)} columns from {len(self.df)} rows...")
        
        # 1. Detect relationships using CustomPPS
        relationships = self._detect_relationships()
        
        # 2. Get correlation matrix (fast fallback)
        correlations = self._get_correlations()
        
        # 3. Combine results
        all_relationships = self._merge_relationships(relationships, correlations)
        
        # 4. Select dashboard columns
        dashboard_columns = self._select_dashboard_columns()
        
        # 5. Identify primary metric
        primary_metric = self._identify_primary_metric()
        
        # 6. Calculate confidence
        confidence = self._calculate_confidence(all_relationships)
        
        result = {
            "relationships": all_relationships,
            "dashboard_columns": dashboard_columns,
            "column_categories": self.column_types,
            "primary_metric": primary_metric,
            "confidence_score": confidence,
            "total_relationships": len(all_relationships)
        }
        
        logger.info(f"Analysis complete: {len(all_relationships)} relationships found")
        
        return result
    
    def _categorize_columns(self) -> Dict[str, List[str]]:
        """Categorize each column by type"""
        
        categories = {
            "metrics": [],
            "dimensions": [],
            "identifiers": [],
            "temporal": [],
            "geographic": [],
            "other": []
        }
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            col_lower = col.lower()
            
            # Identifiers (high uniqueness)
            if self._is_identifier(col):
                categories["identifiers"].append(col)
            
            # Binary flags / helper columns → "other" (before temporal/metric checks)
            elif self._is_binary_flag(col):
                categories["other"].append(col)
            
            # Temporal (dates) — must actually parse as dates
            elif pd.api.types.is_datetime64_any_dtype(dtype) or self._is_date_column(col):
                categories["temporal"].append(col)
            
            # Geographic
            elif self._is_geographic(col):
                categories["geographic"].append(col)
            
            # Metrics (numeric, not ID, not helper flags)
            elif pd.api.types.is_numeric_dtype(dtype) and self._is_metric_candidate(col):
                categories["metrics"].append(col)
            
            # Dimensions (categorical with reasonable cardinality)
            elif self._is_dimension(col):
                categories["dimensions"].append(col)
            
            else:
                categories["other"].append(col)
        
        return categories
    
    def _is_identifier(self, col: str) -> bool:
        """
        Check if column is an identifier / key.

        Delegates to the advanced heuristic used in RelationshipDetector so
        we have a single, consistent definition of "identifier" across the app.
        """

        return RelationshipDetector._is_identifier(self.df, col)
    
    def _is_binary_flag(self, col: str) -> bool:
        """Check if column is a binary flag / boolean indicator.
        
        Catches columns like Is_Promotion, Discount_Flag, Price_outlier, etc.
        that should NOT be treated as metrics or temporal columns.
        """
        col_lower = col.lower()
        
        # Name-based detection for flag/boolean patterns
        flag_patterns = re.compile(
            r"(^is_|^has_|_flag$|_indicator$|_outlier$|outlier|^flag_|_dummy$|_bool$|^bool_)",
            re.IGNORECASE,
        )
        if flag_patterns.search(col_lower):
            return True
        
        # Value-based detection: only {0, 1} or {True, False} or {Yes, No}
        series = self.df[col].dropna()
        if series.empty:
            return False
        
        unique_vals = set(series.unique().tolist())
        
        # Numeric binary: exactly {0, 1} (or subset)
        if pd.api.types.is_numeric_dtype(self.df[col]):
            numeric_vals = set(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
            if len(numeric_vals) <= 2 and numeric_vals.issubset({0, 1, 0.0, 1.0}):
                return True
        
        # String binary: True/False, Yes/No, Y/N
        if pd.api.types.is_object_dtype(self.df[col]):
            str_vals = {str(v).strip().lower() for v in unique_vals}
            binary_sets = [
                {'true', 'false'}, {'yes', 'no'}, {'y', 'n'},
                {'0', '1'}, {'t', 'f'},
            ]
            if len(str_vals) <= 2 and any(str_vals.issubset(bs) for bs in binary_sets):
                return True
        
        return False

    def _is_date_column(self, col: str) -> bool:
        """Check if column contains dates.
        
        Rules:
        - Never classify numeric dtype columns as dates (Price, Ram, etc.)
        - Require BOTH a date-like name AND successful date parsing, or
          very high date-parsing success rate (>90%) even without name match.
        """
        col_lower = col.lower()

        # Ignore obvious index-like columns
        if self.INDEX_LIKE_PATTERNS.search(col_lower):
            return False
        
        # NEVER classify numeric columns as dates
        if pd.api.types.is_numeric_dtype(self.df[col]):
            return False
        
        # Name patterns
        date_patterns = ['date', 'time', 'timestamp', 'created', 'updated',
                        'datetime', 'period']
        has_date_name = any(pattern in col_lower for pattern in date_patterns)
        
        # Try parsing sample values
        parses_as_date = False
        try:
            sample = self.df[col].dropna().head(200)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce")
                valid_ratio = parsed.notna().mean()
                if valid_ratio >= 0.7:
                    parses_as_date = True
        except Exception:
            pass

        # Require BOTH name + parsing, or very strong parsing alone
        if has_date_name and parses_as_date:
            return True
        
        # Even without name match, if >90% parse as dates it's a date column
        if parses_as_date:
            try:
                sample = self.df[col].dropna().head(200)
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.9:
                    return True
            except Exception:
                pass
        
        return False

    def _is_metric_candidate(self, col: str) -> bool:
        """Filter out helper/index/binary numeric columns from KPI metrics."""
        col_lower = col.lower()
        if self.INDEX_LIKE_PATTERNS.search(col_lower):
            return False
        if self.HELPER_METRIC_PATTERNS.search(col_lower):
            return False

        series = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if series.empty:
            return False

        # Exclude boolean-like numeric columns (0/1 style).
        unique_vals = set(series.unique().tolist())
        if len(unique_vals) <= 2 and unique_vals.issubset({0, 1}):
            return False
        return True
    
    def _is_geographic(self, col: str) -> bool:
        """Check if column is geographic"""
        col_lower = col.lower()
        
        geo_patterns = [
            'city', 'town', 'ville', 'ciudad',
            'region', 'state', 'province', 'estado',
            'country', 'pays', 'nation', 'pais',
            'zip', 'postal', 'address', 'location',
            'latitude', 'longitude', 'lat', 'lng', 'lon', 'geo'
        ]
        
        return any(pattern in col_lower for pattern in geo_patterns)
    
    def _is_dimension(self, col: str) -> bool:
        """Check if column is a good dimension for grouping"""
        
        if pd.api.types.is_numeric_dtype(self.df[col]):
            return False
        
        nunique = self.df[col].nunique()
        total = len(self.df)
        
        # Good cardinality for grouping: 2-50 unique values
        if 2 <= nunique <= 50:
            return True
        
        # Or very low cardinality ratio (<5%)
        if nunique / total < 0.05:
            return True
        
        return False
    
    def _get_noise_columns(self) -> List[str]:
        """Identify columns that should be excluded from relationship detection.
        
        These are columns that create noise:
        - Helper/derived columns (outlier flags, binary indicators)
        - Columns in the 'other' category that are binary flags
        """
        noise_cols = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            # Exclude helper metrics (outlier flags, indicators, etc.)
            if self.HELPER_METRIC_PATTERNS.search(col_lower):
                noise_cols.append(col)
            # Exclude binary flags that ended up in 'other'
            elif self._is_binary_flag(col):
                noise_cols.append(col)
        
        return list(set(noise_cols))

    def _detect_relationships(self) -> List[Dict]:
        """
        Detect relationships using CustomPPS
        Modern, no external dependencies!
        """
        
        logger.info("Detecting relationships with CustomPPS...")
        
        # Exclude identifier-like columns AND noise/helper columns from PPS analysis
        id_cols = self.column_types.get("identifiers", [])
        noise_cols = self._get_noise_columns()
        exclude_cols = list(set(id_cols + noise_cols))
        
        logger.info(f"Excluding from PPS: {exclude_cols}")
        df_for_pps = self.df.drop(columns=exclude_cols, errors="ignore")

        if df_for_pps.shape[1] < 2:
            logger.info("Not enough non-identifier columns for PPS analysis")
            return []

        try:
            pps = CustomPPS()
            pps_matrix = pps.matrix(df_for_pps)
            
            relationships = []
            
            # Extract significant relationships
            for _, row in pps_matrix[pps_matrix['ppscore'] > 0.3].iterrows():
                relationships.append({
                    "columns": [row['x'], row['y']],
                    "type": "predictive",
                    "strength": round(float(row['ppscore']), 3),
                    "direction": f"{row['x']} → {row['y']}",
                    "method": "custom_pps",
                    "insight": f"{row['x']} can predict {row['y']}",
                    "case": row['case']
                })
            
            logger.info(f"Found {len(relationships)} PPS relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"CustomPPS analysis failed: {e}")
            return []
    
    def _get_correlations(self) -> List[Dict]:
        """Get correlation matrix (fast, numeric only)"""
        
        numeric_df = self.df.select_dtypes(include=['number'])
        
        # Exclude noise columns from correlations too
        noise_cols = self._get_noise_columns()
        id_cols = self.column_types.get("identifiers", [])
        exclude_cols = list(set(noise_cols + id_cols))
        numeric_df = numeric_df.drop(columns=[c for c in exclude_cols if c in numeric_df.columns], errors="ignore")
        
        if len(numeric_df.columns) < 2:
            return []
        
        corr_matrix = numeric_df.corr()
        relationships = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.5:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    
                    relationships.append({
                        "columns": [col1, col2],
                        "type": "correlation",
                        "strength": round(abs(corr_value), 3),
                        "direction": "positive" if corr_value > 0 else "negative",
                        "method": "pearson",
                        "chart_suggestion": "scatter"
                    })
        
        logger.info(f"Found {len(relationships)} correlations")
        return relationships
    
    def _merge_relationships(
        self,
        pps_relationships: List[Dict],
        correlations: List[Dict]
    ) -> List[Dict]:
        """Merge and deduplicate relationships"""
        
        all_relationships = []
        seen_pairs = set()
        
        # Add PPS first (stronger method)
        for rel in pps_relationships:
            pair = tuple(sorted(rel["columns"]))
            if pair not in seen_pairs:
                all_relationships.append(rel)
                seen_pairs.add(pair)
        
        # Add correlations
        for rel in correlations:
            pair = tuple(sorted(rel["columns"]))
            if pair not in seen_pairs:
                all_relationships.append(rel)
                seen_pairs.add(pair)
        
        # Sort by strength
        all_relationships.sort(key=lambda x: x["strength"], reverse=True)
        
        return all_relationships
    
    def _select_dashboard_columns(self) -> List[str]:
        """Select best columns for dashboards"""
        
        selected = []
        
        # Add top metrics
        if self.column_types["metrics"]:
            # Sort by variance (more interesting metrics first)
            metrics_sorted = sorted(
                self.column_types["metrics"],
                key=lambda col: self.df[col].var() if self.df[col].var() > 0 else 0,
                reverse=True
            )
            selected.extend(metrics_sorted[:3])
        
        # Add key dimensions
        if self.column_types["dimensions"]:
            selected.extend(self.column_types["dimensions"][:3])
        
        # Add temporal if exists
        if self.column_types["temporal"]:
            selected.append(self.column_types["temporal"][0])
        
        # Add geographic if exists
        if self.column_types["geographic"]:
            selected.append(self.column_types["geographic"][0])
        
        # Remove duplicates
        unique_selected = []
        for col in selected:
            if col not in unique_selected:
                unique_selected.append(col)
        
        return unique_selected[:6]  # Max 6 columns
    
    def _identify_primary_metric(self) -> str:
        """Identify main metric to track"""
        
        if not self.column_types["metrics"]:
            return None
        
        # Priority metric names
        priority_names = [
            'revenue', 'sales', 'profit', 'income', 'amount',
            'total', 'value', 'price', 'cost', 'earnings'
        ]

        best_metric = None
        best_score = -1.0
        for metric in self.column_types["metrics"]:
            metric_lower = metric.lower()
            series = pd.to_numeric(self.df[metric], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                continue

            # Weighted score for business-friendly KPI selection.
            score = 0.0
            if any(name in metric_lower for name in priority_names):
                score += 3.0
            if self.HELPER_METRIC_PATTERNS.search(metric_lower):
                score -= 3.0
            valid_ratio = non_null.shape[0] / max(len(series), 1)
            score += valid_ratio
            score += min(float(non_null.nunique()) / max(len(non_null), 1), 1.0)
            score += min(float(non_null.var() if non_null.var() > 0 else 0.0), 1.0)

            if score > best_score:
                best_score = score
                best_metric = metric

        return best_metric or self.column_types["metrics"][0]
    
    def _calculate_confidence(self, relationships: List[Dict]) -> float:
        """Calculate overall confidence score"""
        
        if not relationships:
            return 0.5
        
        # Average strength of top 5 relationships
        top_strengths = [r["strength"] for r in relationships[:5]]
        avg_strength = np.mean(top_strengths) if top_strengths else 0.5
        
        # Data completeness
        completeness = 1 - (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)))
        
        # Combine
        confidence = (avg_strength * 0.7) + (completeness * 0.3)
        
        return round(confidence, 2)