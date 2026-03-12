"""
Column Analysis Service
Analyzes dataset columns and detects relationships using CustomPPS
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from app.services.analysis.custom_pps import CustomPPS
from app.services.analysis.relationship_detector import RelationshipDetector

logger = logging.getLogger(__name__)


class ColumnAnalyzer:
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
            
            # Temporal (dates)
            elif pd.api.types.is_datetime64_any_dtype(dtype) or self._is_date_column(col):
                categories["temporal"].append(col)
            
            # Geographic
            elif self._is_geographic(col):
                categories["geographic"].append(col)
            
            # Metrics (numeric, not ID)
            elif pd.api.types.is_numeric_dtype(dtype):
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
    
    def _is_date_column(self, col: str) -> bool:
        """Check if column contains dates"""
        col_lower = col.lower()
        
        # Name patterns
        date_patterns = ['date', 'time', 'timestamp', 'created', 'updated', 
                        'day', 'month', 'year', 'dt']
        if any(pattern in col_lower for pattern in date_patterns):
            return True
        
        # Try parsing sample
        try:
            sample = self.df[col].dropna().head(100)
            if len(sample) > 0:
                pd.to_datetime(sample)
                return True
        except:
            pass
        
        return False
    
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
    
    def _detect_relationships(self) -> List[Dict]:
        """
        Detect relationships using CustomPPS
        Modern, no external dependencies!
        """
        
        logger.info("Detecting relationships with CustomPPS...")

        # Exclude identifier-like columns (e.g. customer_id) from PPS analysis
        id_cols = self.column_types.get("identifiers", [])
        df_for_pps = self.df.drop(columns=id_cols, errors="ignore")

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
        
        for metric in self.column_types["metrics"]:
            metric_lower = metric.lower()
            if any(name in metric_lower for name in priority_names):
                return metric
        
        # Default: first metric
        return self.column_types["metrics"][0]
    
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