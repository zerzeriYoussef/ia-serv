"""
Custom Predictive Power Score Implementation
Modern replacement for outdated PPScore library

Compatible with:
- pandas >= 2.2.2
- scikit-learn >= 1.5.0
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CustomPPS:
    """
    Predictive Power Score calculator using Decision Trees
    
    Measures how well one column predicts another:
    - Score 0.0 = No predictive power
    - Score 1.0 = Perfect prediction
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def matrix(self, df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
        """
        Calculate PPS matrix for all column pairs
        d
        Args:
            df: Input DataFrame
            sample_size: Max rows to use (for performance)
        
        Returns:
            DataFrame with columns: x, y, ppscore, baseline_score, model_score, case
        """
        
        # Sample if dataset is large
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=self.random_state)
            logger.info(f"Sampled {sample_size} rows from {len(df)} for PPS analysis")
        else:
            df_sample = df.copy()
        
        results = []
        columns = df_sample.columns.tolist()
        
        # Calculate PPS for each pair
        for x_col in columns:
            for y_col in columns:
                if x_col == y_col:
                    continue
                
                try:
                    score_info = self._calculate_pps(df_sample, x_col, y_col)
                    results.append({
                        'x': x_col,
                        'y': y_col,
                        'ppscore': score_info['ppscore'],
                        'baseline_score': score_info['baseline'],
                        'model_score': score_info['model_score'],
                        'case': score_info['case']
                    })
                except Exception as e:
                    logger.warning(f"PPS calculation failed for {x_col} → {y_col}: {e}")
                    results.append({
                        'x': x_col,
                        'y': y_col,
                        'ppscore': 0.0,
                        'baseline_score': 0.0,
                        'model_score': 0.0,
                        'case': 'error'
                    })
        
        return pd.DataFrame(results)
    
    def score(self, df: pd.DataFrame, x_col: str, y_col: str) -> float:
        """Calculate single PPS score"""
        result = self._calculate_pps(df, x_col, y_col)
        return result['ppscore']
    
    def _calculate_pps(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """Calculate PPS for x → y"""
        
        # Clean data
        df_clean = df[[x_col, y_col]].dropna()
        
        if len(df_clean) < 10:
            return {
                'ppscore': 0.0,
                'baseline': 0.0,
                'model_score': 0.0,
                'case': 'insufficient_data'
            }
        
        X = df_clean[[x_col]]
        y = df_clean[y_col]
        
        # Determine case
        x_is_numeric = pd.api.types.is_numeric_dtype(X[x_col])
        y_is_numeric = pd.api.types.is_numeric_dtype(y)
        
        if x_is_numeric and y_is_numeric:
            return self._numeric_to_numeric(X, y)
        elif not x_is_numeric and y_is_numeric:
            return self._categorical_to_numeric(X, y, x_col)
        elif x_is_numeric and not y_is_numeric:
            return self._numeric_to_categorical(X, y)
        else:
            return self._categorical_to_categorical(X, y, x_col)
    
    def _numeric_to_numeric(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Numeric → Numeric using Decision Tree Regressor"""
        
        baseline_score = 0.0  # Predicting mean = R² of 0
        
        model = DecisionTreeRegressor(
            max_depth=4,
            random_state=self.random_state,
            min_samples_leaf=max(1, len(X) // 50)
        )
        
        try:
            cv_scores = cross_val_score(
                model, X, y,
                cv=min(5, len(X) // 10),
                scoring='r2'
            )
            model_score = max(0, cv_scores.mean())
        except:
            model_score = 0.0
        
        ppscore = self._normalize_score(model_score, baseline_score)
        
        return {
            'ppscore': round(ppscore, 3),
            'baseline': baseline_score,
            'model_score': round(model_score, 3),
            'case': 'numeric_to_numeric'
        }
    
    def _categorical_to_numeric(self, X: pd.DataFrame, y: pd.Series, x_col: str) -> Dict:
        """Categorical → Numeric"""
        
        le = LabelEncoder()
        X_encoded = pd.DataFrame({
            x_col: le.fit_transform(X[x_col].astype(str))
        })
        
        return self._numeric_to_numeric(X_encoded, y)
    
    def _numeric_to_categorical(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Numeric → Categorical using Decision Tree Classifier"""
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        
        baseline_score = (y_encoded == np.bincount(y_encoded).argmax()).mean()
        
        model = DecisionTreeClassifier(
            max_depth=4,
            random_state=self.random_state,
            min_samples_leaf=max(1, len(X) // 50)
        )
        
        try:
            cv_scores = cross_val_score(
                model, X, y_encoded,
                cv=min(5, len(X) // 10),
                scoring='accuracy'
            )
            model_score = cv_scores.mean()
        except:
            model_score = baseline_score
        
        ppscore = self._normalize_score(model_score, baseline_score)
        
        return {
            'ppscore': round(ppscore, 3),
            'baseline': round(baseline_score, 3),
            'model_score': round(model_score, 3),
            'case': 'numeric_to_categorical'
        }
    
    def _categorical_to_categorical(self, X: pd.DataFrame, y: pd.Series, x_col: str) -> Dict:
        """Categorical → Categorical"""
        
        le_x = LabelEncoder()
        X_encoded = pd.DataFrame({
            x_col: le_x.fit_transform(X[x_col].astype(str))
        })
        
        return self._numeric_to_categorical(X_encoded, y)
    
    def _normalize_score(self, model_score: float, baseline_score: float) -> float:
        """Normalize to 0-1 range"""
        
        if baseline_score >= 1.0 or model_score <= baseline_score:
            return 0.0
        
        normalized = (model_score - baseline_score) / (1.0 - baseline_score)
        return max(0.0, min(1.0, normalized))