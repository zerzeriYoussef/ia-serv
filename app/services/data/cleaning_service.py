import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class CleaningService:
    """Service for cleaning and preprocessing data"""
    
    @staticmethod
    def clean_dataframe(
        df: pd.DataFrame,
        profile: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean dataframe based on cleaning profile
        
        Args:
            df: Input dataframe
            profile: Cleaning configuration
            
        Returns:
            Tuple of (cleaned_df, cleaning_report)
        """
        report = {
            "rows_before": len(df),
            "operations": [],
            "changes": {}
        }
        
        df_clean = df.copy()
        
        # 1. Text cleaning (should be done early to strip whitespace before type conversion)
        if profile.get("strip_whitespace", False):
            df_clean = CleaningService._clean_text(
                df_clean,
                standardize=profile.get("standardize_text", False)
            )
            report["operations"].append("text_cleaning")
            
        # 2. Fix data types (must be done before missing values to know true types)
        if profile.get("fix_data_types", False):
            df_clean, type_report = CleaningService._fix_data_types(df_clean)
            report["operations"].append("data_types")
            report["changes"]["data_types"] = type_report
            
        # 3. Handle missing values
        if profile.get("handle_missing"):
            df_clean, missing_report = CleaningService._handle_missing(
                df_clean,
                strategy=profile.get("handle_missing", "drop"),
                fill_strategy=profile.get("missing_fill_strategy", "mean"),
                fill_value=profile.get("missing_fill_value")
            )
            report["operations"].append("missing_values")
            report["changes"]["missing_values"] = missing_report
        
        # 4. Remove duplicates
        if profile.get("remove_duplicates", False):
            df_clean, dup_report = CleaningService._remove_duplicates(
                df_clean,
                subset=profile.get("duplicate_subset")
            )
            report["operations"].append("duplicates")
            report["changes"]["duplicates"] = dup_report
        
        # 5. Detect/handle outliers
        if profile.get("detect_outliers", False):
            df_clean, outlier_report = CleaningService._handle_outliers(
                df_clean,
                method=profile.get("outlier_method", "iqr"),
                threshold=profile.get("outlier_threshold", 1.5),
                action=profile.get("outlier_action", "flag")
            )
            report["operations"].append("outliers")
            report["changes"]["outliers"] = outlier_report
        
        # 6. Date standardization
        if profile.get("standardize_dates", False):
            df_clean, date_report = CleaningService._standardize_dates(
                df_clean,
                target_format=profile.get("date_format")
            )
            report["operations"].append("date_standardization")
            report["changes"]["dates"] = date_report
        
        report["rows_after"] = len(df_clean)
        report["rows_removed"] = report["rows_before"] - report["rows_after"]
        
        return df_clean, report
    
    @staticmethod
    def _handle_missing(
        df: pd.DataFrame,
        strategy: str = "drop",
        fill_strategy: str = "mean",
        fill_value: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values"""
        
        missing_before = df.isnull().sum().to_dict()
        total_missing = df.isnull().sum().sum()
        
        if strategy == "drop":
            df_clean = df.dropna()
            
        elif strategy == "fill":
            df_clean = df.copy()
            
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        # Numeric columns
                        if fill_strategy == "mean":
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                        elif fill_strategy == "median":
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        elif fill_strategy == "mode":
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                        elif fill_strategy == "constant":
                            df_clean[col] = df_clean[col].fillna(float(fill_value) if fill_value else 0)
                    
                    elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                        # Datetime columns
                        if fill_strategy == "mode":
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                        elif fill_strategy == "constant" and fill_value:
                            try:
                                df_clean[col] = df_clean[col].fillna(pd.to_datetime(fill_value))
                            except (ValueError, TypeError):
                                pass  # leave na if invalid
                        else:
                            # Cannot reliably default-fill datetimes without domain context
                            pass
                            
                    else:
                        # Non-numeric string/categorical columns
                        if fill_strategy == "mode":
                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                        elif fill_strategy == "constant":
                            df_clean[col] = df_clean[col].fillna(fill_value if fill_value else "Unknown")
                        else:
                            df_clean[col] = df_clean[col].fillna("Unknown")
        
        elif strategy == "interpolate":
            df_clean = df.copy()
            # Only interpolate numeric columns
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate()
            # Fill remaining with mode or constant
            df_clean = df_clean.ffill().bfill()
        
        else:
            df_clean = df.copy()
        
        missing_after = df_clean.isnull().sum().to_dict()
        
        report = {
            "strategy": strategy,
            "fill_strategy": fill_strategy,
            "missing_before": {k: int(v) for k, v in missing_before.items() if v > 0},
            "missing_after": {k: int(v) for k, v in missing_after.items() if v > 0},
            "total_filled": total_missing - df_clean.isnull().sum().sum()
        }
        
        logger.info(f"Handled {report['total_filled']} missing values using {strategy}")
        
        return df_clean, report
    
    @staticmethod
    def _remove_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate rows"""
        
        duplicates_before = df.duplicated(subset=subset).sum()
        df_clean = df.drop_duplicates(subset=subset, keep='first')
        duplicates_removed = duplicates_before
        
        report = {
            "duplicates_found": int(duplicates_before),
            "duplicates_removed": int(duplicates_removed),
            "subset": subset
        }
        
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df_clean, report
    
    @staticmethod
    def _fix_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Auto-detect and fix data types by coercing gracefully"""
        
        df_clean = df.copy()
        type_changes = {}
        
        for col in df_clean.columns:
            original_type = str(df_clean[col].dtype)
            
            if df_clean[col].dtype == 'object':
                non_na = df_clean[col].dropna()
                if len(non_na) == 0:
                    continue
                    
                # 1. Try to convert to numeric gracefully
                num_coerced = pd.to_numeric(non_na, errors='coerce')
                num_success_rate = num_coerced.notna().mean()
                
                # If more than 30% of non-null values are numbers, treat as numeric column
                if num_success_rate > 0.3:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    type_changes[col] = {
                        "from": original_type,
                        "to": str(df_clean[col].dtype)
                    }
                    continue
                
                # 2. Try to convert to datetime gracefully
                date_coerced = pd.to_datetime(non_na, errors='coerce')
                date_success_rate = date_coerced.notna().mean()
                
                # If more than 30% of non-null values are valid dates, treat as datetime column
                if date_success_rate > 0.3:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    type_changes[col] = {
                        "from": original_type,
                        "to": "datetime64[ns]" # standardized date representation
                    }
                    continue
        
        report = {
            "types_changed": len(type_changes),
            "changes": type_changes
        }
        
        logger.info(f"Fixed {len(type_changes)} column data types")
        
        return df_clean, report
    
    @staticmethod
    def _handle_outliers(
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
        action: str = "flag"
    ) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers"""
        
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        
        outliers_detected = {}
        total_outliers = 0
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outlier_mask = pd.Series(False, index=df_clean.index)
                outlier_mask.loc[df_clean[col].notna()] = z_scores > threshold
            
            else:
                # Default to IQR
                outlier_mask = pd.Series(False, index=df_clean.index)
            
            num_outliers = outlier_mask.sum()
            
            if num_outliers > 0:
                outliers_detected[col] = int(num_outliers)
                total_outliers += num_outliers
                
                if action == "remove":
                    df_clean = df_clean[~outlier_mask]
                
                elif action == "cap":
                    if method == "iqr":
                        df_clean.loc[outlier_mask & (df_clean[col] < lower_bound), col] = lower_bound
                        df_clean.loc[outlier_mask & (df_clean[col] > upper_bound), col] = upper_bound
                    elif method == "zscore":
                        mean_val = df_clean[col].mean()
                        std_val = df_clean[col].std()
                        lower_bound_z = mean_val - threshold * std_val
                        upper_bound_z = mean_val + threshold * std_val
                        df_clean.loc[outlier_mask & (df_clean[col] < lower_bound_z), col] = lower_bound_z
                        df_clean.loc[outlier_mask & (df_clean[col] > upper_bound_z), col] = upper_bound_z
                
                elif action == "flag":
                    # Add a flag column
                    df_clean[f'{col}_outlier'] = outlier_mask
        
        report = {
            "method": method,
            "threshold": threshold,
            "action": action,
            "outliers_by_column": outliers_detected,
            "total_outliers": total_outliers
        }
        
        logger.info(f"Detected {total_outliers} outliers using {method} method")
        
        return df_clean, report
    
    @staticmethod
    def _clean_text(
        df: pd.DataFrame,
        standardize: bool = False
    ) -> pd.DataFrame:
        """Clean text columns"""
        
        df_clean = df.copy()
        text_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            # Strip whitespace
            df_clean[col] = df_clean[col].str.strip()
            
            if standardize:
                # Lowercase
                df_clean[col] = df_clean[col].str.lower()
                # Remove extra spaces
                df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        return df_clean
    
    @staticmethod
    def _standardize_dates(
        df: pd.DataFrame,
        target_format: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Standardize date columns"""
        
        df_clean = df.copy()
        date_cols = df_clean.select_dtypes(include=['datetime64']).columns
        
        changes = {}
        
        for col in date_cols:
            if target_format:
                df_clean[col] = df_clean[col].dt.strftime(target_format)
                changes[col] = f"Formatted to {target_format}"
        
        report = {
            "date_columns": list(date_cols),
            "format": target_format,
            "changes": changes
        }
        
        return df_clean, report
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            
            # Missing values
            "missing_values": {
                "total": int(df.isnull().sum().sum()),
                "by_column": {col: int(count) for col, count in df.isnull().sum().items() if count > 0},
                "percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
            },
            
            # Duplicates
            "duplicates": {
                "count": int(df.duplicated().sum()),
                "percentage": round(df.duplicated().sum() / len(df) * 100, 2)
            },
            
            # Data types
            "data_types": df.dtypes.astype(str).to_dict(),
            
            # Numeric columns statistics
            "numeric_summary": {},
            
            # Categorical columns
            "categorical_summary": {}
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            report["numeric_summary"][col] = {
                "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                "median": float(df[col].median()) if not df[col].isnull().all() else None,
                "std": float(df[col].std()) if not df[col].isnull().all() else None,
                "min": float(df[col].min()) if not df[col].isnull().all() else None,
                "max": float(df[col].max()) if not df[col].isnull().all() else None,
                "missing": int(df[col].isnull().sum())
            }
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            report["categorical_summary"][col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                "missing": int(df[col].isnull().sum())
            }
        
        return report