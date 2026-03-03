import pandas as pd
from typing import Dict, List
import re
import logging

logger = logging.getLogger(__name__)


class ValidationService:
    """Service for data validation"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, rules: Dict = None) -> Dict:
        """
        Validate dataframe against rules
        
        Returns validation report with pass/fail status
        """
        
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks_performed": []
        }
        
        # Default validation checks
        
        # 1. Check for completely empty dataframe
        if df.empty:
            report["valid"] = False
            report["errors"].append("Dataframe is empty")
            return report
        
        # 2. Check for columns with all missing values
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            report["warnings"].append(f"Columns with all missing values: {empty_cols}")
            report["checks_performed"].append("empty_columns")
        
        # 3. Check for excessive missing values (>50%)
        high_missing_cols = []
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                high_missing_cols.append(f"{col} ({missing_pct:.1f}%)")
        
        if high_missing_cols:
            report["warnings"].append(f"Columns with >50% missing: {high_missing_cols}")
            report["checks_performed"].append("high_missing_values")
        
        # 4. Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            report["errors"].append(f"Duplicate column names: {duplicate_cols}")
            report["valid"] = False
            report["checks_performed"].append("duplicate_columns")
        
        # 5. Check for single-value columns (no variance)
        single_value_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                single_value_cols.append(col)
        
        if single_value_cols:
            report["warnings"].append(f"Columns with single value: {single_value_cols}")
            report["checks_performed"].append("single_value_columns")
        
        # Custom rules validation
        if rules:
            custom_report = ValidationService._validate_custom_rules(df, rules)
            report["errors"].extend(custom_report.get("errors", []))
            report["warnings"].extend(custom_report.get("warnings", []))
            if not custom_report.get("valid", True):
                report["valid"] = False
        
        report["total_checks"] = len(report["checks_performed"])
        
        return report
    
    @staticmethod
    def _validate_custom_rules(df: pd.DataFrame, rules: Dict) -> Dict:
        """Validate against custom rules"""
        
        report = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required columns
        if "required_columns" in rules:
            missing_cols = set(rules["required_columns"]) - set(df.columns)
            if missing_cols:
                report["errors"].append(f"Missing required columns: {list(missing_cols)}")
                report["valid"] = False
        
        # Minimum rows
        if "min_rows" in rules:
            if len(df) < rules["min_rows"]:
                report["errors"].append(f"Dataframe has {len(df)} rows, minimum is {rules['min_rows']}")
                report["valid"] = False
        
        # Column type validation
        if "column_types" in rules:
            for col, expected_type in rules["column_types"].items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if expected_type not in actual_type:
                        report["warnings"].append(f"Column {col}: expected {expected_type}, got {actual_type}")
        
        # Value range validation
        if "value_ranges" in rules:
            for col, (min_val, max_val) in rules["value_ranges"].items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    if out_of_range > 0:
                        report["warnings"].append(f"Column {col}: {out_of_range} values outside range [{min_val}, {max_val}]")
        
        return report
    
    @staticmethod
    def validate_email(series: pd.Series) -> Dict:
        """Validate email addresses"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        valid_count = series.str.match(email_pattern).sum()
        invalid_count = len(series) - valid_count
        
        return {
            "valid": valid_count,
            "invalid": invalid_count,
            "validity_rate": round((valid_count / len(series)) * 100, 2)
        }
    
    @staticmethod
    def validate_phone(series: pd.Series) -> Dict:
        """Validate phone numbers"""
        # Simple pattern - adjust based on your needs
        phone_pattern = r'^\+?1?\d{9,15}$'
        
        # Clean phone numbers
        cleaned = series.str.replace(r'[^0-9+]', '', regex=True)
        
        valid_count = cleaned.str.match(phone_pattern).sum()
        invalid_count = len(series) - valid_count
        
        return {
            "valid": valid_count,
            "invalid": invalid_count,
            "validity_rate": round((valid_count / len(series)) * 100, 2)
        }