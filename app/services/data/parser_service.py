import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import csv

logger = logging.getLogger(__name__)


class ParserService:
    
    @staticmethod
    async def parse_file(file_path: str, file_type: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Parse file and extract metadata
        
        Returns:
            Tuple of (dataframe, metadata)
        """
        try:
            # Parse based on file type
            if file_type == 'csv':
                df = ParserService._read_csv_robust(file_path)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Extract metadata
            metadata = ParserService._extract_metadata(df)
            
            logger.info(f"Parsed {file_type} file: {len(df)} rows, {len(df.columns)} columns")
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error parsing file: {e}")
            raise

    @staticmethod
    def _read_csv_robust(file_path: str) -> pd.DataFrame:
        """
        Read CSV files defensively.

        Handles common real-world issues:
        - Inconsistent number of fields on some rows
        - Unknown delimiter
        - Occasional malformed lines
        """
        # 1) Fast path
        try:
            return pd.read_csv(file_path)
        except pd.errors.ParserError as e:
            logger.warning(f"CSV parser error (fast path): {e}")

        # 2) Try delimiter sniffing + python engine (more tolerant)
        delimiter = None
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                sample = f.read(8192)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
            except Exception:
                delimiter = None
        except Exception as e:
            logger.warning(f"Failed to sniff CSV delimiter: {e}")

        try:
            return pd.read_csv(
                file_path,
                sep=delimiter,  # None means default ','
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as e:
            logger.warning(f"CSV parser error (python engine): {e}")

        # 3) Last resort: try python engine with sep autodetection
        return pd.read_csv(
            file_path,
            sep=None,
            engine="python",
            on_bad_lines="skip",
        )
    
    @staticmethod
    def _extract_metadata(df: pd.DataFrame) -> Dict:
        """Extract metadata from dataframe"""
        
        # Basic info
        metadata = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
        }
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Use to_json -> loads to ensure NaN->null and numpy types->native python types
            stats_json = df[numeric_cols].describe().to_json(orient="columns")
            metadata["summary_stats"] = json.loads(stats_json)
        else:
            metadata["summary_stats"] = {}
        
        # Detect column categories
        metadata["column_categories"] = ParserService._categorize_columns(df)
        
        # Missing values
        missing = df.isnull().sum()
        metadata["missing_values"] = {
            col: int(count) for col, count in missing.items() if count > 0
        }
        
        # Memory usage
        metadata["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        
        return metadata
    
    @staticmethod
    def _categorize_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize columns by data type"""
        categories = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
            "boolean": []
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                categories["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                categories["datetime"].append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                categories["boolean"].append(col)
            elif pd.api.types.is_categorical_dtype(dtype) or df[col].nunique() < len(df) * 0.5:
                # If unique values < 50% of rows, consider categorical
                categories["categorical"].append(col)
            else:
                categories["text"].append(col)
        
        return categories
    
    @staticmethod
    def get_sample_data(df: pd.DataFrame, n: int = 5) -> Dict:
        """Get sample rows as JSON"""
        return df.head(n).to_dict(orient='records')