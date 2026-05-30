"""
Additional relationship detection patterns
Complements CustomPPS with specific relationship types
"""



import pandas as pd

from typing import List, Dict

import logging



logger = logging.getLogger(__name__)





class RelationshipDetector:

    """
    Detect specific relationship patterns:
    - Hierarchies (city → region → country)
    - Primary/Foreign keys
    - Potential joins
    """



    @staticmethod

    def _is_identifier(df: pd.DataFrame, col: str) -> bool:

        """
        Advanced identifier detection with multiple heuristics.

        Returns True if column is likely an identifier/key that should be
        excluded from semantic analysis (hierarchies, aggregations).
        """



        col_lower = col.lower()





        id_patterns = [



            "id",

            "_id",

            "key",

            "_key",

            "code",

            "_code",

            "uuid",

            "guid",



            "number",

            "num",

            "no",

            "seq",

            "index",

            "idx",

            "record",



            "ref",

            "reference",

            "token",

            "hash",

            "session",



            "sku",

            "barcode",

            "upc",

            "isbn",

            "ean",



            "email",

            "phone",

            "mobile",

            "username",

            "login",



            "transaction",

            "order_number",

            "invoice",

            "receipt",

        ]



        has_pattern = any(pattern in col_lower for pattern in id_patterns)







        try:

            uniqueness = df[col].nunique(dropna=True) / max(len(df), 1)

        except Exception:

            uniqueness = 0





        if has_pattern and uniqueness > 0.9:

            return True





        if pd.api.types.is_integer_dtype(df[col]):



            try:

                values = df[col].dropna().sort_values().values

                if len(values) > 10:



                    import numpy as np



                    sample = values[:100]

                    diffs = np.diff(sample)





                    if (diffs == 1).sum() / len(diffs) > 0.9:

                        return True

            except Exception:

                pass





        if uniqueness > 0.8:

            try:

                sample_values = df[col].dropna().astype(str).head(50)





                uuid_pattern = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"

                uuid_match = sample_values.str.match(uuid_pattern, case=False).sum()

                if uuid_match / len(sample_values) > 0.8:

                    return True





                email_pattern = (

                    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

                )

                email_match = sample_values.str.match(email_pattern).sum()

                if email_match / len(sample_values) > 0.8:

                    return True





                hash_pattern = r"^[a-f0-9]{32,}$"

                hash_match = sample_values.str.match(hash_pattern, case=False).sum()

                if hash_match / len(sample_values) > 0.8:

                    return True



            except Exception:

                pass





        try:

            if df.columns.tolist().index(col) == 0:

                if uniqueness > 0.95:



                    return True

        except ValueError:



            pass



        return False



    @staticmethod

    def detect_hierarchies(df: pd.DataFrame) -> List[Dict]:

        """
        Detect hierarchical relationships

        Example: Each city belongs to exactly one region
        """



        categorical_cols = df.select_dtypes(include=["object", "category"]).columns



        categorical_cols = [

            col

            for col in categorical_cols

            if not RelationshipDetector._is_identifier(df, col)

        ]

        hierarchies = []



        for i, col1 in enumerate(categorical_cols):

            for col2 in categorical_cols[i+1:]:



                mapping = df.groupby(col1)[col2].nunique()



                if (mapping == 1).all() and len(mapping) > 1:

                    hierarchies.append({

                        "columns": [col1, col2],

                        "type": "hierarchy",

                        "strength": 1.0,

                        "direction": f"{col1} → {col2}",

                        "insight": f"Each {col1} belongs to exactly one {col2}",

                        "chart_suggestion": "treemap",

                        "method": "hierarchy_detector"

                    })



        return hierarchies



    @staticmethod

    def detect_keys(df: pd.DataFrame) -> List[Dict]:

        """Detect primary key candidates"""



        keys = []



        for col in df.columns:

            uniqueness = df[col].nunique() / len(df)

            has_nulls = df[col].isnull().any()





            if uniqueness == 1.0 and not has_nulls:

                keys.append({

                    "column": col,

                    "type": "primary_key",

                    "confidence": 1.0,

                    "insight": f"{col} is unique and complete - likely primary key"

                })





            elif 0.1 < uniqueness < 0.9:

                keys.append({

                    "column": col,

                    "type": "foreign_key",

                    "confidence": 0.7,

                    "insight": f"{col} might reference another table"

                })



        return keys



    @staticmethod

    def suggest_aggregations(df: pd.DataFrame) -> List[Dict]:

        """
        Suggest useful aggregations based on column types

        Example: "Sum sales by region"
        """



        suggestions = []



        metrics = df.select_dtypes(include=["number"]).columns.tolist()

        dimensions = df.select_dtypes(include=["object", "category"]).columns.tolist()





        dimensions = [

            col

            for col in dimensions

            if not RelationshipDetector._is_identifier(df, col)

        ]





        dimensions = [col for col in dimensions if 2 <= df[col].nunique() <= 50]



        for metric in metrics:

            for dimension in dimensions:

                suggestions.append({

                    "aggregation": f"Sum {metric} by {dimension}",

                    "metric": metric,

                    "dimension": dimension,

                    "type": "group_sum",

                    "chart_suggestion": "bar"

                })





        return suggestions[:10]
