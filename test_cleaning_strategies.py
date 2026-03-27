import pandas as pd
import numpy as np
from app.services.data.cleaning_service import CleaningService

data = {
    'Age': [25, np.nan, 30, 40, 100],        # Numeric
    'Salary': [50000, 60000, np.nan, 80000, 50000],  # Numeric
    'City': ['Paris', 'Lyon', np.nan, 'Paris', 'Marseille'], # Categorical
}
df = pd.DataFrame(data)

def test_strategy(name, strategy):
    print(f"\n{'='*40}")
    print(f"--- TESTING STRATEGY: {name} ---")
    print(f"{'='*40}")
    profile = {
        "handle_missing": "fill",
        "missing_fill_strategy": strategy,
        "missing_fill_value": "999" if strategy == "constant" else None,
        "fix_data_types": True,
        "strip_whitespace": True,
        "detect_outliers": False 
    }
    
    clean_df, report = CleaningService.clean_dataframe(df.copy(), profile)
    print("Cleaned Data:")
    print(clean_df)

print("Original Data (with missing values):")
print(df)

strategies = ["mean", "median", "mode", "constant"]
for s in strategies:
    test_strategy(s.upper(), s)
