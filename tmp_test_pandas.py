import pandas as pd
import numpy as np

df = pd.DataFrame({
    "_t": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01"]),
    "_m": [10, 20, 30]
})

try:
    monthly = (
        df.groupby(df["_t"].dt.to_period("M"), as_index=False)["_m"]
        .sum()
    )
    print("Columns:", monthly.columns.tolist())
    print(monthly.sort_values("_t"))
except Exception as e:
    print("Error:", e)
