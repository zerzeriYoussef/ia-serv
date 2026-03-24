import pandas as pd
import numpy as np

df = pd.DataFrame({
    "_t": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01"]),
    "_m": [10, 20, 30]
})

try:
    # Option 1: groupby with index=True then reset
    monthly = (
        df.groupby(df["_t"].dt.to_period("M"))["_m"]
        .sum()
        .reset_index()
    )
    print("Option 1 Columns:", monthly.columns.tolist())
    print(monthly.sort_values("_t"))

    # Option 2: rename the grouping object
    monthly2 = (
        df.groupby(df["_t"].dt.to_period("M").rename("_t"), as_index=False)["_m"]
        .sum()
    )
    print("Option 2 Columns:", monthly2.columns.tolist())
    print(monthly2.sort_values("_t"))
except Exception as e:
    print("Error:", e)
