---
tags: pandas time series line_chart trend temporal resample datetime kpi
---

# Metric over time

Use when a **temporal** column exists and a **metric** should be tracked over time.

**Pandas**
```python
df["period"] = pd.to_datetime(df["time_col"], errors="coerce")
df.dropna(subset=["period"]).groupby(df["period"].dt.to_period("M"))["metric_col"].sum()
```

**Chart**: **line** chart (x = time, y = aggregated metric). Mention granularity (monthly if many points).
