---
tags: pandas groupby bar_chart metric dimension kpi aggregation
---

# Metric by dimension (KPI slice)

Use when you have one numeric **metric** and one categorical **dimension**.

**Pandas**
```python
df.groupby("dimension_col")["metric_col"].sum().sort_values(ascending=False).head(10)
```

**Chart**: horizontal or vertical **bar** chart; sort bars by value. Avoid pie charts when there are many categories.
