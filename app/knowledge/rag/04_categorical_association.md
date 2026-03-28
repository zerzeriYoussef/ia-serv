---
tags: chi_square categorical heatmap crosstab association
---

# Categorical association

When two categorical dimensions co-vary.

**Pandas**
```python
pd.crosstab(df["cat_a"], df["cat_b"], normalize="index")
```

**Chart**: **heatmap** of counts or row-normalized proportions; or stacked bar for few categories.
