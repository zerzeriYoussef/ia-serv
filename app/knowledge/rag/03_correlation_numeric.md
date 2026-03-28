---
tags: correlation pearson scatter numeric predictive strength
---

# Two numeric columns (correlation / predictive)

When `relationship.type` is `correlation` or two columns are numeric metrics.

**Pandas**
```python
df[["col_a", "col_b"]].corr()
# or scatter-ready:
df.plot.scatter(x="col_a", y="col_b")
```

**Chart**: **scatter** plot. Add trend line only if linear relationship is plausible.
