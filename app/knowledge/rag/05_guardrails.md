---
tags: identifier email id uuid junk downrank ignore
---

# Interpretation guardrails

- **Downrank or ignore** relationships where one column is an **identifier**, **email**, **phone**, **uuid**, or near-unique key predicting personal fields — often trivial, not actionable KPIs.
- Prefer relationships involving **metrics**, **dimensions**, and **temporal** columns from the analysis categories.
- **KPI** recommendations should use **primary_metric** when present and real business dimensions.
