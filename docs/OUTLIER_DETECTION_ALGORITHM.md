# Outlier Detection Algorithm

This document explains the data quality scan used by `ScanService` and
`CleaningService` for numeric outlier detection.

## Goal

The detector should work on many dataset types without assuming that every
statistical outlier is dirty data. It separates:

- invalid values, such as negative prices or percentages above 100
- unusual values, such as large orders or expensive products
- identifier columns, such as invoice numbers, stock codes, IDs, and SKUs

The default method is `auto`.

## Pipeline

1. Normalize missing tokens and data types before outlier detection.
2. Select numeric columns.
3. Skip identifier-like numeric columns.
4. Infer a semantic role for each numeric column.
5. Choose an outlier detector based on the role and distribution.
6. Apply domain rules for impossible values.
7. Return metadata with severity, role, confidence, and suggested action.

## Column Role Detection

The role detector uses two signals:

- column name hints
- value distribution

Column names are normalized before matching. Accents are removed, punctuation is
converted to underscores, and text is lowercased. This means names like
`Quantité`, `quantite`, and `Prix_Unitaire` can be matched consistently.

Supported name hints include:

- quantity/count: `quantity`, `qty`, `count`, `qte`, `quantite`, `nombre`, `nb`
- money/price: `price`, `cost`, `amount`, `prix`, `cout`, `montant`, `tarif`
- percentage/rate: `percent`, `percentage`, `pct`, `pourcentage`, `taux`, `rate`

If the column name is meaningless, such as `asnfqosinqn`, the detector falls
back to value-based inference.

## Value-Based Inference

For each numeric column, the algorithm computes a value profile:

- `integer_rate`: share of values that are integer-like
- `non_negative_rate`: share of values greater than or equal to zero
- `positive_rate`: share of values greater than zero
- `unique_ratio`: unique values divided by row count
- `skew`: distribution skewness
- `percentage_0_100_rate`: share of values between 0 and 100
- `count_like_score`: weighted score for repeated, mostly non-negative integers

Example:

```text
column name: asnfqosinqn
95% values: positive repeated integers
5% values: negative
```

This can be inferred as:

```json
{
  "column_role": "count_like",
  "role_confidence": 0.80,
  "role_reason": "values are mostly integer, non-negative, and repeated"
}
```

The negative values are then flagged because count-like columns should normally
not contain negatives.

## Method Selection

When `method=auto`, the detector chooses:

- `log_iqr` for `quantity`, `count_like`, `money`, and positive skewed columns
- `mad` for generic numeric columns
- `skip` for columns with too few usable values or too little variation

Manual methods are still supported:

- `iqr`
- `mad`
- `log_iqr`
- `zscore`
- `isolation_forest`
- `lof`

## Domain Rules

Domain rules are applied in addition to statistical detection:

- `quantity` and `count_like`: values below 0 are invalid
- `money`: values below 0 are invalid
- `percentage` and `percentage_like`: values below 0 or above 100 are invalid

Domain-rule failures are returned as `critical`.

Statistical outliers are returned as `warning`, because they might be valid
business events.

## Metadata Returned

Outlier metadata includes:

```json
{
  "method": "auto",
  "effective_method": "log_iqr",
  "severity": "warning",
  "column_role": "money",
  "role_confidence": 0.95,
  "role_reason": "column name looks like a price/money metric",
  "suggested_action": "review",
  "lower_bound": 0.12,
  "upper_bound": 98.70,
  "value_profile": {
    "integer_rate": 0.2,
    "non_negative_rate": 1.0,
    "unique_ratio": 0.35,
    "skew": 2.1
  }
}
```

## Why This Is Safer Than Plain IQR

Plain IQR treats every numeric column the same. On commerce datasets, this can
mark normal bulk orders as dirty values. For example, if most quantities are
small, plain IQR might flag `Quantity = 48` even though it is a valid bulk sale.

The new approach first asks what kind of column it is, then chooses a detector.
For quantities and prices, `log_iqr` is less aggressive on right-skewed data.
For generic numeric columns, `mad` is robust against extreme values.

## Limitations

No generic algorithm can know every business rule. A value can be statistically
rare but valid. The detector therefore defaults to `warning` and `review` for
statistical outliers.

For high-confidence cleaning, add explicit `column_rules` when domain knowledge
exists, for example allowed ranges, product-level price rules, or whether
negative quantities represent returns.
