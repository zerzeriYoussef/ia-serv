# Implementation summary (IA-service)

This document summarizes major work completed on the **AI Data Analytics Service**: dependencies, data analysis, cleaning, API behavior, RAG/Gemini insights, and related fixes.

---

## Dependencies & environment

- **`ydata-profiling`** is listed in `requirements.txt` for profiling; it depends on **`pkg_resources`**, which was removed in **setuptools 82+**. The project pins **`setuptools<82`** so installs and imports remain reliable (including Docker).
- **`ppscore`** was removed from `requirements.txt` because it conflicts with modern **pandas** / **scikit-learn** and failed to build on Python 3.13; relationship detection uses **CustomPPS** and other in-house logic instead.
- **SciPy / statsmodels** support statistical relationship features where used.

---

## Database & models

- **`ColumnAnalysis`** (`app/models/column_analysis.py`) stores per-dataset analysis: column categories (metrics, dimensions, identifiers, temporal, geographic, other), relationships JSON, dashboard columns, primary metric, confidence, etc.
- **Alembic** imports this model in `migrations/env.py` so autogenerate picks up the table. Migrations were generated and applied for **`column_analysis`**.

---

## File parsing

- **`ParserService`** (`app/services/data/parser_service.py`) uses a **robust CSV path**: on `ParserError`, it retries with delimiter sniffing, the Python engine, and **`on_bad_lines="skip"`** so malformed rows do not crash analysis or cleaning endpoints.

---

## Column analysis

- **`ColumnAnalyzer`** (`app/services/analysis/column_analyzer.py`): categorizes columns, runs **CustomPPS**, correlations, merges results, suggests dashboard columns and primary metric.
- **Identifier handling**: PPS runs on a DataFrame **without** identifier columns; **`RelationshipDetector._is_identifier`** (advanced heuristics) aligns with categorization so emails / IDs are not used for trivial predictive edges where possible.
- **`ColumnAnalyzerImproved`** (`column_analyzer_improved.py`): optional extended analyzer with chi-square / ANOVA-style statistical filters (when used).
- **`RelationshipDetector`**: hierarchy detection excludes ID-like columns; hierarchy edges include **`method: "hierarchy_detector"`** for API schema compliance.

---

## Analysis API

- **`RelationshipSchema`** requires **`method`**. Older DB rows lacked it; **`_normalize_relationships`** in `analysis.py` backfills **`method`** by relationship type when returning cached analysis.
- **KPI block removed from analysis responses**: `AnalysisResultSchema` no longer includes **`kpi`**; `POST /datasets/{id}/analyze` and `GET /datasets/{id}/analysis` do not compute or return the structured KPI object. Dashboard-oriented KPIs are available via **RAG insights** (below).

---

## RAG + Gemini insights

- **Config** (`app/core/config.py`): optional **`GEMINI_API_KEY`**, **`GEMINI_MODEL`** (default `gemini-2.0-flash`; override if your project only exposes e.g. `gemini-1.5-flash`).
- **Knowledge base** (`app/knowledge/rag/*.md`): short playbooks (pandas groupby, time series, correlation, categorical association, guardrails) with YAML `tags` for retrieval.
- **Pipeline** (`app/services/rag/`):
  - **`knowledge_retriever.py`**: loads markdown chunks, scores by token overlap with a query built from analysis columns/types.
  - **`gemini_client.py`**: async **Google Generative Language API** `generateContent` with **JSON** response mode.
  - **`rag_insights_service.py`**: loads stored analysis + file once, builds **compact** context and **KPI hints** (text only, for the model prompt—not echoed as a big JSON blob), retrieves chunks, calls Gemini.
- **Endpoint**: `POST /api/v1/datasets/{dataset_id}/rag-insights`
  - Response: **`executive_summary_kpis`**, **`dashboard_charts`**, **`dataset_id`**.
  - Optional **`?debug=true`**: adds **`rag_meta`** (chunk IDs, query preview).
  - Requires analysis to exist (`POST .../analyze` first) and **`GEMINI_API_KEY`** set.

---

## Cleaning

- Cleaning flows use **`ParserService`**; messy CSVs benefit from the robust reader described above.
- Cleaning profiles and **`/clean`** behavior are separate from this summary; see `cleaning` routes and `cleaning_service`.

---

## Running the API

- App module: **`app.main:app`** (not `main:app`).
- Example: `uvicorn app.main:app --reload` (port from settings, often **8001**).
- **`fastapi dev`** needs **`fastapi[standard]`** if you use the CLI.

---

## Files worth knowing

| Area | Paths |
|------|--------|
| Analysis routes | `app/api/v1/routes/analysis.py` |
| RAG route | `app/api/v1/routes/rag.py` |
| RAG schemas | `app/api/v1/schemas/rag_schema.py` |
| Analysis schemas | `app/api/v1/schemas/analysis_schema.py` |
| KPI builder (internal / RAG hints) | `app/services/analysis/kpi_service.py` |
| Main app | `app/main.py` |

---

*Last updated from project context: March 2026.*
