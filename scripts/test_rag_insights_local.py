"""
Local tester for the RAG insights flow.

It uses:
- your example analysis JSON (no DB required)
- the local markdown knowledge base under `app/knowledge/rag/*.md`
- the existing Gemini client in `app/services/rag/gemini_client.py`

This script is meant for quick validation of the *response shape*:
`executive_summary_kpis` + `dashboard_charts`.

Usage:
  python scripts/test_rag_insights_local.py --dry-run
  python scripts/test_rag_insights_local.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from app.services.rag.gemini_client import generate_json_response
from app.services.rag.knowledge_retriever import retrieve_for_query


logger = logging.getLogger(__name__)


EXAMPLE_PAYLOAD: Dict[str, Any] = json.loads("""
{
  "dataset_id": 404,
  "relationships": [
    { "columns": ["Bytes_Sent", "Total_Traffic"], "type": "correlation", "strength": 1.0, "direction": "positive", "method": "pearson", "insight": "Identical metrics" },
    { "columns": ["Login_Attempts", "Security_Alerts"], "type": "correlation", "strength": 0.94, "direction": "positive", "method": "pearson", "insight": "Brute force pattern detected" },
    { "columns": ["Security_Patch_Level", "Infection_Probability"], "type": "correlation", "strength": -0.92, "direction": "negative", "method": "spearman", "insight": "Updated systems are safer" },
    { "columns": ["Admin_ID", "Threat_Score"], "type": "correlation", "strength": 0.01, "direction": "neutral", "method": "pearson", "insight": "Complete noise" },
    { "columns": ["Server_Location", "Latency_ms"], "type": "association", "strength": 0.88, "direction": null, "method": "cramers_v", "insight": "Distance impacts speed" },
    { "columns": ["User_Role", "Access_Level"], "type": "association", "strength": 0.98, "direction": null, "method": "cramers_v", "insight": "Strict permission hierarchy" },
    { "columns": ["Time_of_Day", "Traffic_Volume"], "type": "correlation", "strength": 0.72, "direction": "positive", "method": "pearson", "insight": "Peak hour usage patterns" },
    { "columns": ["Firewall_Status", "Blocked_Packets"], "type": "association", "strength": 0.85, "direction": null, "method": "eta_squared", "insight": "Active firewall increases blocks" },
    { "columns": ["Department", "VPN_Usage"], "type": "association", "strength": 0.45, "direction": null, "method": "cramers_v", "insight": "Moderate departmental variation" },
    { "columns": ["Uptime_Days", "Hardware_Failure_Risk"], "type": "correlation", "strength": 0.68, "direction": "positive", "method": "pearson", "insight": "Older sessions risk crashes" },
    { "columns": ["Employee_Tenure", "Password_Strength"], "type": "correlation", "strength": 0.12, "direction": "positive", "method": "pearson", "insight": "Experience doesn't mean better security" },
    { "columns": ["Protocol_Type", "Encryption_Level"], "type": "association", "strength": 0.91, "direction": null, "method": "cramers_v", "insight": "Protocols dictate security" },
    { "columns": ["CPU_Usage", "Server_Temperature"], "type": "correlation", "strength": 0.96, "direction": "positive", "method": "pearson", "insight": "Physical heat follows load" },
    { "columns": ["Failed_Logins", "Account_Lockouts"], "type": "correlation", "strength": 0.89, "direction": "positive", "method": "pearson", "insight": "Direct policy enforcement" },
    { "columns": ["OS_Version", "Malware_Vulnerability"], "type": "association", "strength": 0.77, "direction": null, "method": "eta_squared", "insight": "Legacy OS versions are targets" },
    { "columns": ["Coffee_Consumption", "Code_Bugs"], "type": "correlation", "strength": 0.05, "direction": "positive", "method": "pearson", "insight": "Funny but irrelevant noise" },
    { "columns": ["Subscription_Plan", "Storage_Limit"], "type": "association", "strength": 1.0, "direction": null, "method": "cramers_v", "insight": "Plan strictly defines space" },
    { "columns": ["Incident_Response_Time", "Financial_Loss"], "type": "correlation", "strength": 0.82, "direction": "positive", "method": "pearson", "insight": "Slow response costs money" },
    { "columns": ["Wifi_Signal_Strength", "Download_Speed"], "type": "correlation", "strength": 0.79, "direction": "positive", "method": "pearson", "insight": "Infrastructure quality matters" },
    { "columns": ["Dark_Web_Mentions", "Active_Attacks"], "type": "correlation", "strength": 0.88, "direction": "positive", "method": "spearman", "insight": "Predictive threat intelligence" }
  ],
  "dashboard_columns": [
    "Threat_Score", "Total_Traffic", "Security_Alerts", "Login_Attempts", "Server_Status"
  ],
  "column_categories": {
    "metrics": ["Threat_Score", "Latency_ms", "Total_Traffic", "Failed_Logins", "CPU_Usage"],
    "dimensions": ["Server_Location", "User_Role", "Protocol_Type", "Department"],
    "identifiers": ["Server_ID", "Admin_ID"],
    "temporal": ["Timestamp"],
    "geographic": ["Warehouse_Region", "Data_Center_City"]
  },
  "primary_metric": "Threat_Score",
  "confidence_score": 0.91,
  "total_relationships": 20,
  "created_at": "2026-03-29T15:20:00.000Z"
}
""")

SYSTEM_PROMPT = """You are a senior analytics product lead. Output valid JSON ONLY.
Audience: executives and dashboard builders — no dumping internal analysis structures.
Use ONLY column names that appear in DATASET_CONTEXT or KPI_HINTS.
Downrank trivial links (IDs, emails, UUIDs → names). Prefer primary metric and real business dimensions.
Ground pandas_grouping and chart_type in RETRIEVED_KNOWLEDGE when it fits.
"""


def _build_slim_analysis_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    cats: Dict[str, Any] = payload.get("column_categories") or {}
    relationships: List[Dict[str, Any]] = payload.get("relationships") or []

    slim_relationships: List[Dict[str, Any]] = []
    for rel in relationships[:24]:
        slim_relationships.append(
            {
                "columns": rel.get("columns"),
                "type": rel.get("type"),
                "strength": rel.get("strength"),
                "method": rel.get("method"),
            }
        )

    dash_cols: List[str] = payload.get("dashboard_columns") or []

    return {
        "dataset_id": payload.get("dataset_id"),
        "row_count": payload.get("row_count", None),
        "primary_metric": payload.get("primary_metric"),
        "dashboard_columns": dash_cols[:12],
        "column_categories": {
            "metrics": cats.get("metrics") or [],
            "dimensions": cats.get("dimensions") or [],
            "temporal": cats.get("temporal") or [],
            "geographic": cats.get("geographic") or [],
            "identifiers": cats.get("identifiers") or [],
            "other": cats.get("other") or [],
        },
        "relationships": slim_relationships,
    }


def _build_query(analysis_context: Dict[str, Any], relationships: List[Dict[str, Any]]) -> str:
    cats = analysis_context.get("column_categories") or {}
    return " ".join(
        [*(cats.get("metrics") or []), *(cats.get("dimensions") or []), *(cats.get("temporal") or []), analysis_context.get("primary_metric") or ""]
        + [
            " ".join((r.get("columns") or [])) + " " + str(r.get("type") or "")
            for r in relationships[:20]
        ]
    )


def _kpi_hints_from_payload(payload: Dict[str, Any]) -> str:
    # In the real service, KPI_HINTS are computed from the dataframe.
    # Here we only have analysis JSON, so we provide a safe fallback.
    # If you add a `kpi` object to the payload later, you can parse it here.
    primary = payload.get("primary_metric")
    row_count = payload.get("row_count")
    if not primary:
        return "No primary metric KPI could be computed."
    return f"Primary metric: {primary}. row_count={row_count if row_count is not None else 'unknown'}."


def build_user_prompt(analysis_context: Dict[str, Any], kpi_hints: str, retrieved_kb: str) -> str:
    return f"""DATASET_CONTEXT (for column names and structure only — do not echo this object in your answer):
{json.dumps(analysis_context, default=str)}

KPI_HINTS (one-line facts computed in our backend — use to inspire KPI names and copy, do NOT paste this block into the response):
{kpi_hints}

RETRIEVED_KNOWLEDGE:
{retrieved_kb}

Return a SINGLE JSON object with EXACTLY these two top-level keys and nothing else:

{{
  "executive_summary_kpis": [
    {{
      "kpi_name": "Short professional name",
      "target_column": "exact column name from context",
      "pandas_function": "e.g. .sum() or groupby pattern label",
      "logic_hint": "df['col'].mean() or one-line pandas",
      "description": "Why this KPI matters (non-technical)",
      "icon": "material icon name e.g. trending_up, payments, groups"
    }}
  ],
  "dashboard_charts": [
    {{
      "rank": 1,
      "title": "Chart title for a dashboard tile",
      "relationship": "Column A ↔ Column B",
      "strength_score": 0.0,
      "x_axis_column": "column or null",
      "y_axis_column": "column or null",
      "pandas_grouping": "df.groupby('x')['y'].sum() style one-liner",
      "chart_type": "e.g. Grouped Bar, Line, Scatter, Heatmap, Box Plot",
      "business_insight": "What this visual proves for a business reader"
    }}
  ]
}}

Rules:
- 4–8 items in executive_summary_kpis.
- 6–10 items in dashboard_charts, sorted by rank.
- strength_score between 0 and 1 (use relationship strength from context when available).
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt and retrieved chunks only.")
    parser.add_argument("--top-k", type=int, default=6, help="How many knowledge chunks to retrieve.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    analysis_context = _build_slim_analysis_context(EXAMPLE_PAYLOAD)
    relationships = analysis_context.get("relationships") or []

    query = _build_query(analysis_context, relationships)
    chunks = retrieve_for_query(query, top_k=args.top_k)
    kb_text = "\n\n---\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)

    kpi_hints = _kpi_hints_from_payload(EXAMPLE_PAYLOAD)
    user_prompt = build_user_prompt(analysis_context, kpi_hints, kb_text)

    print("=== Retrieval meta ===")
    print("retrieved_chunk_ids:", [c.chunk_id for c in chunks])
    print("query_preview:", query[:200])

    if args.dry_run:
        print("\n=== Prompt preview ===")
        print(user_prompt[:2000])
        return

    # Gemini call (requires GEMINI_API_KEY in your environment)
    result = asyncio.run(generate_json_response(SYSTEM_PROMPT, user_prompt))
    print("\n=== Gemini JSON result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

