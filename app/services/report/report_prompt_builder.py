"""
Report Prompt Builder — constructs the structured system + user prompts
sent to Gemini for report generation.

Uses LangChain's ChatPromptTemplate for clean, composable prompt engineering.
The report is generated section-by-section so each prompt is focused.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ---------------------------------------------------------------------------
# System instruction (shared across all section prompts)
# ---------------------------------------------------------------------------

REPORT_SYSTEM_INSTRUCTION = """\
You are a senior BI analyst: explain the data in plain language to non-technical business managers.
They just want a conversation: "What happened?", "Why?", and "What should I do?"

CONTEXT SOURCES:
- INTERNAL_CONTEXT: the user's dataset (RAG and KPIs).
- CONVERSATION_CONTEXT: the recent chat history with the user (if any), indicating what they care about.
- EXTERNAL_CONTEXT: industry benchmarks and trends from web search.

RULES:
1. No statistical jargon. You must follow the GOLDEN RULE: Numbers should TELL A STORY, not PROVE A STATISTIC.
   DO include: "Correlation 0.89", "45% of total screen time", "3.2x higher", "Top 10% of users"
   DON'T include: "r=0.89, p-value<0.05", "percentage: 45.2%, std dev: 2.3", "95% CI"
2. No complex delta calculations or gap-to-target metrics unless explicitly asked in CONVERSATION_CONTEXT.
3. Be conversational, direct, and actionable.
4. Output ONLY the requested JSON object.
"""

# ---------------------------------------------------------------------------
# Per-section prompt templates (LangChain ChatPromptTemplate)
# ---------------------------------------------------------------------------

def _make_template(human_template: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(REPORT_SYSTEM_INSTRUCTION),
        HumanMessagePromptTemplate.from_template(human_template),
    ])


# ── What Happened ──────────────────────────────────────────────────────────
WHAT_HAPPENED_TEMPLATE = _make_template("""\
INTERNAL_CONTEXT:
{internal_context}

CONVERSATION_CONTEXT (What the user has been asking about):
{conversation_context}

DATASET: {dataset_name} | Primary metric: {primary_metric}

Generate the what_happened section as JSON:
{{
  "narrative": "<1-2 paragraphs plain language explanation of the current state of the business based on the data and what the user cares about>"
}}
""")

# ── Why It Happened ────────────────────────────────────────────────────────
WHY_IT_HAPPENED_TEMPLATE = _make_template("""\
INTERNAL_CONTEXT (Relationships and dimensions):
{internal_context}

CONVERSATION_CONTEXT:
{conversation_context}

EXTERNAL_CONTEXT (Serper web snippets [WEB-n]):
{external_context}

Generate the why_it_happened section as JSON:
{{
  "narrative": "<1-2 paragraphs plain language explanation of the root causes, blending internal data relationships and external industry context>",
  "sources": [
    {{"type": "internal", "title": "<e.g. daily_screen_time ↔ weekend_screen_time>"}},
    {{"type": "web", "title": "<Web article title>", "url": "<url from [WEB-n]>"}}
  ]
}}
IMPORTANT: If you incorporate any information from EXTERNAL_CONTEXT, you MUST include a corresponding object with "type": "web" in the sources array, using its title and URL.
""")

# ── What To Do ─────────────────────────────────────────────────────────────
WHAT_TO_DO_TEMPLATE = _make_template("""\
WHAT_HAPPENED:
{what_happened}

WHY_IT_HAPPENED:
{why_it_happened}

CONVERSATION_CONTEXT:
{conversation_context}

Generate the what_to_do section as a JSON object containing what_to_do and what_to_avoid:
{{
  "what_to_do": [
    {{
      "priority": <1=highest, 2, 3>,
      "action": "<specific, actionable recommendation in plain language>",
      "expected_outcome": "<plausible business outcome in plain language>"
    }}
  ],
  "what_to_avoid": [
    "<Optional: 1-2 things the user should NOT do or watch out for, based on the context>"
  ]
}}
""")


# ---------------------------------------------------------------------------
# Prompt formatters — format a template with context data
# ---------------------------------------------------------------------------

def build_what_happened_prompt(
    internal_ctx: str,
    conversation_ctx: str,
    dataset_name: str,
    primary_metric: str,
) -> List[Any]:
    return WHAT_HAPPENED_TEMPLATE.format_messages(
        internal_context=internal_ctx,
        conversation_context=conversation_ctx,
        dataset_name=dataset_name,
        primary_metric=primary_metric,
    )


def build_why_it_happened_prompt(
    internal_ctx: str,
    conversation_ctx: str,
    external_ctx: str,
) -> List[Any]:
    return WHY_IT_HAPPENED_TEMPLATE.format_messages(
        internal_context=internal_ctx,
        conversation_context=conversation_ctx,
        external_context=external_ctx,
    )


def build_what_to_do_prompt(
    what_happened: str,
    why_it_happened: str,
    conversation_ctx: str,
) -> List[Any]:
    return WHAT_TO_DO_TEMPLATE.format_messages(
        what_happened=what_happened,
        why_it_happened=why_it_happened,
        conversation_context=conversation_ctx,
    )

