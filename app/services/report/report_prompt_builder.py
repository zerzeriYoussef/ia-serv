"""
Report Prompt Builder - constructs the structured system + user prompts
sent to Gemini for report generation.

The report is generated section-by-section, but every section receives the
same deterministic FACT_PACK so the model writes from evidence instead of
guessing from vague context.
"""



from __future__ import annotations



from typing import Any, List



from langchain_core.prompts import (

    ChatPromptTemplate,

    HumanMessagePromptTemplate,

    SystemMessagePromptTemplate,

)





REPORT_SYSTEM_INSTRUCTION = """\
You are a senior BI analyst writing for non-technical business managers.
Your job is to turn dataset facts into a useful decision report.

CONTEXT SOURCES:
- FACT_PACK: trusted computed facts from the user's dataset. Prefer this.
- INTERNAL_CONTEXT: RAG snippets and analysis metadata from the dataset.
- CONVERSATION_CONTEXT: recent chat history showing what the user cares about.
- EXTERNAL_CONTEXT: web snippets, only when relevant and cited.

RULES:
1. Be evidence-led. Every important claim must be grounded in FACT_PACK or INTERNAL_CONTEXT.
2. Do not invent columns, segments, business domains, targets, sources, or recommendations.
3. Numbers should tell a story, not prove a statistic.
   Do include: "Correlation 0.89", "45% of total", "3.2x higher", "top segment".
   Do not include: "p-value<0.05", "95% CI", raw formulas, or academic jargon.
4. Use the requested language for all prose. Keep dataset column names unchanged.
5. If the dataset is historical/descriptive, recommend better analysis, dashboards,
   monitoring, or next data to collect instead of pretending the user can change the past.
6. Output ONLY the requested JSON object.
"""





def _make_template(human_template: str) -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages(

        [

            SystemMessagePromptTemplate.from_template(REPORT_SYSTEM_INSTRUCTION),

            HumanMessagePromptTemplate.from_template(human_template),

        ]

    )





WHAT_HAPPENED_TEMPLATE = _make_template(

    """\
REQUESTED_LANGUAGE:
{language}

FACT_PACK:
{fact_pack}

INTERNAL_CONTEXT:
{internal_context}

CONVERSATION_CONTEXT:
{conversation_context}

DATASET: {dataset_name}
PRIMARY_METRIC: {primary_metric}

Generate the what_happened section as JSON:
{{
  "narrative": "<2-3 short paragraphs. Start with the headline finding. Include at least 3 exact numbers from FACT_PACK. Explain what changed, what is high/low, and which segment or period matters most.>"
}}

Do not write a generic introduction. Do not say "it would be worth exploring"
when FACT_PACK already gives the answer.
"""

)





WHY_IT_HAPPENED_TEMPLATE = _make_template(

    """\
REQUESTED_LANGUAGE:
{language}

FACT_PACK:
{fact_pack}

INTERNAL_CONTEXT:
{internal_context}

CONVERSATION_CONTEXT:
{conversation_context}

EXTERNAL_CONTEXT:
{external_context}

Generate the why_it_happened section as JSON:
{{
  "narrative": "<2 short paragraphs explaining the most likely drivers visible in the data. Separate facts from possible interpretation. Include exact relationship strengths or ranked dimension facts when available.>",
  "sources": [
    {{"type": "internal", "title": "<specific metric, dimension, trend, or relationship used>"}},
    {{"type": "web", "title": "<Web article title>", "url": "<url from EXTERNAL_CONTEXT>"}}
  ]
}}

If EXTERNAL_CONTEXT is weak or irrelevant, ignore it and use only internal
sources. If you use EXTERNAL_CONTEXT, include a matching web source.
"""

)





WHAT_TO_DO_TEMPLATE = _make_template(

    """\
REQUESTED_LANGUAGE:
{language}

FACT_PACK:
{fact_pack}

WHAT_HAPPENED:
{what_happened}

WHY_IT_HAPPENED:
{why_it_happened}

CONVERSATION_CONTEXT:
{conversation_context}

Generate the what_to_do section as JSON:
{{
  "what_to_do": [
    {{
      "priority": <1=highest, 2, 3>,
      "action": "<specific action tied to one metric, period, segment, relationship, or data quality issue from FACT_PACK>",
      "expected_outcome": "<concrete expected outcome or decision this action enables>"
    }}
  ],
  "what_to_avoid": [
    "<1-2 specific things to avoid, grounded in the data>"
  ]
}}

Return exactly 3 recommendations. Avoid generic advice.
"""

)





def build_what_happened_prompt(

    internal_ctx: str,

    fact_pack: str,

    conversation_ctx: str,

    dataset_name: str,

    primary_metric: str,

    language: str = "French",

) -> List[Any]:

    return WHAT_HAPPENED_TEMPLATE.format_messages(

        internal_context=internal_ctx,

        fact_pack=fact_pack,

        conversation_context=conversation_ctx,

        dataset_name=dataset_name,

        primary_metric=primary_metric,

        language=language,

    )





def build_why_it_happened_prompt(

    internal_ctx: str,

    fact_pack: str,

    conversation_ctx: str,

    external_ctx: str,

    language: str = "French",

) -> List[Any]:

    return WHY_IT_HAPPENED_TEMPLATE.format_messages(

        internal_context=internal_ctx,

        fact_pack=fact_pack,

        conversation_context=conversation_ctx,

        external_context=external_ctx,

        language=language,

    )





def build_what_to_do_prompt(

    what_happened: str,

    why_it_happened: str,

    fact_pack: str,

    conversation_ctx: str,

    language: str = "French",

) -> List[Any]:

    return WHAT_TO_DO_TEMPLATE.format_messages(

        what_happened=what_happened,

        why_it_happened=why_it_happened,

        fact_pack=fact_pack,

        conversation_context=conversation_ctx,

        language=language,

    )

