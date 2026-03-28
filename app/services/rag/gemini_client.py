"""
Async Gemini generateContent client (JSON mode) via HTTP.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if m:
        return m.group(1).strip()
    return text


async def generate_json_response(
    system_instruction: str,
    user_prompt: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured")

    mdl = model or settings.GEMINI_MODEL
    url = GEMINI_URL.format(model=mdl)
    params = {"key": key}

    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.25,
            "responseMimeType": "application/json",
        },
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, params=params, json=body)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("Gemini HTTP error: %s %s", e.response.status_code, e.response.text[:500])
            raise

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts:
        raise ValueError("Gemini returned empty content")

    text = parts[0].get("text") or ""
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Gemini JSON parse error: %s | snippet=%s", e, cleaned[:400])
        raise ValueError("Model did not return valid JSON") from e
