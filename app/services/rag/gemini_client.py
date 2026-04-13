"""
Async Gemini generateContent client (JSON mode + streaming) via HTTP.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_STREAM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"


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


async def generate_json_with_schema(
    system_instruction: str,
    user_prompt: str,
    response_schema: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """
    Like generate_json_response but passes a responseSchema to Gemini for
    strict structured output (Gemini 2.x feature).
    """
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured")

    mdl = model or settings.GEMINI_MODEL
    url = GEMINI_URL.format(model=mdl)

    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, params={"key": key}, json=body)
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
    text = parts[0].get("text") or "" if parts else ""
    cleaned = _strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Gemini JSON parse error (schema mode): %s | snippet=%s", e, cleaned[:400])
        raise ValueError("Model did not return valid JSON") from e


async def stream_text_response(
    system_instruction: str,
    user_prompt: str,
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = 90.0,
) -> AsyncGenerator[str, None]:
    """
    Stream text tokens from Gemini's streamGenerateContent endpoint.

    Yields individual text deltas as they arrive from the SSE-like chunked
    response. The caller is responsible for assembling the full answer.
    """
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured")

    mdl = model or settings.GEMINI_MODEL
    url = GEMINI_STREAM_URL.format(model=mdl)

    contents: List[Dict[str, Any]] = list(history or [])
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})

    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
        },
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        async with client.stream(
            "POST",
            url,
            params={"key": key, "alt": "sse"},
            json=body,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                candidates = chunk.get("candidates") or []
                for cand in candidates:
                    parts = (cand.get("content") or {}).get("parts") or []
                    for part in parts:
                        delta = part.get("text", "")
                        if delta:
                            yield delta
