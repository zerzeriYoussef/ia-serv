import asyncio
import httpx

from app.core.config import settings

async def test_stream():
    key = settings.GEMINI_API_KEY
    mdl = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:streamGenerateContent"

    body = {
        "systemInstruction": {"parts": [{"text": "You are a friendly data analyst explaining results to a business user."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "[CONVERSATION SUMMARY SO FAR]\nSummary"}]},
            {"role": "model", "parts": [{"text": "Understood. I have the conversation summary."}]},
            {"role": "user", "parts": [{"text": "How has coffee consumption changed over the years?"}]}
        ],
        "generationConfig": {"temperature": 0.4}
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, params={"key": key, "alt": "sse"}, json=body)
        print("Status Code:", resp.status_code)
        print("Response:", resp.text)

asyncio.run(test_stream())
