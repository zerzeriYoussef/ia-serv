#!/usr/bin/env python3
"""
================================================================================
CHAT API — MANUAL / SMOKE TEST SCRIPT
================================================================================

WHAT THIS FILE IS
-----------------
A command-line helper that exercises the FastAPI **Chat** routes under
`/api/v1/datasets/{dataset_id}/conversations...`. It is NOT pytest; it is meant
for humans (or CI smoke jobs) to verify the service is up, auth works, DB +
Gemini are wired, and both JSON and SSE flows behave sensibly.

WHY IT EXISTS
-------------
- Chat requires a **valid JWT** whose `sub` matches a user that **owns** at
  least one dataset with **analysis** already run (`POST .../analyze`).
- SSE clients are awkward to test in a browser alone; this script prints
  parsed events so you can see `intent` → `tool_*` → `token` → `done`.

HOW TO RUN
----------
From the **project root** (directory that contains `app/`):

  .\\venv\\Scripts\\python.exe scripts\\test_chat_api.py

Optional flags (see `main()`):

  --base-url http://127.0.0.1:8001
  --user-id <uuid>     JWT `sub`; must own the dataset (default: from env CHAT_TEST_USER_ID)
  --dataset-id N       Skip discovery via /datasets/my-datasets
  --skip-stream        Only test non-streaming POST /messages
  --skip-ask           Only create thread + list messages (no LLM calls)
  --index              After tests, POST .../datasets/{id}/index (background Chroma rebuild)

Environment:
  - `.env` / `SECRET_KEY` — used to **mint** a test JWT (same secret as the API).
  - `GEMINI_API_KEY` — must be set on the **server** for ask/stream/index to work.

================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

# ---------------------------------------------------------------------------
# HOW: Make `import app.*` work when this file lives in `scripts/`
#
# WHY: Running `python scripts/test_chat_api.py` does not automatically put the
#      repo root on PYTHONPATH. Without this, `from app.core.config` fails.
#
# WHAT: Insert parent of `scripts/` (the project root) at the front of sys.path.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
from jose import jwt

from app.core.config import settings


# ---------------------------------------------------------------------------
# DEFAULTS — change here or override via CLI / env in main()
#
# WHAT: Base URL of the running uvicorn instance.
# WHY: The app defaults to port 8001 in Settings; many older notes use 8000.
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = f"http://{settings.HOST if settings.HOST != '0.0.0.0' else '127.0.0.1'}:{settings.PORT}"


def build_bearer_token(user_id: str, *, role: str = "user", hours: int = 2) -> str:
    # =========================================================================
    # WHAT
    #   Returns a signed JWT string suitable for `Authorization: Bearer ...`.
    #
    # WHY
    #   All Chat routes use `Depends(require_auth)`. The API decodes `sub` as
    #   the owning user id for listing datasets and scoping conversations.
    #   Wrong `sub` → empty `my-datasets` or 404 on resources you do not own.
    #
    # HOW
    #   We sign with `settings.SECRET_KEY` and `settings.ALGORITHM` — the same
    #   values the running server uses — so the token is accepted locally.
    #
    # SHOULD RETURN
    #   A compact JWT (three dot-separated segments). Not the decoded payload.
    # =========================================================================
    payload = {
        "sub": user_id,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=hours),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def check_health(client: httpx.Client, base: str) -> bool:
    # =========================================================================
    # WHAT
    #   GET `/health` (no auth, no `/api/v1` prefix).
    #
    # WHY
    #   Fast fail if uvicorn is not running or points at the wrong port.
    #
    # HOW
    #   Simple GET; no JSON schema requirement beyond `status` optional.
    #
    # SHOULD RETURN
    #   HTTP 200 and a small JSON body (e.g. `{"status":"healthy",...}`).
    #   Prints the body and returns True on success; False on network/HTTP error.
    # =========================================================================
    url = f"{base.rstrip('/')}/health"
    try:
        r = client.get(url, timeout=10.0)
    except httpx.RequestError as e:
        print(f"[health] REQUEST FAILED: {e}")
        return False
    print(f"[health] GET {url} -> {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text[:500])
    return r.status_code == 200


def fetch_first_dataset_id(
    client: httpx.Client,
    base_api: str,
    headers: Dict[str, str],
) -> Optional[int]:
    # =========================================================================
    # WHAT
    #   GET `/api/v1/datasets/my-datasets` and pick the first dataset `id`.
    #
    # WHY
    #   Chat is always scoped to a `dataset_id`. You need one that exists and
    #   belongs to the JWT `sub`. Listing "my" datasets is the safest discovery
    #   path (respects ownership rules in upload routes).
    #
    # HOW
    #   Paginated response: we only read the first page (`page_size` default).
    #
    # SHOULD RETURN
    #   - `int` dataset id when `datasets` is non-empty.
    #   - `None` when the list is empty (wrong user, or no uploads yet).
    #
    # TYPICAL FAILURE MODES
    #   - 401: missing/invalid Bearer token.
    #   - 200 + empty list: `sub` does not match `datasets.user_id` in DB.
    # =========================================================================
    url = f"{base_api.rstrip('/')}/datasets/my-datasets"
    r = client.get(url, headers=headers, timeout=30.0)
    print(f"[my-datasets] GET {url} -> {r.status_code}")
    if r.status_code != 200:
        print(r.text[:800])
        return None
    data = r.json()
    items = data.get("datasets") or []
    if not items:
        print("[my-datasets] No datasets for this user. Upload + analyze first, or pass --dataset-id + --user-id owner.")
        return None
    ds_id = int(items[0]["id"])
    print(f"[my-datasets] Using dataset_id={ds_id} (first page, first row)")
    return ds_id


def create_conversation(
    client: httpx.Client,
    base_api: str,
    headers: Dict[str, str],
    dataset_id: int,
    title: str = "test_chat_api script",
) -> Optional[int]:
    # =========================================================================
    # WHAT
    #   POST `/api/v1/datasets/{dataset_id}/conversations` with optional JSON body.
    #
    # WHY
    #   Every chat turn is tied to a conversation row. This creates the thread
    #   and records `analysis_version` for staleness warnings later.
    #
    # HOW
    #   JSON body matches `CreateConversationRequest` (optional `title`).
    #
    # SHOULD RETURN
    #   HTTP 201 and `CreateConversationResponse`:
    #     - conversation_id (int)
    #     - dataset_id
    #     - title
    #     - created_at (ISO timestamp)
    #
    # TYPICAL FAILURE MODES
    #   - 404: dataset_id does not exist.
    #   - 401: auth failure.
    # =========================================================================
    url = f"{base_api.rstrip('/')}/datasets/{dataset_id}/conversations"
    r = client.post(url, headers=headers, json={"title": title}, timeout=30.0)
    print(f"[create_conversation] POST {url} -> {r.status_code}")
    print(r.text[:1200])
    if r.status_code != 201:
        return None
    return int(r.json()["conversation_id"])


def get_messages(
    client: httpx.Client,
    base_api: str,
    headers: Dict[str, str],
    dataset_id: int,
    conversation_id: int,
) -> List[Dict[str, Any]]:
    # =========================================================================
    # WHAT
    #   GET `/api/v1/datasets/{dataset_id}/conversations/{id}/messages`
    #
    # WHY
    #   Verifies persistence and ordering contract: API returns **chronological**
    #   order (oldest first) even though the repository may store newest-first.
    #
    # HOW
    #   Optional `limit` query (default 50). We use defaults.
    #
    # SHOULD RETURN
    #   HTTP 200 and a JSON array of `MessageSchema` objects:
    #     id, conversation_id, role (`user`|`assistant`), content, metadata fields...
    #
    # TYPICAL FAILURE MODES
    #   - 404: conversation not found or not for this dataset_id.
    # =========================================================================
    url = (
        f"{base_api.rstrip('/')}/datasets/{dataset_id}/conversations/"
        f"{conversation_id}/messages"
    )
    r = client.get(url, headers=headers, timeout=30.0)
    print(f"[get_messages] GET {url} -> {r.status_code}")
    if r.status_code != 200:
        print(r.text[:800])
        return []
    messages = r.json()
    print(f"[get_messages] count={len(messages)}")
    if messages:
        print(json.dumps(messages[-1], indent=2, default=str))
    return messages


def ask_non_streaming(
    client: httpx.Client,
    base_api: str,
    headers: Dict[str, str],
    dataset_id: int,
    conversation_id: int,
    message: str,
) -> Optional[Dict[str, Any]]:
    # =========================================================================
    # WHAT
    #   POST `/api/v1/datasets/{dataset_id}/conversations/{id}/messages`
    #   with body `{"message": "..."}`.
    #
    # WHY
    #   Same orchestrator as SSE, but the route **consumes** the whole generator
    #   and returns one JSON object — easier for scripts and Postman.
    #
    # HOW
    #   The route persists the user message, runs `run_chat_turn`, persists the
    #   assistant message, returns `AskResponse`.
    #
    # SHOULD RETURN
    #   HTTP 200 and `AskResponse`:
    #     - message_id: DB id of the assistant row
    #     - conversation_id
    #     - answer: full narrative string
    #     - intent: orchestrator intent string (e.g. retrieve_only, analyze)
    #     - chunk_ids: RAG chunk ids used (may be [] if Chroma empty / not indexed)
    #     - chart_spec: optional dict
    #     - latency_ms
    #
    # TYPICAL FAILURE MODES
    #   - 503: `GEMINI_API_KEY` not set on the server (see `_require_gemini`).
    #   - Timeouts: slow Gemini or large CSV parse — increase httpx timeout.
    # =========================================================================
    url = (
        f"{base_api.rstrip('/')}/datasets/{dataset_id}/conversations/"
        f"{conversation_id}/messages"
    )
    r = client.post(url, headers=headers, json={"message": message}, timeout=180.0)
    print(f"[ask_json] POST {url} -> {r.status_code}")
    if r.status_code != 200:
        print(r.text[:2000])
        return None
    body = r.json()
    print(json.dumps(body, indent=2, default=str)[:4000])
    return body


def parse_sse_bytes(raw: str) -> List[Tuple[str, Dict[str, Any]]]:
    # =========================================================================
    # WHAT
    #   Parse a chunk of SSE text into a list of `(event_name, data_dict)` pairs.
    #
    # WHY
    #   The streaming endpoint emits multiple frames:
    #     event: intent
    #     data: {...}
    #
    #     (blank line separates events)
    #
    #     Clients must accumulate `event:` + following `data:` lines. This
    #     helper does a **best-effort** parse of whatever we read in one buffer.
    #
    # HOW
    #   Split on blank lines → event blocks; within each block read `event:` and
    #   one `data:` line, `json.loads` the payload.
    #
    # SHOULD RETURN
    #   List of tuples, e.g. `("token", {"delta": "Hello"})`.
    #
    # NOTE
    #   Production clients should stream incrementally; we read a bounded prefix
    #   for smoke testing only.
    # =========================================================================
    events: List[Tuple[str, Dict[str, Any]]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev_name = "message"
        data_line = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_line = line.split(":", 1)[1].strip()
        if not data_line:
            continue
        try:
            events.append((ev_name, json.loads(data_line)))
        except json.JSONDecodeError:
            events.append((ev_name, {"_raw": data_line}))
    return events


def ask_streaming_sample(
    client: httpx.Client,
    base_api: str,
    bearer: str,
    dataset_id: int,
    conversation_id: int,
    message: str,
    max_bytes: int = 32_768,
) -> None:
    # =========================================================================
    # WHAT
    #   GET `/api/v1/datasets/{dataset_id}/conversations/{id}/messages/stream`
    #   with query param `message=` (URL-encoded) and `Accept: text/event-stream`.
    #
    # WHY
    #   This is the “ChatGPT-like” path: tokens arrive as `event: token` deltas.
    #   Also exposes `tool_start` / `tool_result` for debugging analysis steps.
    #
    # HOW
    #   httpx `stream("GET", ...)`, read up to `max_bytes` of the body, parse
    #   SSE blocks. The server sends `message_persisted` after DB write with the
    #   real `message_id` (the prior `done` event often has message_id=0 by design).
    #
    # SHOULD RETURN (at the HTTP level)
    #   HTTP 200, `Content-Type` related to text/event-stream, body is SSE frames.
    #
    # EVENT SEQUENCE (typical, best-effort)
    #   intent → retrieval → (tool_start → tool_result)* → token* → done →
    #   optional message_persisted
    #
    # TYPICAL FAILURE MODES
    #   - 503: Gemini not configured on server.
    #   - SSE mid-stream error → `event: error` frame.
    # =========================================================================
    q = quote(message, safe="")
    url = (
        f"{base_api.rstrip('/')}/datasets/{dataset_id}/conversations/"
        f"{conversation_id}/messages/stream?message={q}"
    )
    headers = {
        "Authorization": f"Bearer {bearer}",
        "Accept": "text/event-stream",
    }
    print(f"[ask_stream] GET {url[:120]}...")
    with client.stream("GET", url, headers=headers, timeout=180.0) as r:
        print(f"[ask_stream] status={r.status_code} content-type={r.headers.get('content-type')}")
        buf: List[str] = []
        total = 0
        for chunk in r.iter_text():
            buf.append(chunk)
            total += len(chunk)
            if total >= max_bytes:
                break
    raw = "".join(buf)
    print(f"[ask_stream] captured ~{len(raw)} chars (cap {max_bytes})")
    events = parse_sse_bytes(raw)
    for name, payload in events[:40]:
        preview = json.dumps(payload, default=str)
        if len(preview) > 220:
            preview = preview[:220] + "…"
        print(f"  sse: {name:18} {preview}")


def trigger_index(
    client: httpx.Client,
    base_api: str,
    headers: Dict[str, str],
    dataset_id: int,
) -> None:
    # =========================================================================
    # WHAT
    #   POST `/api/v1/datasets/{dataset_id}/index`
    #
    # WHY
    #   Chat retrieval (`chunk_ids` in responses) uses Chroma. If you never
    #   indexed, `retrieval` events show empty `chunk_ids` — the orchestrator
    #   still works from `DATASET_CONTEXT`, but RAG context is weaker.
    #
    # HOW
    #   Schedules a FastAPI `BackgroundTasks` job; HTTP returns immediately with
    #   a small JSON status (does not wait for indexing to finish).
    #
    # SHOULD RETURN
    #   HTTP 200 and a JSON object like:
    #     {"status":"indexing_started","dataset_id":...,"message":"..."}
    #
    # TYPICAL FAILURE MODES
    #   - 400: no analysis row for dataset (run POST .../analyze first).
    #   - 503: Gemini not configured (embedding calls need API key on server).
    # =========================================================================
    url = f"{base_api.rstrip('/')}/datasets/{dataset_id}/index"
    r = client.post(url, headers=headers, timeout=60.0)
    print(f"[index] POST {url} -> {r.status_code}")
    print(r.text[:800])


def main() -> int:
    # =========================================================================
    # WHAT
    #   CLI entry: parse args, run the sequence of checks above.
    #
    # WHY
    #   One command to validate the full stack you care about before UI work.
    #
    # HOW
    #   argparse → build token → httpx.Client → ordered calls.
    #
    # EXIT CODES
    #   0 — all requested steps succeeded
    #   1 — health failed, no dataset, conversation create failed, or ask failed
    # =========================================================================
    p = argparse.ArgumentParser(description="Smoke-test Chat API (REST + SSE sample).")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server root (e.g. http://127.0.0.1:8001)")
    p.add_argument(
        "--user-id",
        default=None,
        help="JWT `sub` (must own the dataset). Default: env CHAT_TEST_USER_ID or prompt failure if missing.",
    )
    p.add_argument("--dataset-id", type=int, default=None, help="Skip my-datasets discovery")
    p.add_argument("--skip-stream", action="store_true", help="Do not call the SSE endpoint")
    p.add_argument("--skip-ask", action="store_true", help="Do not call Gemini (no POST/stream ask)")
    p.add_argument("--index", action="store_true", help="After success, POST .../datasets/{id}/index")
    p.add_argument("--no-health", action="store_true", help="Skip GET /health")
    args = p.parse_args()

    import os

    user_id = args.user_id or os.environ.get("CHAT_TEST_USER_ID")
    if not user_id:
        print(
            "ERROR: pass --user-id <uuid> or set CHAT_TEST_USER_ID to a user that owns a dataset.",
            file=sys.stderr,
        )
        return 1

    base = args.base_url.rstrip("/")
    base_api = f"{base}/api/v1"

    # -------------------------------------------------------------------------
    # WHAT: Mint local JWT.
    # WHY: Chat routes require Bearer auth; we avoid depending on external auth service.
    # -------------------------------------------------------------------------
    token = build_bearer_token(user_id)
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client() as client:
        if not args.no_health:
            if not check_health(client, base):
                return 1

        dataset_id = args.dataset_id
        if dataset_id is None:
            dataset_id = fetch_first_dataset_id(client, base_api, headers)
        if dataset_id is None:
            return 1

        conv_id = create_conversation(client, base_api, headers, dataset_id)
        if conv_id is None:
            return 1

        get_messages(client, base_api, headers, dataset_id, conv_id)

        if args.skip_ask:
            print("[main] --skip-ask: stopping before LLM calls.")
            if args.index:
                trigger_index(client, base_api, headers, dataset_id)
            return 0

        # ---------------------------------------------------------------------
        # WHAT: One non-streaming question.
        # WHY: Exercises full orchestrator + persistence in one JSON response.
        # ---------------------------------------------------------------------
        ask_body = ask_non_streaming(
            client,
            base_api,
            headers,
            dataset_id,
            conv_id,
            "How many rows does this dataset have?",
        )
        if ask_body is None:
            return 1

        if not args.skip_stream:
            # -----------------------------------------------------------------
            # WHAT: One streaming question (partial body read).
            # WHY: Validates SSE framing and tool/token events end-to-end.
            # -----------------------------------------------------------------
            ask_streaming_sample(
                client,
                base_api,
                token,
                dataset_id,
                conv_id,
                "What is the mean of the first numeric column?",
            )

        get_messages(client, base_api, headers, dataset_id, conv_id)

        if args.index:
            trigger_index(client, base_api, headers, dataset_id)

    print("[main] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
