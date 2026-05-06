"""
Chat API routes — per-dataset conversational Q&A with SSE streaming.

Routes:
  POST   /datasets/{dataset_id}/conversations               → CreateConversationResponse
  GET    /datasets/{dataset_id}/conversations               → list[ConversationSchema]
  GET    /datasets/{dataset_id}/conversations/{id}/messages → list[MessageSchema]
  POST   /datasets/{dataset_id}/conversations/{id}/messages → AskResponse (non-streaming)
  GET    /datasets/{dataset_id}/conversations/{id}/messages/stream → SSE stream
  POST   /datasets/{dataset_id}/index                       → {chunks_indexed} (admin)
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import CurrentUser, require_auth
from app.api.v1.schemas.chat_schema import (
    AskRequest,
    AskResponse,
    ConversationSchema,
    CreateConversationRequest,
    CreateConversationResponse,
    MessageSchema,
)
from app.core.config import settings
from app.core.database import get_db
from app.models.conversation import MessageRole
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.dataset_repository import DatasetRepository
from app.services.chat.orchestrator import run_chat_turn
from app.services.rag.dataset_indexer import index_dataset

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_gemini() -> None:
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GEMINI_API_KEY is not configured",
        )


async def _get_conversation_or_404(
    db: AsyncSession, dataset_id: int, conversation_id: int
):
    conv = await ConversationRepository.get_by_id(db, conversation_id)
    if not conv or conv.dataset_id != dataset_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found for dataset {dataset_id}",
        )
    return conv


async def _get_dataset_or_404(db: AsyncSession, dataset_id: int):
    ds = await DatasetRepository.get_by_id(db, dataset_id)
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found",
        )
    return ds


def _analysis_version(analysis_row) -> str:
    if not analysis_row:
        return ""
    return ConversationRepository.make_analysis_version(
        analysis_row.id,
        analysis_row.updated_at or analysis_row.created_at,
    )


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/conversations
# ---------------------------------------------------------------------------

@router.post(
    "/datasets/{dataset_id}/conversations",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new conversation for a dataset",
    tags=["Chat"],
)
async def create_conversation(
    dataset_id: int,
    body: CreateConversationRequest = CreateConversationRequest(),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """Creates a new conversation thread scoped to a dataset and the authenticated user."""
    await _get_dataset_or_404(db, dataset_id)

    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)
    av = _analysis_version(analysis_row)

    conv = await ConversationRepository.create(
        db,
        dataset_id=dataset_id,
        user_id=current_user.user_id,
        title=body.title or "New conversation",
        analysis_version=av,
    )
    await db.commit()
    await db.refresh(conv)

    return CreateConversationResponse(
        conversation_id=conv.id,
        dataset_id=conv.dataset_id,
        title=conv.title or "",
        created_at=conv.created_at,
    )


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/conversations
# ---------------------------------------------------------------------------

@router.get(
    "/datasets/{dataset_id}/conversations",
    response_model=List[ConversationSchema],
    summary="List conversations for a dataset",
    tags=["Chat"],
)
async def list_conversations(
    dataset_id: int,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    await _get_dataset_or_404(db, dataset_id)
    conversations = await ConversationRepository.get_by_dataset(
        db,
        dataset_id,
        user_id=current_user.user_id,
        skip=skip,
        limit=limit,
    )
    return [ConversationSchema.model_validate(c) for c in conversations]


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/conversations/{conversation_id}/messages
# ---------------------------------------------------------------------------

@router.get(
    "/datasets/{dataset_id}/conversations/{conversation_id}/messages",
    response_model=List[MessageSchema],
    summary="Get chat history for a conversation",
    tags=["Chat"],
)
async def get_messages(
    dataset_id: int,
    conversation_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    await _get_conversation_or_404(db, dataset_id, conversation_id)
    messages = await ConversationRepository.get_messages(
        db, conversation_id, limit=limit
    )
    # Return chronological order (oldest first)
    return [MessageSchema.model_validate(m) for m in reversed(messages)] 
    #Transform each m into a MessageSchema object


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/conversations/{conversation_id}/messages
# (non-streaming, for testing / debug)
# ---------------------------------------------------------------------------

@router.post(
    "/datasets/{dataset_id}/conversations/{conversation_id}/messages",
    response_model=AskResponse,
    summary="Ask a question (non-streaming, debug mode)",
    tags=["Chat"],
)
async def ask_message(
    dataset_id: int,
    conversation_id: int,
    body: AskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    Non-streaming version — collects all SSE events internally and returns a
    single JSON response. Useful for testing and API clients that can't handle SSE.
    """
    _require_gemini()
    conv = await _get_conversation_or_404(db, dataset_id, conversation_id)
    
    # Persist user message
    user_msg = await ConversationRepository.add_message(
        db,
        conversation_id=conversation_id,
        role=MessageRole.user,
        content=body.message,
    )
    await db.commit()

    # Load history (excluding the message we just saved context-wise, we fetch before it or limit carefully)
    history_messages = await ConversationRepository.get_messages(
        db, conversation_id, limit=settings.CHAT_MAX_RECENT_TURNS * 2
    )
    # The get_messages returns newest first. 
    # We want chronological (oldest first) AND we MUST exclude the user message we just inserted
    # so the LLM doesn't see it twice.
    chronological_history = reversed([m for m in history_messages if m.id != user_msg.id])

    # Drain the generator
    result_out: dict[str, Any] = {}

    async for _ in run_chat_turn(
        db=db,
        dataset_id=dataset_id,
        conversation_id=conversation_id,
        user_message=body.message,
        history_messages=list(chronological_history),
        result_out=result_out,
    ):
        pass  # Consume all events

    full_answer = result_out.get("full_answer", "")

    # Persist assistant message
    assistant_msg = await ConversationRepository.add_message(
        db,
        conversation_id=conversation_id,
        role=MessageRole.assistant,
        content=full_answer,
        intent=result_out.get("intent"),
        tool_calls=result_out.get("tool_calls"),
        chunk_ids=result_out.get("chunk_ids"),
        latency_ms=result_out.get("latency_ms"),
    )
    await db.commit()

    return AskResponse(
        message_id=assistant_msg.id,
        conversation_id=conversation_id,
        answer=full_answer,
        intent=result_out.get("intent"),
        chunk_ids=result_out.get("chunk_ids") or [],
        chart_spec=result_out.get("chart_spec"),
        caveats=[],
        latency_ms=result_out.get("latency_ms", 0),
    )


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/conversations/{conversation_id}/messages/stream
# ---------------------------------------------------------------------------

@router.get(
    "/datasets/{dataset_id}/conversations/{conversation_id}/messages/stream",
    summary="Ask a question via SSE streaming",
    tags=["Chat"],
    response_class=StreamingResponse,
)
async def stream_ask(
    dataset_id: int,
    conversation_id: int,
    message: str = Query(..., min_length=1, max_length=4096, description="The user's question"),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    SSE streaming endpoint. Connect with `Accept: text/event-stream`.

    Events streamed in order:
    - `event: intent`      — orchestrator plan
    - `event: retrieval`   — chunk IDs used
    - `event: tool_start`  — before each tool call
    - `event: tool_result` — after each tool call (preview)
    - `event: chart`       — (optional) chart spec JSON
    - `event: token`       — narrative text deltas
    - `event: done`        — final metadata
    - `event: error`       — if something fails

    After the stream completes, the assistant message is persisted to the DB.
    """
    _require_gemini()
    conv = await _get_conversation_or_404(db, dataset_id, conversation_id)

    # Persist user message first
    user_msg = await ConversationRepository.add_message(
        db,
        conversation_id=conversation_id,
        role=MessageRole.user,
        content=message,
    )
    await db.commit()

    # Load recent history
    history_messages = await ConversationRepository.get_messages(
        db, conversation_id, limit=settings.CHAT_MAX_RECENT_TURNS * 2
    )
    # Filter out current user msg, reverse to chronological
    chronological_history = reversed([m for m in history_messages if m.id != user_msg.id])

    async def event_stream():
        collected_events = []
        result_out: dict[str, Any] = {}
        try:
            async for frame in run_chat_turn(
                db=db,
                dataset_id=dataset_id,
                conversation_id=conversation_id,
                user_message=message,
                history_messages=list(chronological_history),
                result_out=result_out,
            ):
                collected_events.append(frame)
                yield frame
        except Exception as exc:
            logger.error("SSE stream error: %s", exc)
            error_frame = f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            yield error_frame
            return

        # After stream completes: persist assistant message
        full_answer = result_out.get("full_answer", "")
        try:
            assistant_msg = await ConversationRepository.add_message(
                db,
                conversation_id=conversation_id,
                role=MessageRole.assistant,
                content=full_answer,
                intent=result_out.get("intent"),
                tool_calls=result_out.get("tool_calls"),
                chunk_ids=result_out.get("chunk_ids"),
                latency_ms=result_out.get("latency_ms"),
            )
            await db.commit()
            # Re-emit done event with the real message_id
            if full_answer:
                yield (
                    f"event: message_persisted\n"
                    f"data: {json.dumps({'message_id': assistant_msg.id})}\n\n"
                )
        except Exception as exc:
            logger.error("Failed to persist assistant message: %s", exc)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/index  (admin — rebuild Chroma index)
# ---------------------------------------------------------------------------

@router.post(
    "/datasets/{dataset_id}/index",
    summary="Rebuild the semantic search index for a dataset (admin)",
    tags=["Chat"],
)
async def rebuild_index(
    dataset_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(require_auth),
):
    """
    Triggers a Chroma re-index for the dataset. Runs in the background so
    the response is immediate. Returns the number of chunks scheduled.
    """
    _require_gemini()
    await _get_dataset_or_404(db, dataset_id)

    analysis_row = await AnalysisRepository.get_by_dataset(db, dataset_id)
    if not analysis_row:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No analysis found for dataset {dataset_id}. Run /analyze first.",
        )

    async def _do_index():
        try:
            n = await index_dataset(db, dataset_id)
            logger.info("Background index complete: dataset=%s chunks=%s", dataset_id, n)
        except Exception as exc:
            logger.error("Background index failed: dataset=%s error=%s", dataset_id, exc)

    background_tasks.add_task(_do_index)

    return {
        "status": "indexing_started",
        "dataset_id": dataset_id,
        "message": "Chroma index rebuild started in background.",
    }
