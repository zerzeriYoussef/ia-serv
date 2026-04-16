import asyncio
from app.services.chat.orchestrator import run_chat_turn
from app.schemas.chat import AskRequest

async def test():
    req = AskRequest(message="How has coffee consumption changed over the years?.")
    # Actually wait, we need an active DB session and a conversation_id
