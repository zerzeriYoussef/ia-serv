"""
Conversation and ChatMessage models for per-dataset conversational Q&A.

Tables:
  conversations  – one conversation thread per dataset (and user)
  chat_messages  – individual turns (user + assistant) within a conversation
"""



from sqlalchemy import (

    BigInteger,

    Column,

    DateTime,

    Enum,

    ForeignKey,

    Integer,

    JSON,

    String,

    Text,

)

from sqlalchemy.sql import func

import enum



from app.core.database import Base





class MessageRole(str, enum.Enum):

    user = "user"

    assistant = "assistant"





class Conversation(Base):

    __tablename__ = "conversations"



    id = Column(Integer, primary_key=True, index=True)





    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)

    user_id = Column(String, nullable=True, index=True)





    title = Column(String, nullable=True)







    analysis_version = Column(String, nullable=True)





    created_at = Column(DateTime(timezone=True), server_default=func.now())

    updated_at = Column(DateTime(timezone=True), onupdate=func.now())



    def __repr__(self) -> str:

        return f"<Conversation {self.id} dataset={self.dataset_id}>"





class ChatMessage(Base):

    __tablename__ = "chat_messages"



    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(

        Integer, ForeignKey("conversations.id"), nullable=False, index=True

    )





    role = Column(Enum(MessageRole), nullable=False)

    content = Column(Text, nullable=False)





    intent = Column(String, nullable=True)

    tool_calls = Column(JSON, nullable=True)

    chunk_ids = Column(JSON, nullable=True)

    chart_spec = Column(JSON, nullable=True)

    latency_ms = Column(Integer, nullable=True)



    created_at = Column(DateTime(timezone=True), server_default=func.now())



    def __repr__(self) -> str:

        return f"<ChatMessage {self.id} role={self.role} conv={self.conversation_id}>"

