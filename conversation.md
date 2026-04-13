# Conversational Analytics Implementation

Here is a summary of the new files that were created to build the Conversational Q&A system for the IA-service.

## 1. Database & Data Models
- **`app/models/conversation.py`**
  Defines the `Conversation` (stores thread metadata like the dataset it belongs to) and `ChatMessage` (stores actual user/assistant messages) SQLAlchemy models.
- **`app/repositories/conversation_repository.py`**
  Provides database operations (CRUD) for creating conversation threads and fetching chat message history.
- **`migrations/versions/52e516ec0852_add_chat_tables.py`**
  The Alembic Python migration script used to create the tables in PostgreSQL.

## 2. API & Schemas
- **`app/api/v1/schemas/chat_schema.py`**
  Contains all Pydantic models to strictly validate tool arguments, multi-agent intentions, system prompts, Server-Sent Event (SSE) structures, and standard API request/response paths.
- **`app/api/v1/routes/chat.py`** (all routes are under **`/api/v1`** and require `Authorization: Bearer <JWT>`)
  The API endpoints layer housing the main conversational interactions, including:
  - `POST /datasets/{dataset_id}/conversations` — create a thread scoped to that dataset
  - `GET /datasets/{dataset_id}/conversations` — list threads
  - `GET /datasets/{dataset_id}/conversations/{id}/messages` — history
  - `POST /datasets/{dataset_id}/conversations/{id}/messages` — ask (JSON, non-streaming)
  - `GET /datasets/{dataset_id}/conversations/{id}/messages/stream?message=...` — SSE streaming (use `Accept: text/event-stream`)
  - `POST /datasets/{dataset_id}/index` — rebuild Chroma index (needs analysis + `GEMINI_API_KEY`)

## 3. Agents & Orchestration
- **`app/services/chat/orchestrator.py`**
  The core reasoning brain of the service. Combines chat history, retrieves context, uses Gemini to plan intent/tools, and streams answers securely back to the user.
- **`app/services/chat/analysis_agent.py`**
  A dedicated safety boundary and execution agent. Traps logic from Gemini tool-calling, validates column names, strictly enforces allowed basic operations (`groupby`, `filter`, etc.), sets timeouts on execution, and passes back DataFrame representations.
- **`app/services/chat/viz_agent.py`**
  The deterministic visualization engine that maps tabular results from `analysis_agent` into valid `ChartSpec` JSON objects (without using slow/hallucination-prone LLM calls).

## 4. RAG Vectors & Context Retrieval
- **`app/services/rag/dataset_indexer.py`**
  Manages semantic chunking and embedding metadata/relationship definitions utilizing ChromaDB so the chat orchestrated can find relevant context based on natural language queries.
