"""Chat endpoints (REST and WebSocket)."""

import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.dependencies import get_faq_store, get_llm
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat.orchestrator import ChatOrchestrator
from app.services.faq.base import VectorStore
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm: LLMProvider = Depends(get_llm),
    vector_store: VectorStore = Depends(get_faq_store),
    db: AsyncSession = Depends(get_db),
):
    orchestrator = ChatOrchestrator(llm=llm, vector_store=vector_store, db=db)
    return await orchestrator.handle_message(request)


@router.websocket("/ws")
async def chat_websocket(
    websocket: WebSocket,
    llm: LLMProvider = Depends(get_llm),
    vector_store: VectorStore = Depends(get_faq_store),
    db: AsyncSession = Depends(get_db),
):
    await websocket.accept()
    session_id = None

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                request = ChatRequest(
                    message=payload.get("message", ""),
                    session_id=session_id or payload.get("session_id"),
                    channel=payload.get("channel", "websocket"),
                )
                orchestrator = ChatOrchestrator(
                    llm=llm, vector_store=vector_store, db=db
                )
                response = await orchestrator.handle_message(request)
                session_id = response.session_id

                await websocket.send_text(
                    json.dumps({
                        "message": response.message,
                        "session_id": response.session_id,
                        "intent": response.intent.value if response.intent else None,
                        "metadata": response.metadata,
                    })
                )
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"error": "Invalid JSON"})
                )
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
