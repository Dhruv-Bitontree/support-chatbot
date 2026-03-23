"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, default=gen_uuid)
    customer_email = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="pending")
    items = Column(JSON, nullable=False, default=list)
    total = Column(Float, nullable=False, default=0.0)
    tracking_number = Column(String, nullable=True)
    estimated_delivery = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, nullable=True)
    category = Column(String, nullable=False, default="general")
    description = Column(Text, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String, nullable=True)
    priority = Column(String, nullable=False, default="medium")
    status = Column(String, nullable=False, default="open")
    assigned_to = Column(String, nullable=True)
    customer_email = Column(String, nullable=True)
    order_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=gen_uuid)
    channel = Column(String, nullable=False, default="web")
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")
