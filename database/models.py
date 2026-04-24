"""
database/models.py — SQLAlchemy ORM models.
"""

from sqlalchemy import Column, String, Integer, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

from database.db import Base


class Evaluation(Base):
    """A code evaluation request (one problem + code pair)."""
    __tablename__ = "evaluations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    problem = Column(Text, nullable=False)
    code = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    results = relationship("JudgeResult", back_populates="evaluation", cascade="all, delete-orphan")


class JudgeResult(Base):
    """A single judge model's scores for an evaluation."""
    __tablename__ = "judge_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    evaluation_id = Column(String, ForeignKey("evaluations.id"), nullable=False)
    model = Column(String, nullable=False)
    correctness = Column(Integer, nullable=False)
    code_quality = Column(Integer, nullable=False)
    efficiency = Column(Integer, nullable=False)
    explanation = Column(Text, nullable=False)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    evaluation = relationship("Evaluation", back_populates="results")
