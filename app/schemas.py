"""
app/schemas.py — Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Code Evaluation — Request
# ---------------------------------------------------------------------------

class CodeEvaluationRequest(BaseModel):
    """Payload for submitting a code evaluation."""
    problem: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The problem description / requirements.",
    )
    code: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The code submission to evaluate.",
    )


# ---------------------------------------------------------------------------
# Code Evaluation — Response
# ---------------------------------------------------------------------------

class CodeJudgeResult(BaseModel):
    """Single judge's code evaluation output."""
    model: str
    correctness: int = Field(..., ge=1, le=5)
    code_quality: int = Field(..., ge=1, le=5)
    efficiency: int = Field(..., ge=1, le=5)
    explanation: str
    latency_ms: float


class AggregateScores(BaseModel):
    """Mean scores across all successful judges."""
    mean_correctness: float
    mean_code_quality: float
    mean_efficiency: float
    num_judges: int


class CodeEvaluationResponse(BaseModel):
    """Full code evaluation response with all judge results."""
    evaluation_id: str
    problem: str
    code: str
    results: list[CodeJudgeResult]
    errors: list[dict] = Field(default_factory=list, description="Models that failed.")
    aggregate: AggregateScores


# ---------------------------------------------------------------------------
# Results — List / Detail responses
# ---------------------------------------------------------------------------

class EvaluationDetail(BaseModel):
    """A single stored evaluation with judge results and aggregates."""
    evaluation_id: str
    problem: str
    code: str
    created_at: Optional[str] = None
    results: list[CodeJudgeResult]
    aggregate: AggregateScores


class EvaluationListResponse(BaseModel):
    """Paginated list of stored evaluations."""
    total: int
    limit: int
    offset: int
    evaluations: list[EvaluationDetail]


# ---------------------------------------------------------------------------
# General Evaluation (backward compatible)
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    """Payload for submitting a general evaluation."""
    prompt: str = Field(..., description="The original prompt given to the LLM.")
    response: str = Field(..., description="The LLM-generated response to evaluate.")
    criteria: Optional[str] = Field(
        "helpfulness, accuracy, coherence",
        description="Comma-separated evaluation criteria.",
    )


class EvaluationResult(BaseModel):
    """Single judge's evaluation output."""
    model: str
    score: int = Field(..., ge=1, le=5)
    reasoning: str


class EvaluationResponse(BaseModel):
    """Full evaluation response returned to the client."""
    evaluation_id: str
    prompt: str
    response: str
    results: list[EvaluationResult]
