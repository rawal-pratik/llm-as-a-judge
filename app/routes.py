"""
app/routes.py — API route definitions.

Endpoints:
- POST /evaluate        — Submit code for multi-model judging
- POST /evaluate-code   — Alias (backward compatible)
- GET  /results         — List past evaluations (paginated)
- GET  /results/{id}    — Get a single evaluation by ID
- GET  /evaluations     — Alias for /results
- GET  /evaluations/{id}— Alias for /results/{id}
- GET  /metrics/agreement — Cohen's Kappa inter-rater metrics
- GET  /metrics/bias    — Bias detection metrics
- POST /test-llm        — Smoke-test OpenRouter connectivity
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from models.openrouter_client import OpenRouterClient, OpenRouterError
from evaluation.judge import evaluate_code_multi, JudgeError
from app.schemas import (
    CodeEvaluationRequest,
    CodeEvaluationResponse,
    CodeJudgeResult,
    AggregateScores,
    EvaluationDetail,
    EvaluationListResponse,
)
from config import JUDGE_MODELS
from database.db import get_db
from database.crud import save_evaluation, get_evaluations, get_evaluation_by_id, count_evaluations, get_evaluations_for_metrics, get_judge_results_for_bias
from evaluation.judge import _aggregate_code_scores
from evaluation.metrics import compute_pairwise_agreement
from evaluation.bias import analyze_bias

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "LLM-as-a-Judge API"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_evaluation_detail(evaluation) -> EvaluationDetail:
    """Convert an ORM Evaluation object to a typed Pydantic response."""
    results = [
        CodeJudgeResult(
            model=r.model,
            correctness=r.correctness,
            code_quality=r.code_quality,
            efficiency=r.efficiency,
            explanation=r.explanation,
            latency_ms=r.latency_ms,
        )
        for r in evaluation.results
    ]
    agg = _aggregate_code_scores([
        {
            "correctness": r.correctness,
            "code_quality": r.code_quality,
            "efficiency": r.efficiency,
        }
        for r in evaluation.results
    ])
    return EvaluationDetail(
        evaluation_id=evaluation.id,
        problem=evaluation.problem,
        code=evaluation.code,
        created_at=evaluation.created_at.isoformat() if evaluation.created_at else None,
        results=results,
        aggregate=AggregateScores(**agg),
    )


# ---------------------------------------------------------------------------
# POST /evaluate  (canonical) + POST /evaluate-code (alias)
# ---------------------------------------------------------------------------

async def _do_evaluate(request: CodeEvaluationRequest, db: Session) -> CodeEvaluationResponse:
    """Core evaluation logic shared by /evaluate and /evaluate-code."""
    evaluation_id = str(uuid.uuid4())
    logger.info(
        "Evaluation request | id=%s problem_len=%d code_len=%d",
        evaluation_id[:8], len(request.problem), len(request.code),
    )

    try:
        multi_result = await evaluate_code_multi(
            problem=request.problem,
            code=request.code,
        )

        save_evaluation(
            db=db,
            evaluation_id=evaluation_id,
            problem=request.problem,
            code=request.code,
            results=multi_result["results"],
        )

        return CodeEvaluationResponse(
            evaluation_id=evaluation_id,
            problem=request.problem,
            code=request.code,
            results=multi_result["results"],
            errors=multi_result["errors"],
            aggregate=AggregateScores(**multi_result["aggregate"]),
        )

    except JudgeError as exc:
        logger.error("All judges failed | id=%s error=%s", evaluation_id[:8], str(exc))
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/evaluate", response_model=CodeEvaluationResponse)
async def evaluate(request: CodeEvaluationRequest, db: Session = Depends(get_db)):
    """Evaluate a code submission using multiple LLM judges concurrently."""
    return await _do_evaluate(request, db)


@router.post("/evaluate-code", response_model=CodeEvaluationResponse)
async def evaluate_code_endpoint(request: CodeEvaluationRequest, db: Session = Depends(get_db)):
    """Alias for /evaluate (backward compatible)."""
    return await _do_evaluate(request, db)


# ---------------------------------------------------------------------------
# GET /results  (canonical) + GET /evaluations (alias)
# ---------------------------------------------------------------------------

async def _do_list_results(
    limit: int,
    offset: int,
    db: Session,
) -> EvaluationListResponse:
    """Core list logic shared by /results and /evaluations."""
    evaluations = get_evaluations(db, limit=limit, offset=offset)
    total = count_evaluations(db)

    return EvaluationListResponse(
        total=total,
        limit=limit,
        offset=offset,
        evaluations=[_build_evaluation_detail(e) for e in evaluations],
    )


@router.get("/results", response_model=EvaluationListResponse)
async def list_results(
    limit: int = Query(default=50, ge=1, le=200, description="Max results per page"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db),
):
    """List past evaluations with their judge results and aggregate scores."""
    return await _do_list_results(limit, offset, db)


@router.get("/evaluations", response_model=EvaluationListResponse)
async def list_evaluations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Alias for /results (backward compatible)."""
    return await _do_list_results(limit, offset, db)


# ---------------------------------------------------------------------------
# GET /results/{id}  (canonical) + GET /evaluations/{id} (alias)
# ---------------------------------------------------------------------------

async def _do_get_result(evaluation_id: str, db: Session) -> EvaluationDetail:
    """Core detail logic shared by /results/{id} and /evaluations/{id}."""
    evaluation = get_evaluation_by_id(db, evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _build_evaluation_detail(evaluation)


@router.get("/results/{evaluation_id}", response_model=EvaluationDetail)
async def get_result(evaluation_id: str, db: Session = Depends(get_db)):
    """Retrieve a single evaluation by ID."""
    return await _do_get_result(evaluation_id, db)


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationDetail)
async def get_single_evaluation(evaluation_id: str, db: Session = Depends(get_db)):
    """Alias for /results/{id} (backward compatible)."""
    return await _do_get_result(evaluation_id, db)


# ---------------------------------------------------------------------------
# Utilities & Metrics
# ---------------------------------------------------------------------------

@router.post("/test-llm")
async def test_llm(prompt: str = "Say hello in one sentence.", model: str | None = None):
    """Smoke-test endpoint to verify OpenRouter connectivity."""
    target_model = model or JUDGE_MODELS[0]
    logger.info("Test LLM call | model=%s prompt=%s", target_model, prompt[:80])

    try:
        client = OpenRouterClient()
        result = await client.chat_completion(
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "model": result["model"],
            "content": result["content"],
            "latency_ms": round(result["latency_ms"], 1),
            "usage": result["usage"],
        }
    except OpenRouterError as exc:
        logger.error("Test LLM failed | %s", exc)
        raise HTTPException(status_code=exc.status_code or 502, detail=exc.detail)


@router.get("/metrics/agreement")
async def get_agreement_metrics(db: Session = Depends(get_db)):
    """Compute Cohen's Kappa agreement between all judge model pairs."""
    evaluations = get_evaluations_for_metrics(db)

    if len(evaluations) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need ≥2 evaluations for agreement metrics, found {len(evaluations)}",
        )

    result = compute_pairwise_agreement(evaluations)
    return result


@router.get("/metrics/bias")
async def get_bias_metrics(db: Session = Depends(get_db)):
    """Detect systematic scoring bias across judge models."""
    judge_results = get_judge_results_for_bias(db)

    if len(judge_results) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need ≥2 judge results for bias analysis, found {len(judge_results)}",
        )

    return analyze_bias(judge_results)
