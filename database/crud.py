"""
database/crud.py — Create, Read operations for evaluations and judge results.
"""

import logging
from sqlalchemy.orm import Session

from database.models import Evaluation, JudgeResult

logger = logging.getLogger(__name__)


def save_evaluation(db: Session, evaluation_id: str, problem: str, code: str, results: list[dict]) -> Evaluation:
    """Persist a complete evaluation (input + all judge results) in one transaction."""
    evaluation = Evaluation(
        id=evaluation_id,
        problem=problem,
        code=code,
    )

    for r in results:
        judge_result = JudgeResult(
            evaluation_id=evaluation_id,
            model=r["model"],
            correctness=r["correctness"],
            code_quality=r["code_quality"],
            efficiency=r["efficiency"],
            explanation=r["explanation"],
            latency_ms=r.get("latency_ms"),
        )
        evaluation.results.append(judge_result)

    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    logger.info(
        "Saved evaluation | id=%s judges=%d",
        evaluation_id[:8], len(results),
    )
    return evaluation


def get_evaluations(db: Session, limit: int = 50, offset: int = 0) -> list[Evaluation]:
    """Retrieve evaluations ordered by newest first."""
    return (
        db.query(Evaluation)
        .order_by(Evaluation.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_evaluation_by_id(db: Session, evaluation_id: str) -> Evaluation | None:
    """Retrieve a single evaluation by ID."""
    return db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()


def get_all_judge_results(db: Session) -> list[JudgeResult]:
    """Retrieve all judge results (for metrics computation)."""
    return db.query(JudgeResult).order_by(JudgeResult.created_at.desc()).all()


def count_evaluations(db: Session) -> int:
    """Return total number of evaluations."""
    return db.query(Evaluation).count()


def get_evaluations_for_metrics(db: Session) -> list[dict]:
    """Get all evaluations with results structured for metrics computation."""
    evaluations = db.query(Evaluation).all()
    output = []
    for ev in evaluations:
        results_by_model = {}
        for r in ev.results:
            results_by_model[r.model] = {
                "correctness": r.correctness,
                "code_quality": r.code_quality,
                "efficiency": r.efficiency,
            }
        output.append({
            "evaluation_id": ev.id,
            "results": results_by_model,
        })
    return output


def get_judge_results_for_bias(db: Session) -> list[dict]:
    """Get all judge results as flat dicts for bias analysis."""
    results = db.query(JudgeResult).all()
    return [
        {
            "model": r.model,
            "evaluation_id": r.evaluation_id,
            "correctness": r.correctness,
            "code_quality": r.code_quality,
            "efficiency": r.efficiency,
        }
        for r in results
    ]
