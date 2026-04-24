"""
evaluation/metrics.py — Inter-rater reliability and agreement metrics.
"""

import math
import logging
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

SCORE_METRICS = ["correctness", "code_quality", "efficiency"]


def compute_cohens_kappa(
    ratings_a: list[int],
    ratings_b: list[int],
    labels: list[int] | None = None,
) -> float:
    """Compute Cohen's Kappa between two sets of ratings."""
    if len(ratings_a) != len(ratings_b):
        raise ValueError(
            f"Rating lists must be same length: {len(ratings_a)} vs {len(ratings_b)}"
        )
    if len(ratings_a) == 0:
        raise ValueError("Rating lists must not be empty")

    if labels is None:
        labels = [1, 2, 3, 4, 5]

    kappa = cohen_kappa_score(ratings_a, ratings_b, labels=labels)

    if math.isnan(kappa):
        return 1.0

    return kappa


def compute_pairwise_agreement(
    evaluations: list[dict],
) -> dict:
    """Compute Cohen's Kappa for every pair of models on every metric."""
    all_models: set[str] = set()
    for ev in evaluations:
        all_models.update(ev["results"].keys())

    models = sorted(all_models)

    if len(models) < 2:
        return {
            "pairs": [],
            "overall_mean_kappa": None,
            "n_evaluations": len(evaluations),
            "models": models,
            "error": "Need at least 2 models to compute agreement",
        }

    pairs_output = []
    all_kappas = []

    for model_a, model_b in combinations(models, 2):
        shared_evals = [
            ev for ev in evaluations
            if model_a in ev["results"] and model_b in ev["results"]
        ]

        if len(shared_evals) < 2:
            pairs_output.append({
                "model_a": model_a,
                "model_b": model_b,
                "metrics": {},
                "mean_kappa": None,
                "n_samples": len(shared_evals),
                "error": f"Need ≥2 shared evaluations, found {len(shared_evals)}",
            })
            continue

        metric_results = {}
        pair_kappas = []

        for metric in SCORE_METRICS:
            ratings_a = [ev["results"][model_a][metric] for ev in shared_evals]
            ratings_b = [ev["results"][model_b][metric] for ev in shared_evals]

            kappa = compute_cohens_kappa(ratings_a, ratings_b)
            interpretation = interpret_kappa(kappa)

            metric_results[metric] = {
                "kappa": round(kappa, 4),
                "interpretation": interpretation,
            }
            pair_kappas.append(kappa)

        mean_kappa = sum(pair_kappas) / len(pair_kappas)
        all_kappas.append(mean_kappa)

        pairs_output.append({
            "model_a": model_a,
            "model_b": model_b,
            "metrics": metric_results,
            "mean_kappa": round(mean_kappa, 4),
            "n_samples": len(shared_evals),
        })

        logger.info(
            "Kappa | %s vs %s | mean=%.4f | n=%d",
            model_a.split("/")[-1], model_b.split("/")[-1],
            mean_kappa, len(shared_evals),
        )

    overall = round(sum(all_kappas) / len(all_kappas), 4) if all_kappas else None

    return {
        "pairs": pairs_output,
        "overall_mean_kappa": overall,
        "n_evaluations": len(evaluations),
        "models": models,
    }


def interpret_kappa(kappa: float) -> str:
    """Interpret a Cohen's Kappa value using the Landis & Koch (1977) scale."""
    if kappa < 0:
        return "poor (less than chance)"
    elif kappa < 0.21:
        return "slight"
    elif kappa < 0.41:
        return "fair"
    elif kappa < 0.61:
        return "moderate"
    elif kappa < 0.81:
        return "substantial"
    else:
        return "almost perfect"
