"""
evaluation/bias.py — Bias detection for LLM judge models.
"""

import logging
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)

SCORE_METRICS = ["correctness", "code_quality", "efficiency"]
BIAS_THRESHOLD = 0.5  # flag if model deviates ≥0.5 from grand mean


def compute_model_bias(judge_results: list[dict]) -> dict:
    """Compute per-model scoring statistics and bias indicators."""
    if not judge_results:
        return {
            "models": {},
            "grand_means": {},
            "grand_overall_mean": 0.0,
            "bias_threshold": BIAS_THRESHOLD,
            "n_total_results": 0,
            "n_models": 0,
        }

    # Group results by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in judge_results:
        by_model[r["model"]].append(r)

    # Compute grand means (across all models)
    grand_means = {}
    all_scores_flat = []
    for metric in SCORE_METRICS:
        values = [r[metric] for r in judge_results]
        grand_means[metric] = statistics.mean(values)
        all_scores_flat.extend(values)
    grand_overall = statistics.mean(all_scores_flat)

    # Compute per-model stats
    models_output = {}
    for model_name, results in sorted(by_model.items()):
        mean_scores = {}
        deviations = {}
        bias_flags = []
        score_dist: dict[str, dict[int, int]] = {}

        model_all_scores = []
        for metric in SCORE_METRICS:
            values = [r[metric] for r in results]
            mean_val = statistics.mean(values)
            mean_scores[metric] = round(mean_val, 3)
            dev = round(mean_val - grand_means[metric], 3)
            deviations[metric] = dev
            model_all_scores.extend(values)

            if abs(dev) >= BIAS_THRESHOLD:
                bias_flags.append(metric)

            # Score frequency distribution
            dist = {s: 0 for s in range(1, 6)}
            for v in values:
                dist[v] = dist.get(v, 0) + 1
            score_dist[metric] = dist

        overall_mean = round(statistics.mean(model_all_scores), 3)
        overall_dev = round(overall_mean - grand_overall, 3)

        if overall_dev >= BIAS_THRESHOLD:
            direction = "lenient"
        elif overall_dev <= -BIAS_THRESHOLD:
            direction = "severe"
        else:
            direction = "neutral"

        models_output[model_name] = {
            "n_evaluations": len(results),
            "mean_scores": mean_scores,
            "overall_mean": overall_mean,
            "deviation_from_grand_mean": deviations,
            "overall_deviation": overall_dev,
            "bias_flags": bias_flags,
            "is_biased": len(bias_flags) > 0,
            "bias_direction": direction,
            "score_distribution": score_dist,
        }

    return {
        "models": models_output,
        "grand_means": {k: round(v, 3) for k, v in grand_means.items()},
        "grand_overall_mean": round(grand_overall, 3),
        "bias_threshold": BIAS_THRESHOLD,
        "n_total_results": len(judge_results),
        "n_models": len(models_output),
    }


def compute_pairwise_bias(judge_results: list[dict]) -> list[dict]:
    """For each pair of models, compute the mean score difference on shared evaluations."""
    # Index: evaluation_id → model → scores
    by_eval: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in judge_results:
        eid = r.get("evaluation_id", "")
        by_eval[eid][r["model"]] = {
            m: r[m] for m in SCORE_METRICS
        }

    all_models = sorted({r["model"] for r in judge_results})
    pairs = []

    for i, model_a in enumerate(all_models):
        for model_b in all_models[i + 1:]:
            shared_evals = [
                eid for eid, models in by_eval.items()
                if model_a in models and model_b in models
            ]

            if len(shared_evals) < 1:
                pairs.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_shared": 0,
                    "mean_diff": {m: 0.0 for m in SCORE_METRICS},
                    "overall_mean_diff": 0.0,
                    "direction": "insufficient data",
                })
                continue

            diffs = {m: [] for m in SCORE_METRICS}
            for eid in shared_evals:
                for metric in SCORE_METRICS:
                    diff = by_eval[eid][model_a][metric] - by_eval[eid][model_b][metric]
                    diffs[metric].append(diff)

            mean_diffs = {}
            all_diffs = []
            for metric in SCORE_METRICS:
                md = round(statistics.mean(diffs[metric]), 3)
                mean_diffs[metric] = md
                all_diffs.extend(diffs[metric])

            overall_diff = round(statistics.mean(all_diffs), 3)

            if overall_diff > 0.25:
                direction = f"{model_a.split('/')[-1]} scores higher"
            elif overall_diff < -0.25:
                direction = f"{model_b.split('/')[-1]} scores higher"
            else:
                direction = "similar"

            pairs.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_shared": len(shared_evals),
                "mean_diff": mean_diffs,
                "overall_mean_diff": overall_diff,
                "direction": direction,
            })

    return pairs


def analyze_bias(judge_results: list[dict]) -> dict:
    """Run full bias analysis: per-model stats + pairwise comparison."""
    model_bias = compute_model_bias(judge_results)
    pairwise = compute_pairwise_bias(judge_results)
    return {
        **model_bias,
        "pairwise_bias": pairwise,
    }
