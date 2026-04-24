"""
evaluation/judge.py — LLM Judge logic.

Contains the core judging functions that:
1. Build structured evaluation prompts
2. Call an LLM via OpenRouter
3. Parse and validate JSON scores from the response
4. Retry on invalid output
"""

import json
import logging
import re
import time
import asyncio

from models.openrouter_client import OpenRouterClient
from evaluation.prompts import (
    CODE_JUDGE_SYSTEM_PROMPT,
    CODE_JUDGE_USER_PROMPT,
    GENERAL_JUDGE_SYSTEM_PROMPT,
    GENERAL_JUDGE_USER_PROMPT,
)
from config import DEFAULT_TEMPERATURE, JUDGE_MODELS

logger = logging.getLogger(__name__)

# Max attempts to get valid JSON from the LLM
MAX_PARSE_RETRIES = 2


class JudgeError(Exception):
    """Raised when judging fails after all retries."""
    pass


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from LLM output.

    Handles common failure modes:
    - LLM wraps JSON in markdown code fences
    - LLM adds text before/after the JSON
    - LLM returns valid JSON directly
    """
    # Try direct parse first (best case)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")


def _validate_code_scores(data: dict) -> dict:
    """
    Validate and normalize code evaluation scores.

    Ensures all required fields exist, scores are integers in [1, 5],
    and explanation is a non-empty string.
    """
    required_keys = ["correctness", "code_quality", "efficiency", "explanation"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    scores = {}
    for key in ["correctness", "code_quality", "efficiency"]:
        val = data[key]
        if isinstance(val, float):
            val = int(val)
        if not isinstance(val, int) or not 1 <= val <= 5:
            raise ValueError(f"'{key}' must be an integer 1-5, got {val!r}")
        scores[key] = val

    explanation = str(data["explanation"]).strip()
    if not explanation:
        raise ValueError("'explanation' must be a non-empty string")

    return {**scores, "explanation": explanation}


async def evaluate_code(
    problem: str,
    code: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Judge a code submission against a problem description.

    Calls the LLM, parses the JSON response, validates scores,
    and retries if the output is malformed.
    """
    client = OpenRouterClient()
    messages = [
        {"role": "system", "content": CODE_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CODE_JUDGE_USER_PROMPT.format(problem=problem, code=code),
        },
    ]

    last_error: Exception | None = None

    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        logger.info(
            "Judge evaluation | model=%s parse_attempt=%d/%d",
            model, attempt, MAX_PARSE_RETRIES,
        )

        start = time.perf_counter()
        result = await client.chat_completion(
            model=model, messages=messages, temperature=temperature
        )
        latency_ms = (time.perf_counter() - start) * 1000

        raw_content = result["content"]
        logger.debug("Judge raw response | model=%s content=%s", model, raw_content[:300])

        try:
            parsed = _extract_json(raw_content)
            validated = _validate_code_scores(parsed)

            logger.info(
                "Judge success | model=%s correctness=%d quality=%d efficiency=%d latency=%.0fms",
                model,
                validated["correctness"],
                validated["code_quality"],
                validated["efficiency"],
                latency_ms,
            )

            return {
                "model": model,
                **validated,
                "latency_ms": latency_ms,
            }

        except (ValueError, KeyError) as exc:
            logger.warning(
                "Judge parse error | model=%s attempt=%d error=%s",
                model, attempt, str(exc),
            )
            last_error = exc
            # On retry, add a correction message to the conversation
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was not valid JSON. "
                    "Please respond with ONLY a JSON object in this exact format:\n"
                    '{"correctness": <1-5>, "code_quality": <1-5>, "efficiency": <1-5>, '
                    '"explanation": "<string>"}'
                ),
            })

    raise JudgeError(
        f"Failed to get valid JSON from {model} after {MAX_PARSE_RETRIES} attempts: {last_error}"
    )


async def evaluate_general(
    prompt: str,
    response: str,
    model: str,
    criteria: str = "helpfulness, accuracy, coherence",
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """Judge a general LLM response (non-code)."""
    client = OpenRouterClient()
    messages = [
        {
            "role": "system",
            "content": GENERAL_JUDGE_SYSTEM_PROMPT.format(criteria=criteria),
        },
        {
            "role": "user",
            "content": GENERAL_JUDGE_USER_PROMPT.format(prompt=prompt, response=response),
        },
    ]

    last_error: Exception | None = None

    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        result = await client.chat_completion(
            model=model, messages=messages, temperature=temperature
        )
        raw_content = result["content"]

        try:
            parsed = _extract_json(raw_content)
            score = parsed.get("score")
            if not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"'score' must be int 1-5, got {score!r}")
            reasoning = str(parsed.get("reasoning", "")).strip()
            if not reasoning:
                raise ValueError("'reasoning' must be non-empty")

            return {
                "model": model,
                "score": score,
                "reasoning": reasoning,
                "latency_ms": result["latency_ms"],
            }

        except (ValueError, KeyError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": (
                    "Your response was not valid JSON. "
                    "Respond with ONLY: "
                    '{"score": <1-5>, "reasoning": "<string>"}'
                ),
            })

    raise JudgeError(
        f"Failed to get valid JSON from {model} after {MAX_PARSE_RETRIES} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Multi-Model Orchestration
# ---------------------------------------------------------------------------

def _aggregate_code_scores(results: list[dict]) -> dict:
    """Compute aggregate statistics across multiple judge results."""
    if not results:
        return {
            "mean_correctness": None,
            "mean_code_quality": None,
            "mean_efficiency": None,
            "num_judges": 0,
        }

    n = len(results)
    return {
        "mean_correctness": round(sum(r["correctness"] for r in results) / n, 2),
        "mean_code_quality": round(sum(r["code_quality"] for r in results) / n, 2),
        "mean_efficiency": round(sum(r["efficiency"] for r in results) / n, 2),
        "num_judges": n,
    }


async def evaluate_code_multi(
    problem: str,
    code: str,
    models: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Run code evaluation across multiple LLM judges concurrently.

    Handles partial failures — if some models fail, the rest still
    return results. Only raises if ALL models fail.
    """
    target_models = models or JUDGE_MODELS

    logger.info(
        "Multi-model evaluation | models=%s",
        [m.split("/")[-1] for m in target_models],
    )

    # Launch all judge calls concurrently
    tasks = [
        evaluate_code(
            problem=problem,
            code=code,
            model=model,
            temperature=temperature,
        )
        for model in target_models
    ]

    # return_exceptions=True prevents one failure from cancelling others
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    errors = []

    for model, outcome in zip(target_models, outcomes):
        if isinstance(outcome, Exception):
            short_name = model.split("/")[-1]
            logger.warning(
                "Multi-model partial failure | model=%s error=%s",
                short_name, str(outcome),
            )
            errors.append({
                "model": model,
                "error": str(outcome),
            })
        else:
            results.append(outcome)

    if not results:
        raise JudgeError(
            f"All {len(target_models)} judge models failed: "
            + "; ".join(e["error"][:100] for e in errors)
        )

    aggregate = _aggregate_code_scores(results)

    logger.info(
        "Multi-model complete | success=%d/%d | mean_correctness=%.1f quality=%.1f efficiency=%.1f",
        len(results),
        len(target_models),
        aggregate["mean_correctness"] or 0,
        aggregate["mean_code_quality"] or 0,
        aggregate["mean_efficiency"] or 0,
    )

    return {
        "results": results,
        "errors": errors,
        "aggregate": aggregate,
    }
