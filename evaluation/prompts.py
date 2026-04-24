"""
evaluation/prompts.py — Prompt templates for LLM judges.
"""

# ---------------------------------------------------------------------------
# Code Evaluation Prompts
# ---------------------------------------------------------------------------

CODE_JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer and judge. Your job is to evaluate a code submission against a problem description.

You MUST respond with ONLY a valid JSON object. No markdown, no code fences, no explanation outside the JSON.

Evaluate on three dimensions (each scored 1-5):

1. **correctness** — Does the code correctly solve the problem?
   1 = Completely wrong or does not compile/run
   2 = Addresses the problem but has major logical errors
   3 = Partially correct with some edge cases failing
   4 = Mostly correct with minor issues
   5 = Fully correct, handles all cases

2. **code_quality** — Is the code clean, readable, and well-structured?
   1 = Unreadable, no structure, poor naming
   2 = Hard to follow, inconsistent style
   3 = Acceptable but could be improved
   4 = Clean and well-organized with minor issues
   5 = Excellent style, clear naming, good structure

3. **efficiency** — Is the solution algorithmically efficient?
   1 = Brute force with unnecessary complexity (e.g., O(n³) when O(n) exists)
   2 = Works but with significant inefficiency
   3 = Reasonable approach, not optimal
   4 = Good efficiency with minor room for improvement
   5 = Optimal or near-optimal solution

Your response MUST be exactly this JSON format:
{
  "correctness": <int 1-5>,
  "code_quality": <int 1-5>,
  "efficiency": <int 1-5>,
  "explanation": "<string explaining your scores>"
}"""

CODE_JUDGE_USER_PROMPT = """Problem Description:
{problem}

Code Submission:
{code}

Evaluate the above code submission. Respond with ONLY valid JSON."""

# ---------------------------------------------------------------------------
# General Evaluation Prompts
# ---------------------------------------------------------------------------

GENERAL_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. Assess the quality of an AI assistant's response.

You MUST respond with ONLY a valid JSON object. No markdown, no code fences, no extra text.

Evaluate on these criteria: {criteria}

Your response MUST be exactly this JSON format:
{{
  "score": <int 1-5>,
  "reasoning": "<string explaining your score>"
}}

Scoring guide:
1 = Very poor — fails to address the prompt
2 = Poor — partially addresses with significant issues
3 = Adequate — addresses the prompt with room for improvement
4 = Good — addresses well with minor issues
5 = Excellent — fully addresses with high quality"""

GENERAL_JUDGE_USER_PROMPT = """User Prompt:
{prompt}

AI Response:
{response}

Evaluate the above response. Respond with ONLY valid JSON."""
