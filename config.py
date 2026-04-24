"""
Central configuration for the LLM-as-a-Judge system.

Loads settings from environment variables and provides defaults.
All modules import config values from here — single source of truth.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenRouter ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./llm_judge.db")

# --- Models available for judging ---
JUDGE_MODELS = [
    "meta-llama/llama-3-8b-instruct",
    "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mixtral-8x7b-instruct",
]

# --- Default evaluation settings ---
DEFAULT_TEMPERATURE = 0.0  # Deterministic judging
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60  # seconds

# --- SSL ---
# Set to False if behind a corporate proxy with custom CA certs
SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() in ("true", "1", "yes")


# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
