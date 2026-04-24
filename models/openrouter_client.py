"""
models/openrouter_client.py — HTTP client for OpenRouter API.
"""

import asyncio
import logging
import time

import httpx

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_TEMPERATURE,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    SSL_VERIFY,
)

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Raised when the OpenRouter API returns an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"OpenRouter API error {status_code}: {detail}")


class OpenRouterClient:
    """Async HTTP client for OpenRouter's chat completion API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY in your .env file."
            )
        self.base_url = OPENROUTER_BASE_URL
        self.timeout = REQUEST_TIMEOUT
        self.max_retries = MAX_RETRIES

    def _build_headers(self) -> dict:
        """Construct request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://llm-judge.local",
            "X-Title": "LLM-as-a-Judge",
        }

    def _build_payload(
        self, model: str, messages: list[dict], temperature: float
    ) -> dict:
        """Construct the JSON payload for the chat completion request."""
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict:
        """
        Send a chat completion request to OpenRouter with retry logic.
        """
        headers = self._build_headers()
        payload = self._build_payload(model, messages, temperature)

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                logger.info(
                    "OpenRouter request | model=%s attempt=%d/%d",
                    model, attempt, self.max_retries,
                )

                async with httpx.AsyncClient(timeout=self.timeout, verify=SSL_VERIFY) as client:
                    response = await client.post(
                        self.base_url, headers=headers, json=payload
                    )

                latency_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code != 200:
                    error_detail = response.text[:500]
                    logger.warning(
                        "OpenRouter error | model=%s status=%d attempt=%d detail=%s",
                        model, response.status_code, attempt, error_detail,
                    )
                    last_error = OpenRouterError(response.status_code, error_detail)

                    # Retry on server errors and rate limits, not on client errors
                    if response.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** (attempt - 1))  # Exponential backoff
                        continue
                    raise last_error

                data = response.json()

                # Check for error in response body (OpenRouter sometimes returns 200 with error)
                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    logger.warning(
                        "OpenRouter body error | model=%s detail=%s",
                        model, error_msg,
                    )
                    raise OpenRouterError(200, error_msg)

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                logger.info(
                    "OpenRouter success | model=%s latency=%.0fms tokens_in=%s tokens_out=%s",
                    model,
                    latency_ms,
                    usage.get("prompt_tokens", "?"),
                    usage.get("completion_tokens", "?"),
                )

                return {
                    "raw": data,
                    "content": content,
                    "model": model,
                    "usage": usage,
                    "latency_ms": latency_ms,
                }

            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.warning(
                    "OpenRouter timeout | model=%s attempt=%d latency=%.0fms",
                    model, attempt, latency_ms,
                )
                last_error = OpenRouterError(408, f"Request timed out after {self.timeout}s")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue

            except httpx.ConnectError as exc:
                logger.warning(
                    "OpenRouter connection error | model=%s attempt=%d error=%s",
                    model, attempt, str(exc),
                )
                last_error = OpenRouterError(0, f"Connection failed: {exc}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue

        # All retries exhausted
        logger.error(
            "OpenRouter failed after %d attempts | model=%s", self.max_retries, model
        )
        raise last_error  # type: ignore[misc]

    async def list_models(self) -> list[dict]:
        """Fetch available models from OpenRouter."""
        logger.info("Fetching OpenRouter model list")
        async with httpx.AsyncClient(timeout=self.timeout, verify=SSL_VERIFY) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers=self._build_headers(),
            )
            response.raise_for_status()
            return response.json().get("data", [])
