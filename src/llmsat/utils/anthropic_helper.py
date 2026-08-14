import os
import time
import random
from typing import Optional

from llmsat.llmsat import setup_logging, get_logger
import logging

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

_DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"

_client = None


def _get_anthropic_client():
    """Initialize the Anthropic client from ANTHROPIC_API_KEY (cached)."""
    global _client
    if _client is not None:
        return _client
    try:
        from anthropic import Anthropic
    except Exception as exc:
        raise RuntimeError(
            "The 'anthropic' package is required for claude-* models "
            "(pip install anthropic)."
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; add it to .env to use claude-* models."
        )
    _client = Anthropic()
    return _client


def get_response_from_claude(
    prompt: str,
    system_message: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    max_retries: int = 5,
    initial_delay: float = 20.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 100.0,
) -> str:
    """Single-turn call to the Anthropic Messages API.

    Mirrors get_response_from_gemini's retry contract: exponential backoff
    with jitter on rate-limit/overload errors, returns the text output.
    """
    client = _get_anthropic_client()
    chosen_model = model or _DEFAULT_CLAUDE_MODEL

    delay = initial_delay
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=chosen_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system_message:
                kwargs["system"] = system_message
            resp = client.messages.create(**kwargs)
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "".join(parts)
        except Exception as exc:  # retry only transient failures
            last_exc = exc
            msg = str(exc).lower()
            transient = any(
                s in msg for s in ("rate limit", "overloaded", "529", "429", "500", "503")
            )
            if not transient or attempt == max_retries - 1:
                raise
            sleep_s = min(delay, max_backoff) * (1 + random.uniform(-0.2, 0.2))
            logger.warning(
                f"Anthropic transient error (attempt {attempt + 1}/{max_retries}): "
                f"{exc}; retrying in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)
            delay *= backoff_factor
    raise RuntimeError(f"Anthropic call failed after {max_retries} attempts") from last_exc
