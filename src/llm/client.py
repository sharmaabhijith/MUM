from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.cost_tracker import CostTracker

logger = logging.getLogger("mum")

# Provider configurations: each maps to a base_url and env-var for the API key.
PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
    },
}

# Model → provider mapping.  Add new models here as needed.
# All models route through DeepInfra.
MODEL_PROVIDER: dict[str, str] = {
    "deepseek-ai/DeepSeek-V3.2": "deepinfra",
    "google/gemini-2.5-pro": "deepinfra",
}

# Thinking models that may include <think>...</think> tags in output.
THINKING_MODELS: set[str] = {
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "deepseek-ai/DeepSeek-R1",
}

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> blocks that thinking models emit."""
    return _THINK_TAG_RE.sub("", content).strip()


def _resolve_provider(model: str) -> str:
    """Return the provider name for a model, defaulting to 'deepinfra'."""
    return MODEL_PROVIDER.get(model, "deepinfra")


def _build_openai_client(
    model: str,
    api_key: str | None = None,
) -> openai.OpenAI:
    """Create an OpenAI-compatible client for the given model's provider."""
    provider = _resolve_provider(model)
    cfg = PROVIDER_CONFIG[provider]

    resolved_key = api_key or os.environ.get(cfg["api_key_env"], "")
    if not resolved_key:
        raise ValueError(
            f"No API key for provider '{provider}'. "
            f"Set the {cfg['api_key_env']} environment variable or pass api_key."
        )

    return openai.OpenAI(
        api_key=resolved_key,
        base_url=cfg["base_url"],
        timeout=openai.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
    )


class LLMClient:
    def __init__(
        self,
        model: str = "google/gemini-2.5-pro",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 10,
        rate_limit_rpm: int = 500,
        cost_tracker: CostTracker | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_rpm = rate_limit_rpm
        self.cost_tracker = cost_tracker or CostTracker()
        self.client = _build_openai_client(model, api_key)
        self._call_timestamps: deque[float] = deque()

    def _rate_limit_wait(self) -> None:
        now = time.time()
        # Remove timestamps older than 60 seconds
        while self._call_timestamps and self._call_timestamps[0] < now - 60:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= self.rate_limit_rpm:
            sleep_time = 60 - (now - self._call_timestamps[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._call_timestamps.append(time.time())

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
        max_tokens: int = 4096,
        phase: str = "general",
    ) -> tuple[str, str]:
        """Generate a completion. Returns (content, finish_reason)."""
        self._rate_limit_wait()
        temp = temperature if temperature is not None else self.temperature

        @retry(
            wait=wait_exponential(min=5, max=120),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception_type(
                (
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.APIStatusError,
                )
            ),
            before_sleep=lambda rs: logger.warning(
                f"Retry {rs.attempt_number}/{self.max_retries} after {rs.outcome.exception()}"
            ),
        )
        def _call():
            kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            resp = self.client.chat.completions.create(**kwargs)
            # DeepInfra may return an empty choices list when model is busy
            if not resp.choices:
                raise openai.APIStatusError(
                    message="Model returned empty response (model may be busy)",
                    response=resp,
                    body=None,
                )
            return resp

        response = _call()
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "stop"

        # Strip <think>...</think> blocks from thinking models
        if self.model in THINKING_MODELS and "<think>" in content:
            content = _strip_think_tags(content)

        if finish_reason == "length":
            logger.warning(
                f"Response truncated (hit max_tokens={max_tokens}) in phase={phase}"
            )

        # Track costs
        usage = response.usage
        if usage:
            self.cost_tracker.record_call(
                model=self.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                phase=phase,
            )

        return content, finish_reason

    @staticmethod
    def _repair_truncated_json(content: str) -> dict | list | None:
        """Attempt to repair JSON truncated by max_tokens.

        Handles the common case where a JSON array of objects is cut off
        mid-element, e.g. {"turns": [{"turn":1,...}, {"turn":2,...}, {"tur
        """
        # Strategy: find the last complete object in the array, close the array and braces
        # Try progressively stripping from the end to find valid JSON
        # First, try closing open brackets
        bracket_stack = []
        in_string = False
        escape_next = False

        for ch in content:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in '{[':
                bracket_stack.append(ch)
            elif ch in '}]':
                if bracket_stack:
                    bracket_stack.pop()

        # Find the last complete JSON object boundary
        # Look for the last '},' or '}]' which marks a complete object
        last_complete = -1
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(content):
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth >= 1:  # We closed an inner object (not the root)
                    last_complete = i

        if last_complete == -1:
            return None

        # Truncate to last complete object, then close remaining brackets
        truncated = content[:last_complete + 1]
        # Count remaining open brackets that need closing
        stack = []
        in_str = False
        esc = False
        for ch in truncated:
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if stack:
                    stack.pop()

        # Close remaining open brackets in reverse order
        closers = {'[': ']', '{': '}'}
        suffix = ''.join(closers[b] for b in reversed(stack))
        repaired = truncated + suffix

        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            return None

    def generate_json(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int = 4096,
        phase: str = "general",
    ) -> dict | list:
        # DeepInfra supports json_object response_format for all models
        response_format: dict | None = {"type": "json_object"}

        current_max_tokens = max_tokens
        max_retries_for_truncation = 3

        for attempt in range(max_retries_for_truncation):
            content, finish_reason = self.generate(
                messages=messages,
                temperature=temperature,
                response_format=response_format,
                max_tokens=current_max_tokens,
                phase=phase,
            )
            try:
                return json.loads(content, strict=False)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    try:
                        return json.loads(json_str, strict=False)
                    except json.JSONDecodeError:
                        pass
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                    try:
                        return json.loads(json_str, strict=False)
                    except json.JSONDecodeError:
                        pass

                # If truncated by max_tokens, retry with doubled limit
                if finish_reason == "length" and attempt < max_retries_for_truncation - 1:
                    new_max = current_max_tokens * 2
                    logger.warning(
                        f"Response truncated at max_tokens={current_max_tokens} in "
                        f"phase={phase}. Retrying with max_tokens={new_max} "
                        f"(attempt {attempt + 2}/{max_retries_for_truncation})"
                    )
                    current_max_tokens = new_max
                    continue

                # Final attempt: try repair as last resort
                if finish_reason == "length":
                    logger.warning(
                        f"Attempting to repair truncated JSON ({len(content)} chars) "
                        f"in phase={phase}"
                    )
                    repaired = self._repair_truncated_json(content)
                    if repaired is not None:
                        logger.info("Successfully repaired truncated JSON")
                        return repaired
                    logger.error("Failed to repair truncated JSON")

                raise


class MockLLMClient(LLMClient):
    """Mock client for dry runs and testing."""

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "mock")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_retries = 1
        self.rate_limit_rpm = 10000
        self.cost_tracker = kwargs.get("cost_tracker", CostTracker())
        self._call_timestamps: deque[float] = deque()

    def generate(
        self,
        messages: list[dict],
        temperature: float | None = None,
        response_format: dict | None = None,
        max_tokens: int = 4096,
        phase: str = "general",
    ) -> tuple[str, str]:
        if response_format and response_format.get("type") == "json_object":
            return '{"mock": true}', "stop"
        return "Mock response.", "stop"

    def generate_json(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int = 4096,
        phase: str = "general",
    ) -> dict | list:
        return {"mock": True}
