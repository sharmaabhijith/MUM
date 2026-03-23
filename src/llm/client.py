from __future__ import annotations

import json
import logging
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


class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        rate_limit_rpm: int = 500,
        cost_tracker: CostTracker | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_rpm = rate_limit_rpm
        self.cost_tracker = cost_tracker or CostTracker()
        self.client = openai.OpenAI(api_key=api_key)
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
    ) -> str:
        self._rate_limit_wait()
        temp = temperature if temperature is not None else self.temperature

        @retry(
            wait=wait_exponential(min=1, max=60),
            stop=stop_after_attempt(self.max_retries),
            retry=retry_if_exception_type(
                (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
            ),
            before_sleep=lambda rs: logger.warning(
                f"Retry {rs.attempt_number}/{self.max_retries} after {rs.outcome.exception()}"
            ),
        )
        def _call():
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            return self.client.chat.completions.create(**kwargs)

        response = _call()
        content = response.choices[0].message.content or ""

        # Track costs
        usage = response.usage
        if usage:
            self.cost_tracker.record_call(
                model=self.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                phase=phase,
            )

        return content

    def generate_json(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int = 4096,
        phase: str = "general",
    ) -> dict | list:
        content = self.generate(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            phase=phase,
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
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
    ) -> str:
        if response_format and response_format.get("type") == "json_object":
            return '{"mock": true}'
        return "Mock response."

    def generate_json(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int = 4096,
        phase: str = "general",
    ) -> dict | list:
        return {"mock": True}
