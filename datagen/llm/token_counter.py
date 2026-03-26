from __future__ import annotations

import tiktoken


class TokenCounter:
    def __init__(self, model: str = "deepseek-ai/DeepSeek-V3.2"):
        self.model = model
        # cl100k_base is a reasonable approximation for token counting across providers
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            total += 4  # message framing overhead
            total += self.count(msg.get("role", ""))
            total += self.count(msg.get("content", ""))
        total += 2  # reply priming
        return total

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])
