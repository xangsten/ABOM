"""Учёт токенов LLM по usage_metadata (дополнительных запросов не делает)."""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageTracker(BaseCallbackHandler):
    """Суммирует input/output токены по всем LLM-вызовам.

    Учитывает в том числе thinking-токены: DeepSeek возвращает их в
    `output_tokens`, а сам текст — в `additional_kwargs["reasoning_content"]`.
    `cache_read` — попадания в контекстный кэш DeepSeek (дешевле обычного input).
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                self.calls += 1
                self.input_tokens += usage.get("input_tokens") or 0
                self.output_tokens += usage.get("output_tokens") or 0
                details = usage.get("input_token_details") or {}
                self.cache_read += details.get("cache_read") or 0

    def snapshot(self) -> tuple[int, int, int]:
        """Снимок счётчиков (для вычисления вклада отдельного шага)."""
        return (self.input_tokens, self.output_tokens, self.calls)

    def delta(self, snap: tuple[int, int, int]) -> str:
        din = self.input_tokens - snap[0]
        dout = self.output_tokens - snap[1]
        dcalls = self.calls - snap[2]
        return (f"calls={dcalls}, input={din}, output={dout}, "
                f"total={din + dout}")

    def summary(self) -> str:
        return (f"calls={self.calls}, input={self.input_tokens}, "
                f"output={self.output_tokens}, "
                f"total={self.input_tokens + self.output_tokens}, "
                f"cache_read={self.cache_read}")


# Общий счётчик за весь прогон (organ_node работает в основном процессе).
TOTAL = UsageTracker()
