"""Тесты учёта токенов (без обращения к сети)."""
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from abom.usage import UsageTracker


def _result(msg: AIMessage) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_tracker_sums_usage_metadata():
    msg = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "input_token_details": {"cache_read": 30},
        },
    )
    tracker = UsageTracker()
    tracker.on_llm_end(_result(msg))
    tracker.on_llm_end(_result(msg))

    assert tracker.calls == 2
    assert tracker.input_tokens == 200
    assert tracker.output_tokens == 80
    assert tracker.cache_read == 60


def test_tracker_delta_and_summary():
    tracker = UsageTracker()
    before = tracker.snapshot()
    tracker.on_llm_end(_result(AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )))
    assert "calls=1" in tracker.delta(before)
    assert "input=7" in tracker.delta(before)
    assert "total=10" in tracker.summary()


def test_tracker_ignores_messages_without_usage():
    tracker = UsageTracker()
    tracker.on_llm_end(_result(AIMessage(content="no usage")))
    assert tracker.calls == 0
    assert tracker.input_tokens == 0
