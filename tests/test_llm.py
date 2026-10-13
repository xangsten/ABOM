"""Тесты structured output (обход ограничений DeepSeek) без обращения к сети."""
import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from abom.llm import ainvoke_structured


class Opinion(BaseModel):
    value: float
    note: str


def test_fallback_parses_plain_json():
    # FakeListChatModel не умеет with_structured_output -> сработает фолбэк.
    llm = FakeListChatModel(responses=['{"value": 1.5, "note": "ok"}'])
    out = asyncio.run(
        ainvoke_structured(llm, Opinion, [HumanMessage(content="hi")])
    )
    assert out == Opinion(value=1.5, note="ok")


def test_fallback_strips_json_code_fence():
    llm = FakeListChatModel(
        responses=['```json\n{"value": 2.0, "note": "fenced"}\n```']
    )
    out = asyncio.run(
        ainvoke_structured(llm, Opinion, [HumanMessage(content="hi")])
    )
    assert out.value == 2.0
    assert out.note == "fenced"
