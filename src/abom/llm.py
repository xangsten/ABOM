"""Фабрика LLM (DeepSeek) и получение structured output.

Зачем отдельный хелпер: по умолчанию `with_structured_output()` работает через
`method="function_calling"`, а он выставляет принудительный `tool_choice`.
DeepSeek в thinking-режиме такой `tool_choice` отклоняет
(`400: Thinking mode does not support this tool_choice`).

Поэтому:
  1) пробуем `method="json_mode"` (`response_format=json_object`, без tool'ов);
  2) если провайдер/режим не поддерживает — обычный вызов + толерантный
     JSON-парсер (инструкции по схеме добавляются в обоих случаях).
"""
from __future__ import annotations

from typing import TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel

# Замени на свою модель при необходимости (например, deepseek-v4.1-flash).
MODEL = "deepseek-flash"

T = TypeVar("T", bound=BaseModel)


def make_llm() -> ChatDeepSeek:
    """Создаёт новый клиент ChatDeepSeek.

    Ключ берётся из переменной окружения DEEPSEEK_API_KEY
    (её подхватывает python-dotenv из .env в точке входа).
    """
    return ChatDeepSeek(model=MODEL, temperature=0.0, max_tokens=1024)


def as_text(content) -> str:
    """Ответ модели может быть строкой или списком контент-блоков."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


async def ainvoke_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    config: RunnableConfig | None = None,
) -> T:
    """Возвращает валидированный объект `schema`, обходя ограничения DeepSeek.

    См. описание модуля: принудительный tool_choice в thinking-режиме не
    поддерживается, поэтому используем json_mode с фолбэком на JSON-парсер.
    """
    parser = JsonOutputParser(pydantic_object=schema)
    # Инструкции по схеме ставим первым system-сообщением: провайдеры требуют
    # system в начале, а json_mode к тому же требует явного описания формата.
    msgs: list[BaseMessage] = [
        SystemMessage(content=parser.get_format_instructions()),
        *messages,
    ]

    try:
        structured = llm.with_structured_output(schema, method="json_mode")
        return cast(T, await structured.ainvoke(msgs, config))
    except Exception as exc:  # noqa: BLE001 — осознанный фолбэк
        print(f"[llm] json_mode недоступен ({exc.__class__.__name__}), "
              f"фолбэк на текстовый JSON")
        response = await llm.ainvoke(msgs, config)
        return schema.model_validate(parser.parse(as_text(response.content)))

