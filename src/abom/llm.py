"""Фабрика LLM (DeepSeek). Единое место для модели и её параметров."""
from __future__ import annotations

from langchain_deepseek import ChatDeepSeek

# Замени на свою модель при необходимости (например, deepseek-v4.1-flash).
MODEL = "deepseek-flash"


def make_llm() -> ChatDeepSeek:
    """Создаёт новый клиент ChatDeepSeek.

    Ключ берётся из переменной окружения DEEPSEEK_API_KEY
    (её подхватывает python-dotenv из .env в точке входа).
    """
    return ChatDeepSeek(model=MODEL, temperature=0.0, max_tokens=1024)
