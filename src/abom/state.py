"""Состояние графа LangGraph."""
from __future__ import annotations

from typing import Annotated

from typing_extensions import TypedDict


def opinions_reducer(existing: list, new: list) -> list:
    """Дописывает мнения, но позволяет merge_node сбросить через [].

    Параллельные ветки (heart/kidney) пишут в один канал `opinions`;
    reducer аккумулирует их, а merge_node возвращает [] для сброса между итерациями.
    """
    if new == []:
        return []
    return existing + new


class BodyState(TypedDict):
    body: dict[str, float]          # 13 параметров организма
    iteration: int
    max_iterations: int
    opinions: Annotated[list, opinions_reducer]
    history: list
