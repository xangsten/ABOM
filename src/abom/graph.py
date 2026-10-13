"""Сборка графа LangGraph."""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from abom.merge import fan_out, merge_node, route_after_merge
from abom.organs import organ_node
from abom.state import BodyState


def build_graph():
    """Собирает и компилирует граф симуляции.

    START -> fan_out(Send к органам) -> organ -> merge -> (fan_out | END)

    Ветка продолжения снова уходит через fan_out (Send), а не напрямую в
    `organ`, чтобы узел всегда получал `organ_name` в payload.
    """
    g = StateGraph(BodyState)
    g.add_node("organ", organ_node)
    g.add_node("merge", merge_node)

    g.add_conditional_edges(START, fan_out, ["organ"])
    g.add_edge("organ", "merge")
    g.add_conditional_edges("merge", route_after_merge, {"end": END})
    return g.compile()
