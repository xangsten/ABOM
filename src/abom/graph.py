"""Сборка графа LangGraph."""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from abom.merge import fan_out, merge_node, should_continue
from abom.organs import organ_node
from abom.state import BodyState


def build_graph():
    """Собирает и компилирует граф симуляции.

    START -> fan_out(organ xN) -> organ -> merge -> (organ | END)
    """
    g = StateGraph(BodyState)
    g.add_node("organ", organ_node)
    g.add_node("merge", merge_node)

    g.add_conditional_edges(START, fan_out, ["organ"])
    g.add_edge("organ", "merge")
    g.add_conditional_edges("merge", should_continue, {
        "continue": "organ",
        "end": END,
    })
    return g.compile()
