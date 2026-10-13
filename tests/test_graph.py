"""Проверка сборки и прогона графа (узлы органов замоканы, без LLM/MCP)."""
import asyncio

import pytest

import abom.graph as G
from abom.config import INITIAL_BODY
from abom.organs import OrganOpinion


def _initial(max_iterations: int) -> dict:
    return {
        "body": dict(INITIAL_BODY),
        "iteration": 0,
        "max_iterations": max_iterations,
        "opinions": [],
        "history": [{"iter": 0, "body": dict(INITIAL_BODY)}],
    }


def test_graph_compiles():
    assert hasattr(G.build_graph(), "ainvoke")


def test_graph_has_expected_nodes():
    nodes = set(G.build_graph().get_graph().nodes)
    assert {"organ", "merge"} <= nodes


def test_continue_branch_reroutes_via_send(monkeypatch):
    """Регресс: ветка продолжения должна снова идти через fan-out (Send).

    Раньше переход `merge -> organ` шёл напрямую и узел падал с
    KeyError: 'organ_name' (в state нет organ_name).
    """
    calls: list[str] = []

    async def fake_organ(state):
        calls.append(state["organ_name"])  # без fan-out тут был бы KeyError
        name = state["organ_name"]
        params = {"heart_rate": 70.0} if name == "heart" else {"sodium": 141.0}
        return {"opinions": [{"organ": name, "opinion": OrganOpinion(
            proposed_params=params, confidence=1.0, reasoning="stub")}]}

    monkeypatch.setattr(G, "organ_node", fake_organ)
    app = G.build_graph()
    result = asyncio.run(
        app.ainvoke(_initial(max_iterations=3), {"recursion_limit": 50})
    )

    assert set(calls) == {"heart", "kidney"}
    assert len(calls) >= 4  # 2 органа х минимум 2 итерации
    assert result["iteration"] >= 2
    assert result["body"]["heart_rate"] == pytest.approx(70.0)
    assert result["body"]["sodium"] == pytest.approx(141.0)

