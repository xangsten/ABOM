"""Тесты слияния мнений и маршрутизации графа (без обращения к LLM)."""
import asyncio

import pytest

from abom.config import INITIAL_BODY, PHYSIO_LIMITS
from abom.merge import merge_node, normalized_diff, should_continue
from abom.organs import OrganOpinion


def _opinion(params: dict, conf: float = 1.0) -> OrganOpinion:
    return OrganOpinion(proposed_params=params, confidence=conf, reasoning="t")


def _state(opinions: list, iteration: int = 1) -> dict:
    return {
        "body": dict(INITIAL_BODY),
        "iteration": iteration,
        "max_iterations": 8,
        "opinions": opinions,
        "history": [
            {"iter": 0, "body": dict(INITIAL_BODY)},
            {"iter": 1, "body": dict(INITIAL_BODY)},
        ],
    }


def test_weighted_merge_by_authority():
    state = _state([
        {"organ": "heart", "opinion": _opinion(
            {"heart_rate": 100.0, "blood_pressure": 100.0})},
        {"organ": "kidney", "opinion": _opinion(
            {"blood_pressure": 150.0, "sodium": 140.0})},
    ])
    body = asyncio.run(merge_node(state))["body"]

    assert body["heart_rate"] == pytest.approx(100.0)   # только heart (1.0)
    assert body["sodium"] == pytest.approx(140.0)       # только kidney (1.0)
    # BP: heart 0.4, kidney 0.6 -> (0.4*100 + 0.6*150) / 1.0 = 130
    assert body["blood_pressure"] == pytest.approx(130.0)
    # cardiac_output — производная от итоговых HR*SV
    assert body["cardiac_output"] == pytest.approx(100.0 * 70.0 / 1000.0)


def test_confidence_scales_weight():
    state = _state([
        {"organ": "heart", "opinion": _opinion({"blood_pressure": 100.0}, conf=0.5)},
        {"organ": "kidney", "opinion": _opinion({"blood_pressure": 150.0}, conf=1.0)},
    ])
    body = asyncio.run(merge_node(state))["body"]
    # w_heart = 0.4*0.5 = 0.2 ; w_kidney = 0.6*1.0 = 0.6
    # (0.2*100 + 0.6*150) / 0.8 = 137.5
    assert body["blood_pressure"] == pytest.approx(137.5)


def test_clipping_to_limits():
    state = _state([
        {"organ": "heart", "opinion": _opinion({"heart_rate": 999.0})},
    ])
    body = asyncio.run(merge_node(state))["body"]
    assert body["heart_rate"] == PHYSIO_LIMITS["heart_rate"][1]


def test_opinions_reset_and_history_appended():
    state = _state([
        {"organ": "heart", "opinion": _opinion({"heart_rate": 80.0})},
    ])
    res = asyncio.run(merge_node(state))
    assert res["opinions"] == []
    assert res["iteration"] == 2
    assert len(res["history"]) == 3
    assert res["history"][-1]["iter"] == 2


def test_should_continue_stops_at_max_iterations():
    assert should_continue(
        {"iteration": 8, "max_iterations": 8, "history": []}
    ) == "end"


def test_should_continue_loops_when_changing():
    assert should_continue(
        {"iteration": 1, "max_iterations": 8, "history": []}
    ) == "continue"


def test_should_continue_converges_on_small_diff():
    body = dict(INITIAL_BODY)
    state = {
        "iteration": 2, "max_iterations": 8,
        "history": [{"iter": 1, "body": body}, {"iter": 2, "body": dict(body)}],
    }
    assert should_continue(state) == "end"


def test_should_continue_not_converged_on_big_diff():
    a = dict(INITIAL_BODY)
    b = dict(INITIAL_BODY)
    b["blood_pressure"] += 10.0
    state = {
        "iteration": 2, "max_iterations": 8,
        "history": [{"iter": 1, "body": a}, {"iter": 2, "body": b}],
    }
    assert should_continue(state) == "continue"


def test_normalized_diff_scales_by_param_range():
    # +16 мм рт.ст. по BP при диапазоне 160 -> вклад ровно 0.1
    a = dict(INITIAL_BODY)
    b = dict(INITIAL_BODY)
    b["blood_pressure"] += 16.0
    assert normalized_diff(a, b) == pytest.approx(0.1)


def test_normalized_diff_small_params_comparable_to_big():
    # Мелкомасштабный O2: 0.05 из диапазона 0.5 -> вклад 0.1 (как 16 по BP)
    a = dict(INITIAL_BODY)
    b = dict(INITIAL_BODY)
    b["oxygen_saturation"] += 0.05
    assert normalized_diff(a, b) == pytest.approx(0.1)


def test_normalized_diff_ignores_unknown_params():
    a = {"heart_rate": 60.0}
    b = {"heart_rate": 60.0, "unknown": 1e9}
    assert normalized_diff(a, b) == 0.0

