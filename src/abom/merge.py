"""Слияние мнений органов и маршрутизация графа."""
from __future__ import annotations

from langgraph.types import Send

from abom.config import AUTHORITY, PHYSIO_LIMITS
from abom.organs import ORGANS
from abom.state import BodyState

# Порог сходимости по сумме |Δ| между двумя последними снимками.
CONVERGENCE_EPS = 0.5


def fan_out(state: BodyState) -> list[Send]:
    """Параллельно отправляет состояние всем органам."""
    return [
        Send("organ", {"organ_name": name, "body": state["body"]})
        for name in ORGANS
    ]


async def merge_node(state: BodyState) -> dict:
    """Сводит мнения органов взвешенно (authority × confidence)."""
    body = dict(state["body"])
    opinions = state["opinions"]

    # Собираем предложения по параметрам
    proposals: dict[str, list[tuple[str, float, float]]] = {}
    for entry in opinions:
        organ = entry["organ"]
        op = entry["opinion"]
        for param, value in op.proposed_params.items():
            proposals.setdefault(param, []).append((organ, value, op.confidence))

    # Взвешенное усреднение по authority * confidence
    for param, items in proposals.items():
        total_w = 0.0
        weighted = 0.0
        for organ, value, conf in items:
            auth = AUTHORITY.get(param, {}).get(organ, 0.0)
            w = auth * conf
            if w <= 0:
                continue
            weighted += w * value
            total_w += w
        if total_w > 0:
            body[param] = weighted / total_w

    # Клиппинг
    for param, (lo, hi) in PHYSIO_LIMITS.items():
        if param in body:
            body[param] = max(lo, min(hi, body[param]))

    # Пересчёт производных
    body["cardiac_output"] = (body["heart_rate"] * body["stroke_volume"]) / 1000.0

    print(f"[merge] iter {state['iteration']} -> "
          f"{ {k: round(v, 2) for k, v in body.items()} }")

    return {
        "body": body,
        "iteration": state["iteration"] + 1,
        "opinions": [],  # reset
        "history": state["history"] + [
            {"iter": state["iteration"] + 1, "body": dict(body)}
        ],
    }


def should_continue(state: BodyState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    # Критерий сходимости
    if state["iteration"] >= 2:
        prev = state["history"][-2]["body"]
        curr = state["history"][-1]["body"]
        diff = sum(
            abs(curr[k] - prev[k]) for k in curr if isinstance(curr[k], (int, float))
        )
        if diff < CONVERGENCE_EPS:
            print(f"[graph] converged after {state['iteration']} iterations")
            return "end"
    return "continue"
