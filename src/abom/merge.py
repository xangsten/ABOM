"""Слияние мнений органов и маршрутизация графа."""
from __future__ import annotations

from langgraph.types import Send

from abom.config import AUTHORITY, PHYSIO_LIMITS
from abom.organs import ORGANS
from abom.state import BodyState

# Порог сходимости по НОРМИРОВАННОЙ сумме |Δ| (доля от диапазона параметра).
# 0.05 ≈ суммарное движение меньше ~0.4% диапазона на каждый из 13 параметров.
CONVERGENCE_EPS = 0.05


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


def normalized_diff(prev: dict[str, float], curr: dict[str, float]) -> float:
    """Сумма |Δ| по параметрам, нормированная на диапазон PHYSIO_LIMITS.

    Каждое изменение измеряется в долях допустимого диапазона параметра,
    поэтому вклады mmHg, ммоль/л, литров и долей сатурации сопоставимы.
    """
    total = 0.0
    for k, (lo, hi) in PHYSIO_LIMITS.items():
        if k in prev and k in curr:
            span = hi - lo
            if span > 0:
                total += abs(curr[k] - prev[k]) / span
    return total


def should_continue(state: BodyState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    # Критерий сходимости (нормированный)
    if state["iteration"] >= 2:
        diff = normalized_diff(
            state["history"][-2]["body"], state["history"][-1]["body"]
        )
        print(f"[graph] normalized diff = {diff:.4f} (eps={CONVERGENCE_EPS})")
        if diff < CONVERGENCE_EPS:
            print(f"[graph] converged after {state['iteration']} iterations")
            return "end"
    return "continue"


def route_after_merge(state: BodyState):
    """Маршрут после merge: повторный fan-out к органам либо завершение.

    Возвращает список `Send` (LangGraph сам разошлёт их по узлам) либо "end".
    Благодаря этому цикл повторяет ТОТ ЖЕ fan-out, что и на старте, поэтому
    узел `organ` всегда получает payload с ключом `organ_name` (иначе —
    KeyError при прямом переходе merge -> organ с полным состоянием).
    """
    if should_continue(state) == "end":
        return "end"
    return fan_out(state)

