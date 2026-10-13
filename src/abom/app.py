"""Точка входа симуляции: проверка ключа, прогон графа, вывод результатов."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from abom.config import INITIAL_BODY, MAX_ITERATIONS, RECURSION_LIMIT
from abom.graph import build_graph
from abom.state import BodyState
from abom.usage import TOTAL


def _require_api_key() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY не задан. Скопируй .env.example в .env и вставь ключ, "
            "либо экспортируй переменную окружения DEEPSEEK_API_KEY."
        )


def _print_result(result: dict) -> None:
    print("\n===== FINAL BODY =====")
    for k, v in sorted(result["body"].items()):
        print(f"  {k}: {v:.3f}")

    print("\n===== TRAJECTORY =====")
    for snap in result["history"]:
        b = snap["body"]
        print(f"  iter {snap['iter']}: HR={b['heart_rate']:.1f}, "
              f"BP={b['blood_pressure']:.1f}, "
              f"vol={b['blood_volume']:.2f}, "
              f"Na={b['sodium']:.1f}, GFR={b['gfr']:.1f}, "
              f"UO={b['urine_output']:.2f}")

    print(f"\n===== TOKENS =====\n  {TOTAL.summary()}")


async def main() -> None:
    # Подхватываем .env (DEEPSEEK_API_KEY и т.п.). Ключ в .env не коммитится.
    load_dotenv()
    _require_api_key()

    app = build_graph()
    initial: BodyState = {
        "body": dict(INITIAL_BODY),
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "opinions": [],
        "history": [{"iter": 0, "body": dict(INITIAL_BODY)}],
    }
    result = await app.ainvoke(initial, {"recursion_limit": RECURSION_LIMIT})
    _print_result(result)
