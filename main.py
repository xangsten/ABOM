"""Минимальная система: сердце + почки, LangGraph + MCP + DeepSeek."""
import asyncio
import os
import sys
from typing import Annotated, cast

from dotenv import load_dotenv
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

HERE = os.path.dirname(os.path.abspath(__file__))

# Подхватываем .env (DEEPSEEK_API_KEY и т.п.). Ключ в .env не коммитится.
load_dotenv()


# ---------------- STATE ----------------
def opinions_reducer(existing: list, new: list) -> list:
    """Дописывает мнения, но позволяет merge_node сбросить через []."""
    if new == []:
        return []
    return existing + new


class BodyState(TypedDict):
    body: dict[str, float]          # 13 параметров организма
    iteration: int
    max_iterations: int
    opinions: Annotated[list, opinions_reducer]
    history: list


# ---------------- LIMITS & AUTHORITY ----------------
PHYSIO_LIMITS: dict[str, tuple[float, float]] = {
    "heart_rate":         (30.0, 200.0),
    "stroke_volume":      (20.0, 150.0),
    "contractility":      (0.2, 2.5),
    "blood_pressure":     (40.0, 200.0),
    "blood_volume":       (3.0, 8.0),
    "sodium":             (120.0, 160.0),
    "potassium":          (2.0, 7.0),
    "gfr":                (5.0, 180.0),
    "renin":              (0.0, 10.0),
    "aldosterone":        (0.0, 5.0),
    "oxygen_saturation":  (0.5, 1.0),
    "urine_output":       (0.0, 10.0),
    "cardiac_output":     (1.0, 15.0),
}

# Кто и с каким весом влияет на параметр. Пусто = производный.
AUTHORITY: dict[str, dict[str, float]] = {
    "heart_rate":         {"heart": 1.0},
    "stroke_volume":      {"heart": 1.0},
    "contractility":      {"heart": 1.0},
    "blood_pressure":     {"heart": 0.4, "kidney": 0.6},
    "blood_volume":       {"heart": 0.2, "kidney": 0.8},
    "sodium":             {"kidney": 1.0},
    "potassium":          {"kidney": 1.0},
    "gfr":                {"kidney": 1.0},
    "renin":              {"kidney": 1.0},
    "aldosterone":        {"kidney": 1.0},
    "oxygen_saturation":  {"heart": 0.7, "kidney": 0.3},
    "urine_output":       {"kidney": 1.0},
    "cardiac_output":     {},
}


# ---------------- ORGANS ----------------
HEART_PROMPT = """Ты — СЕРДЦЕ организма. Задача — поддерживать перфузию.

Ты управляешь только: heart_rate, stroke_volume, contractility,
blood_pressure, blood_volume, oxygen_saturation.

Workflow:
1. Посмотри текущее состояние (read-view).
2. Вызови свои MCP-инструменты (cardiac output, MAP, baroreflex HR).
3. Предложи целевые значения параметров, которыми управляешь.

Правила:
- ОБЯЗАТЕЛЬНО вызови хотя бы один инструмент до предложения значений.
- Предлагай ТОЛЬКО свои параметры.
- Не выдумывай числа — опирайся на результаты инструментов.
"""

KIDNEY_PROMPT = """Ты — ПОЧКИ организма. Задача — жидкость, электролиты, фильтрация.

Ты управляешь: blood_pressure, blood_volume, sodium, potassium, gfr,
renin, aldosterone, urine_output, oxygen_saturation.

Workflow:
1. Посмотри текущее состояние.
2. Вызови MCP-инструменты (GFR, sodium balance, urine output).
3. Предложи целевые значения своих параметров.

Правила:
- ОБЯЗАТЕЛЬНО вызови хотя бы один инструмент.
- Предлагай ТОЛЬКО свои параметры.
"""

ORGANS: dict[str, dict] = {
    "heart": {
        "reads": [
            "heart_rate", "stroke_volume", "contractility",
            "blood_pressure", "blood_volume", "oxygen_saturation",
            "cardiac_output",
        ],
        "writes": [
            "heart_rate", "stroke_volume", "contractility",
            "blood_pressure", "blood_volume", "oxygen_saturation",
        ],
        "prompt": HEART_PROMPT,
        "mcp": {
            "heart": {
                "command": sys.executable,
                "args": [os.path.join(HERE, "heart_mcp.py")],
                "transport": "stdio",
            }
        },
    },
    "kidney": {
        "reads": [
            "blood_pressure", "blood_volume", "sodium", "potassium",
            "gfr", "renin", "aldosterone", "urine_output", "oxygen_saturation",
        ],
        "writes": [
            "blood_pressure", "blood_volume", "sodium", "potassium",
            "gfr", "renin", "aldosterone", "urine_output", "oxygen_saturation",
        ],
        "prompt": KIDNEY_PROMPT,
        "mcp": {
            "kidney": {
                "command": sys.executable,
                "args": [os.path.join(HERE, "kidney_mcp.py")],
                "transport": "stdio",
            }
        },
    },
}


# ---------------- OPINION SCHEMA ----------------
class OrganOpinion(BaseModel):
    proposed_params: dict[str, float] = Field(
        description="Целевые значения. Только ключи из твоего write set."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="1–3 предложения.")


# ---------------- HELPERS ----------------
def format_params(params: dict[str, float]) -> str:
    return "\n".join(f"  {k}: {v:.3f}" for k, v in sorted(params.items()))


def make_llm() -> ChatDeepSeek:
    # Замени "deepseek-chat" на свою модель (deepseek-v4.1-flash и т.п.)
    return ChatDeepSeek(model="deepseek-chat", temperature=0.0, max_tokens=1024)


# ---------------- NODES ----------------
async def organ_node(state: dict) -> dict:
    organ_name = state["organ_name"]
    cfg = ORGANS[organ_name]
    body = state["body"]

    read_view = {k: body[k] for k in cfg["reads"] if k in body}

    client = MultiServerMCPClient(cfg["mcp"])
    server_name = next(iter(cfg["mcp"]))
    llm = make_llm()

    # Держим одну MCP-сессию (один подпроцесс сервера) на весь вызов агента.
    async with client.session(server_name) as session:
        tools = await load_mcp_tools(session)
        agent = create_agent(llm, tools, system_prompt=cfg["prompt"])

        user_msg = (
            f"Текущее read-view ({organ_name}):\n"
            f"{format_params(read_view)}\n\n"
            f"Вызови инструменты по необходимости и опиши своё мнение."
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content=user_msg)]})

    # Ответ может прийти как строкой, так и списком контент-блоков.
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    reasoning_text = content

    # Второй вызов — структурированное мнение
    struct_llm = make_llm().with_structured_output(OrganOpinion)
    opinion = cast(
        OrganOpinion,
        await struct_llm.ainvoke([
            SystemMessage(content=(
                f"Ты — {organ_name}. На основе анализа ниже выдай структурированное мнение. "
                f"Можно предлагать ТОЛЬКО эти параметры: {cfg['writes']}. "
                f"Если значение не обосновано — не указывай его."
            )),
            HumanMessage(content=reasoning_text),
        ]),
    )

    # Валидация write set + приведение типов
    opinion.proposed_params = {
        k: float(v)
        for k, v in opinion.proposed_params.items()
        if k in cfg["writes"] and isinstance(v, (int, float))
    }

    print(f"[{organ_name}] reasoning: {opinion.reasoning}")
    print(f"[{organ_name}] proposed: {opinion.proposed_params} "
          f"(conf={opinion.confidence:.2f})")

    return {"opinions": [{"organ": organ_name, "opinion": opinion}]}


async def merge_node(state: BodyState) -> dict:
    body = dict(state["body"])
    opinions = state["opinions"]

    # Собираем предложения по параметрам
    proposals: dict[str, list[tuple[str, float, float]]] = {}
    for entry in opinions:
        organ = entry["organ"]
        op: OrganOpinion = entry["opinion"]
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


# ---------------- ROUTING ----------------
def fan_out(state: BodyState):
    return [
        Send("organ", {"organ_name": name, "body": state["body"]})
        for name in ORGANS
    ]


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
        if diff < 0.5:
            print(f"[graph] converged after {state['iteration']} iterations")
            return "end"
    return "continue"


# ---------------- GRAPH ----------------
def build_graph():
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


# ---------------- ENTRY ----------------
INITIAL_BODY = {
    "heart_rate": 60.0,
    "stroke_volume": 70.0,
    "contractility": 1.0,
    "blood_pressure": 120.0,
    "blood_volume": 4.5,        # лёгкая гиповолемия
    "sodium": 145.0,
    "potassium": 4.5,
    "gfr": 90.0,
    "renin": 1.0,
    "aldosterone": 1.0,
    "oxygen_saturation": 0.94,  # чуть ниже нормы
    "urine_output": 1.5,
    "cardiac_output": 4.2,
}


async def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DEEPSEEK_API_KEY не задан. Скопируй .env.example в .env и вставь ключ, "
            "либо экспортируй переменную окружения DEEPSEEK_API_KEY."
        )

    app = build_graph()
    initial: BodyState = {
        "body": dict(INITIAL_BODY),
        "iteration": 0,
        "max_iterations": 8,
        "opinions": [],
        "history": [{"iter": 0, "body": dict(INITIAL_BODY)}],
    }
    result = await app.ainvoke(initial, {"recursion_limit": 100})

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


if __name__ == "__main__":
    asyncio.run(main())