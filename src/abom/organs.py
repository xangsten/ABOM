"""Конфигурация органов-агентов и узел organ_node.

Каждый орган — это react-агент с собственным набором MCP-инструментов
(запускается как отдельный подпроцесс: python -m abom.servers.<organ>).
"""
from __future__ import annotations

import sys
from typing import cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field

from abom.llm import make_llm
from abom.prompts import HEART_PROMPT, KIDNEY_PROMPT

MCP_SERVER_MODULE = "abom.servers.{organ}"


def _mcp(organ: str) -> dict:
    """Конфиг stdio-подключения к MCP-серверу органа как к модулю пакета."""
    return {
        organ: {
            "command": sys.executable,
            "args": ["-m", MCP_SERVER_MODULE.format(organ=organ)],
            "transport": "stdio",
        }
    }


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
        "mcp": _mcp("heart"),
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
        "mcp": _mcp("kidney"),
    },
}


class OrganOpinion(BaseModel):
    """Структурированное мнение органа о целевых значениях параметров."""

    proposed_params: dict[str, float] = Field(
        description="Целевые значения. Только ключи из твоего write set."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="1–3 предложения.")


def format_params(params: dict[str, float]) -> str:
    return "\n".join(f"  {k}: {v:.3f}" for k, v in sorted(params.items()))


def _as_text(content) -> str:
    """Ответ модели может быть строкой или списком контент-блоков."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


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

    reasoning_text = _as_text(result["messages"][-1].content)

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
