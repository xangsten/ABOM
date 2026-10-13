"""Интеграционные тесты: MCP-серверы поднимаются как модули пакета.

Запускается реальный подпроцесс `python -m abom.servers.<organ>` (stdio),
поэтому тесты чуть медленнее. Ключ DeepSeek не нужен.
"""
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from abom.organs import ORGANS

EXPECTED_TOOLS = {
    "heart": {
        "compute_cardiac_output",
        "compute_mean_arterial_pressure",
        "baroreflex_suggested_hr",
    },
    "kidney": {"compute_gfr", "sodium_balance", "urine_output"},
}


def _load(organ: str):
    """Возвращает (имена_инструментов, функция_вызова) внутри одной сессии."""
    async def run():
        cfg = ORGANS[organ]["mcp"]
        client = MultiServerMCPClient(cfg)
        async with client.session(next(iter(cfg))) as session:
            tools = await load_mcp_tools(session)
            names = [t.name for t in tools]
            by_name = {t.name: t for t in tools}
            # вызываем инструмент, пока сессия ещё открыта
            if organ == "heart":
                res = await by_name["compute_cardiac_output"].ainvoke(
                    {"heart_rate": 100.0, "stroke_volume": 70.0})
            else:
                res = await by_name["compute_gfr"].ainvoke(
                    {"blood_pressure": 120.0, "blood_volume": 4.5, "sodium": 145.0})
            return names, str(res)
    return asyncio.run(run())


def test_heart_server_tools_and_call():
    names, res = _load("heart")
    assert set(names) == EXPECTED_TOOLS["heart"]
    assert "7.0" in res  # HR*SV/1000 = 100*70/1000


def test_kidney_server_tools_and_call():
    names, res = _load("kidney")
    assert set(names) == EXPECTED_TOOLS["kidney"]
    assert "111.86" in res  # ожидаемая СКФ при 120/4.5/145
