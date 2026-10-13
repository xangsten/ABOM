"""MCP-сервер: математические модели почек."""
# FastMCP встроен в официальный SDK `mcp` 1.x.
# Отдельный пакет `fastmcp` (>= 4) тянет mcp 2.x и конфликтует с langchain-mcp-adapters.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("KidneyModel")


@mcp.tool()
def compute_gfr(
    blood_pressure: float,
    blood_volume: float,
    sodium: float,
) -> dict:
    """Упрощённая СКФ (mL/min): зависит от перфузии, объёма и Na+."""
    base = 90.0
    pressure_factor = max(0.3, min(1.5, blood_pressure / 90.0))
    volume_factor = max(0.5, min(1.3, blood_volume / 5.0))
    sodium_factor = max(0.7, min(1.2, sodium / 140.0))
    gfr = base * pressure_factor * volume_factor * sodium_factor
    return {"gfr": round(gfr, 2)}


@mcp.tool()
def sodium_balance(
    sodium: float,
    aldosterone: float,
    blood_volume: float,
) -> dict:
    """РААС: альдостерон задерживает Na+, перегрузка объёмом разводит."""
    target = 140.0
    if aldosterone > 1.0:
        target += (aldosterone - 1.0) * 5.0
    if blood_volume > 5.5:
        target -= (blood_volume - 5.5) * 3.0
    suggested = (sodium + target) / 2.0
    suggested = max(125.0, min(155.0, suggested))
    return {"suggested_sodium": round(suggested, 2)}


@mcp.tool()
def urine_output(gfr: float, aldosterone: float) -> dict:
    """Диурез (mL/min) ∝ СКФ, подавляется альдостероном."""
    base = (gfr / 90.0) * 1.5
    if aldosterone > 1.0:
        base = base / aldosterone
    return {"urine_output": round(max(0.1, base), 3)}


def main() -> None:
    mcp.run()  # stdio по умолчанию


if __name__ == "__main__":
    main()
