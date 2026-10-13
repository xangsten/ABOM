"""MCP-сервер: математические модели сердца."""
# FastMCP встроен в официальный SDK `mcp` 1.x.
# Отдельный пакет `fastmcp` (>= 4) тянет mcp 2.x и конфликтует с langchain-mcp-adapters.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("HeartModel")


@mcp.tool()
def compute_cardiac_output(heart_rate: float, stroke_volume: float) -> dict:
    """Cardiac output (L/min) = HR * SV / 1000.
    Args:
        heart_rate: bpm
        stroke_volume: mL/beat
    """
    co = (heart_rate * stroke_volume) / 1000.0
    return {"cardiac_output": round(co, 3)}


@mcp.tool()
def compute_mean_arterial_pressure(systolic: float, diastolic: float) -> dict:
    """MAP = DBP + (SBP - DBP) / 3."""
    map_val = diastolic + (systolic - diastolic) / 3.0
    return {"mean_arterial_pressure": round(map_val, 2)}


@mcp.tool()
def baroreflex_suggested_hr(
    heart_rate: float,
    oxygen_saturation: float,
    blood_pressure: float,
) -> dict:
    """Барорефлекс: если O2 или BP низкие — HR растёт.
    Args:
        heart_rate: текущий HR
        oxygen_saturation: 0..1
        blood_pressure: MAP в mmHg
    """
    target = heart_rate
    if oxygen_saturation < 0.95:
        target += (0.95 - oxygen_saturation) * 150.0
    if blood_pressure < 85:
        target += (85 - blood_pressure) * 1.5
    elif blood_pressure > 120:
        target -= (blood_pressure - 120) * 1.0
    target = max(40.0, min(180.0, target))
    return {"suggested_heart_rate": round(target, 1)}


if __name__ == "__main__":
    mcp.run()  # stdio по умолчанию