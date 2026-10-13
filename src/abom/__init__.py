"""ABOM — мини-организм: агенты-органы на LangGraph + MCP."""
from __future__ import annotations


def cli() -> None:
    """Синхронная точка входа для console-script `abom` (uv run abom)."""
    import asyncio

    from abom.app import main

    asyncio.run(main())


__all__ = ["cli"]

