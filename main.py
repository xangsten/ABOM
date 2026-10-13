"""Тонкая точка входа: `python main.py`.

Основная логика — в пакете `abom` (см. `abom.app.main`).
Альтернатива: console-script `uv run abom` (см. `abom.cli`).
"""
import asyncio

from abom.app import main

if __name__ == "__main__":
    asyncio.run(main())
