# ABOM — мини-организм: сердце + почки (LangGraph + MCP + DeepSeek)

Игрушечная многоагентная симуляция. Организм описан 13 параметрами. Два
агента-«органа» (**сердце** и **почки**) на каждой итерации:

1. читают своё read-view состояния;
2. вызывают свои **MCP-инструменты** с упрощённой матмоделью (FastMCP, stdio);
3. предлагают целевые значения «своих» параметров (structured output);
4. узел `merge` сводит предложения взвешенно (`authority × confidence`),
   клиппит по физиологическим лимитам и пересчитывает производные.

Граф: `START → fan_out(Send к обоим органам) → organ → merge → (organ|END)`.
Выход — по `max_iterations` или по сходимости.

## Файлы

| Файл | Назначение |
|---|---|
| `main.py` | Состояние, граф, узлы органов и `merge`, точка входа |
| `heart_mcp.py` | MCP-сервер `HeartModel` (CO, MAP, барорефлекс) |
| `kidney_mcp.py` | MCP-сервер `KidneyModel` (GFR, Na-баланс, диурез) |
| `.env.example` | Шаблон окружения (ключ DeepSeek) |

> MCP-серверы используют `FastMCP` из официального SDK `mcp` 1.x
> (`from mcp.server.fastmcp import FastMCP`), а не отдельный пакет `fastmcp`:
> отдельный `fastmcp>=4` тянет `mcp 2.x`, который несовместим с
> `langchain-mcp-adapters`.

## Установка

```bash
uv sync
```

## Ключ DeepSeek (безопасно)

Ключ хранится в `.env` (файл в `.gitignore` и **не коммитится**).

```bash
cp .env.example .env
# затем открой .env и вставь свой ключ:
#   DEEPSEEK_API_KEY=sk-...
```

Ключ можно задать и переменной окружения (тогда `.env` не нужен):

```bash
export DEEPSEEK_API_KEY=sk-...
```

`main.py` читает `.env` через `python-dotenv` (`load_dotenv()`), а
`ChatDeepSeek` берёт `DEEPSEEK_API_KEY` из окружения.

## Запуск

```bash
uv run python main.py
# или
.venv/bin/python main.py
```

Если ключ не задан — программа завершится с понятным сообщением.

## Зависимости (совместимый набор)

Зафиксированы ограничения, которые гарантируют совместную работу:

- `mcp>=1.24,<2` — MCP SDK 1.x (`langchain-mcp-adapters` требует `mcp<2`)
- `langchain-mcp-adapters>=0.3.2,<0.4`
- `langchain>=1.4,<2` — даёт `create_agent` (замена устаревшего `create_react_agent`)
- `langgraph>=1.2,<2`, `langchain-core>=1.4,<2`
- `langchain-deepseek>=1.1,<2`, `pydantic>=2.7,<3`, `python-dotenv>=1.0`

## Известные ограничения / заметки

- Критерий сходимости (`sum(|Δ|) < 0.5`) не нормирован по масштабу параметров.
- `cardiac_output` пересчитывается как производная и не клиппится отдельно.
- Ошибки MCP-серверов / API-вызовов отдельно не обрабатываются.
- `[project.scripts] abom = "abom:main"` в `pyproject.toml` указывает на
  заглушку в `src/abom/`; канонический вход сейчас — `main.py`.
