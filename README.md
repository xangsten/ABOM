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

## Структура

```
ABOM/
├── main.py                  # тонкая точка входа: python main.py
├── pyproject.toml
├── README.md
├── .env / .env.example
├── tests/                   # pytest (ключ DeepSeek не нужен)
└── src/abom/
    ├── __init__.py          # cli() для console-script `abom`
    ├── app.py               # точка входа симуляции (main)
    ├── graph.py             # сборка графа LangGraph
    ├── organs.py            # ORGANS, OrganOpinion, organ_node
    ├── merge.py             # merge_node, fan_out, should_continue
    ├── state.py             # BodyState, opinions_reducer
    ├── config.py            # PHYSIO_LIMITS, AUTHORITY, INITIAL_BODY
    ├── prompts.py           # системные промпты органов
    ├── llm.py               # make_llm() (DeepSeek)
    └── servers/
        ├── heart.py         # MCP-сервер сердца
        └── kidney.py        # MCP-сервер почек
```

MCP-серверы запускаются как модули пакета (`python -m abom.servers.heart` /
`abom.servers.kidney`, transport stdio).

> Внутри `servers/*` используется `FastMCP` из официального SDK `mcp` 1.x
> (`from mcp.server.fastmcp import FastMCP`), а не отдельный пакет `fastmcp`:
> отдельный `fastmcp>=4` тянет `mcp 2.x`, несовместимый с
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
cp .env.example .env      # затем вставь свой ключ DeepSeek в .env

uv run python main.py     # или: uv run abom
# или напрямую интерпретатором venv:
.venv/bin/python main.py
```

Если ключ не задан — программа завершится с понятным сообщением.

## Тесты

```bash
uv run pytest
```

`tests/test_servers.py` поднимает реальные MCP-серверы как подпроцессы
(`python -m abom.servers.<organ>`), поэтому чуть медленнее остальных.

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
- Структурированное мнение органа запрашивается через `method="json_mode"`
  с фолбэком на текстовый JSON-парсер: DeepSeek в thinking-режиме отклоняет
  принудительный `tool_choice`, который использует `with_structured_output`
  по умолчанию (см. `abom/llm.py`).
- `[project.scripts] abom = "abom:cli"` — console-script пакета; `main.py` в
  корне — тонкий шим для запуска как `python main.py`.
