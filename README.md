# Hyle

Typed Python primitives for building AI systems.

Hyle is a small, provider-independent foundation for models, tools, structured output, embeddings, streaming, and execution.

Provider integrations live below the core. Higher-level application patterns such as agents, crews, and workflows can be composed above it with ordinary Python.

## Installation

Requires Python 3.14+.

```bash
uv add hyle-core
```

Provider integrations are separate packages.

```bash
uv add hyle-ollama
```

Ollama is the first integration. Gemini support is planned, and additional providers can be implemented using the same core interfaces.

## Quick start

Application code can depend on Hyle without depending on a provider:

```python
import asyncio

from hyle import Model, run, tool
from hyle_ollama import OllamaClient, OllamaModel


@tool
def order_status(order_id: str) -> str:
    """Return the status of an order."""
    return {"A100": "shipped"}.get(order_id, "unknown")


async def answer(model: Model, question: str) -> str:
    result = await run(model, question, tools=(order_status,))
    return result.require_text()


async def main() -> None:
    async with OllamaClient() as client:
        model = OllamaModel("qwen3:4b", client=client)
        print(await answer(model, "Where is order A100?"))


asyncio.run(main())
```

Hyle owns the primitives. Your application owns the form.

See [`examples/`](examples/) for multi-agent orchestration, typed tool use, and evaluator-optimizer workflows built from the same pieces.

## Packages

| Package | Import | Purpose |
| --- | --- | --- |
| `hyle-core` | `hyle` | Provider-independent core |
| `hyle-ollama` | `hyle_ollama` | Ollama integration |

Provider integrations and higher-level application patterns can be built independently around the core.

## Development

```bash
uv sync --locked --all-packages
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

## Status

Hyle is a personal project built for fun. There is no fixed roadmap or timeline.

Alpha. The API may change.

## License

[MIT](LICENSE)
