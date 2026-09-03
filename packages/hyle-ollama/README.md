# hyle-ollama

Ollama integration for Hyle.

Provides generation, streaming, embeddings, and model operations through Ollama's native API.

## Installation

Requires Python 3.14+.

```bash
uv add hyle-ollama
```

This also installs `hyle-core`.

## Quick start

```python
import asyncio

from hyle import Request
from hyle_ollama import OllamaClient, OllamaModel


async def main() -> None:
    async with OllamaClient() as client:
        model = OllamaModel("qwen3:4b", client=client)
        response = await model.generate(Request("Why is the sky blue?"))

    print(response.require_text())


asyncio.run(main())
```

Ollama-specific settings are also supported:

```python
from hyle_ollama import OllamaGenerationConfig, OllamaModel, OllamaOptions


model = OllamaModel(
    "qwen3:4b",
    client=client,
    config=OllamaGenerationConfig(
        think="low",
        options=OllamaOptions(
            num_ctx=16_384,
            temperature=0.2,
        ),
    ),
)
```

`hyle-ollama` follows Ollama's native API, rather than emulating unsupported behavior, and maps it onto Hyle's provider-independent interfaces.

## Status

Hyle is a personal project built for fun. There is no fixed roadmap or timeline.

Alpha. The API may change.

## License

[MIT](LICENSE)
