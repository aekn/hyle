# hyle-core

Provider-independent primitives for building AI systems in Python.

## Installation

Requires Python 3.14+.

```bash
uv add hyle-core
```

Provider integrations are installed separately.

## Quick start

Code can depend on capabilities rather than concrete providers:

```python
from hyle import Model, Request


async def summarize(model: Model, text: str) -> str:
    response = await model.generate(Request(text, instructions="Summarize this in one sentence."))
    return response.require_text()
```

Any compatible `Model` can be passed to `summarize`.

Hyle also provides tools, structured output, embeddings, streaming, and a bounded model-tool runtime. Use the runtime where it fits, or compose the lower-level primitives directly when your application needs a different control flow.

Provider integrations can implement the same core interfaces independently. Higher-level application patterns such as agents, crews, and workflows can be built on top with ordinary Python.

Ollama support is available through `hyle-ollama`. Gemini support is planned next.

## Status

Hyle is a personal project built for fun. There is no fixed roadmap or timeline.

Alpha. The API may change.

## License

[MIT](LICENSE)
