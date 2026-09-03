# Examples

Small programs built from Hyle primitives.

| Example | Pattern | Shows |
| --- | --- | --- |
| `trip_planner.py` | multi-agent orchestration | composing small agent abstractions with ordinary Python and running specialists concurrently |
| `bank_support.py` | typed tool-using agent | application state, tools, automatic tool execution, and structured output |
| `email_workflow.py` | evaluator-optimizer workflow | explicit control flow, typed evaluation, and iterative refinement |

Run an example from the repository root:

```bash
uv run python examples/trip_planner.py
```

Use another model with `HYLE_MODEL`:

```bash
HYLE_MODEL=qwen3:8b uv run python examples/trip_planner.py
```

The examples use Ollama by default.
