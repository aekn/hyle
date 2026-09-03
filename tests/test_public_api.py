from types import ModuleType

import pytest

import hyle
import hyle.content
import hyle.embeddings
import hyle.errors
import hyle.generation
import hyle.runtime
import hyle.schema
import hyle.tools
import hyle.usage
import hyle_ollama

_ROOT_EXPORTS = {
    "Model": hyle.generation.Model,
    "Msg": hyle.generation.Msg,
    "Request": hyle.generation.Request,
    "Response": hyle.generation.Response,
    "Structured": hyle.generation.Structured,
    "run": hyle.runtime.run,
    "tool": hyle.tools.tool,
}


def test_hyle_root_is_deliberately_small() -> None:
    assert hyle.__all__ == tuple(_ROOT_EXPORTS)
    for name, value in _ROOT_EXPORTS.items():
        assert getattr(hyle, name) is value


@pytest.mark.parametrize(
    "module",
    (
        hyle,
        hyle.content,
        hyle.embeddings,
        hyle.errors,
        hyle.generation,
        hyle.runtime,
        hyle.schema,
        hyle.tools,
        hyle.usage,
        hyle_ollama,
    ),
)
def test_public_exports_are_unique_and_resolvable(module: ModuleType) -> None:
    assert len(module.__all__) == len(set(module.__all__))
    for name in module.__all__:
        assert not name.startswith("_")
        assert hasattr(module, name), f"{module.__name__}.{name} is missing"
