from collections.abc import Iterable, Mapping
from typing import TypeIs

from hyle.tools._tool import Tool

type Tools = Mapping[str, Tool] | Iterable[Tool]


def catalog(tools: object, /) -> dict[str, Tool]:
    result: dict[str, Tool] = {}

    if _is_mapping(tools):
        for name, value in tools.items():
            if not isinstance(name, str):
                raise TypeError("tool mapping keys must be strings")
            tool = _tool(value)
            if name != tool.spec.name:
                raise ValueError(f"tool mapping key {name!r} does not match {tool.spec.name!r}")
            result[name] = tool
        return result

    if not _is_iterable(tools):
        raise TypeError("tools must be a mapping or iterable of Tool values")

    for value in tools:
        tool = _tool(value)
        name = tool.spec.name
        if name in result:
            raise ValueError(f"duplicate tool name {name!r}")
        result[name] = tool
    return result


def _tool(value: object, /) -> Tool:
    if not isinstance(value, Tool):
        raise TypeError("tools must contain Tool values")
    return value


def _is_mapping(value: object, /) -> TypeIs[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_iterable(value: object, /) -> TypeIs[Iterable[object]]:
    return isinstance(value, Iterable)
