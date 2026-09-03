__all__ = ("Msg",)

from dataclasses import dataclass
from typing import final

from hyle._validate import require_label, require_utf8_str
from hyle.content import Content

type _Part = str | Content


@final
@dataclass(frozen=True, slots=True, init=False)
class Msg:
    role: str
    parts: tuple[_Part, ...]

    def __init__(self, role: str, first: _Part, /, *parts: _Part) -> None:
        object.__setattr__(self, "role", require_label(role, "message role"))
        object.__setattr__(self, "parts", tuple(_part(part) for part in (first, *parts)))

    @staticmethod
    def system(first: _Part, /, *parts: _Part) -> Msg:
        return Msg("system", first, *parts)

    @staticmethod
    def user(first: _Part, /, *parts: _Part) -> Msg:
        return Msg("user", first, *parts)

    @staticmethod
    def assistant(first: _Part, /, *parts: _Part) -> Msg:
        return Msg("assistant", first, *parts)


def _part(value: object, /) -> _Part:
    if isinstance(value, str):
        return require_utf8_str(value, "message text")
    if isinstance(value, Content) and type(value) is not Content:
        return value
    raise TypeError("message parts must be strings or Content values")
