__all__ = ("Usage",)

from dataclasses import dataclass
from typing import final

from hyle._validate import require_non_negative_int


@final
@dataclass(frozen=True, slots=True)
class Usage:
    input: int | None = None
    output: int | None = None

    def __post_init__(self) -> None:
        require_non_negative_int.optional(self.input, "input tokens")
        require_non_negative_int.optional(self.output, "output tokens")

    @property
    def total(self) -> int | None:
        return _sum(self.input, self.output)

    def __add__(self, other: Usage, /) -> Usage:
        return Usage(_sum(self.input, other.input), _sum(self.output, other.output))


def _sum(left: int | None, right: int | None, /) -> int | None:
    if left is None or right is None:
        return None
    return left + right
