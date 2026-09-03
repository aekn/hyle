__all__ = ("Embeddings",)

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import overload, override

from hyle._validate import require_utf8_str
from hyle.embeddings._embedding import Embedding
from hyle.usage import Usage


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Embeddings(Sequence[Embedding]):
    _values: tuple[Embedding, ...]
    usage: Usage
    provider: str | None
    model: str | None
    id: str | None

    def __init__(
        self,
        values: Iterable[Embedding],
        /,
        *,
        usage: Usage,
        provider: str | None = None,
        model: str | None = None,
        id: str | None = None,
    ) -> None:
        usage = _usage(usage)
        values = _values(values)
        dimensions = values[0].dimensions
        if any(value.dimensions != dimensions for value in values[1:]):
            raise ValueError("all embeddings in one batch must have the same dimensions")

        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(
            self,
            "provider",
            require_utf8_str.optional(provider, "embedding provider"),
        )
        object.__setattr__(
            self,
            "model",
            require_utf8_str.optional(model, "embedding model"),
        )
        object.__setattr__(
            self,
            "id",
            require_utf8_str.optional(id, "embedding response ID"),
        )

    @property
    def dimensions(self) -> int:
        return self._values[0].dimensions

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Embedding]:
        return iter(self._values)

    @overload
    def __getitem__(self, index: int, /) -> Embedding: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[Embedding, ...]: ...

    def __getitem__(self, index: int | slice, /) -> Embedding | tuple[Embedding, ...]:
        return self._values[index]

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(count={len(self)}, dimensions={self.dimensions}, "
            f"provider={self.provider!r}, model={self.model!r})"
        )


def _usage(value: object, /) -> Usage:
    if isinstance(value, Usage):
        return value
    raise TypeError("usage must be Usage")


def _values(values: Iterable[object], /) -> tuple[Embedding, ...]:
    normalized: list[Embedding] = []
    for value in values:
        if not isinstance(value, Embedding):
            raise TypeError("embeddings must contain Embedding values")
        normalized.append(value)
    if not normalized:
        raise ValueError("embeddings must contain at least one vector")
    return tuple(normalized)
