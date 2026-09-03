__all__ = ("Embeddable", "Embedder")

from collections.abc import Sequence
from typing import Protocol

from hyle.content import Content
from hyle.embeddings._result import Embeddings

type Embeddable = str | Content | Sequence[str | Content]


class Embedder(Protocol):
    async def embed(
        self,
        first: Embeddable,
        /,
        *inputs: Embeddable,
        dimensions: int | None = None,
        truncate: bool = False,
    ) -> Embeddings: ...
