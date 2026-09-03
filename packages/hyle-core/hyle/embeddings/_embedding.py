from array import array
from collections.abc import Buffer, Iterable, Iterator, Sequence
from inspect import BufferFlags
from math import isfinite
from typing import Self, final, overload, override

_FLOAT32_ITEMSIZE = array("f").itemsize


@final
class Embedding(Sequence[float], Buffer):
    __slots__ = ("_values",)

    def __init__(self, values: Iterable[float], /) -> None:
        try:
            storage = array("f", values)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("embedding values must be representable as float32") from error

        self._initialize(storage)

    @classmethod
    def from_buffer(cls, values: Buffer, /) -> Self:
        with memoryview(values) as view:
            if view.ndim != 1 or view.format != "f" or view.itemsize != 4:
                raise TypeError("embedding buffer must be one-dimensional native float32 data")

            storage = array("f")

            try:
                if view.c_contiguous:
                    with view.cast("B") as byte_view:
                        storage.frombytes(byte_view)
                else:
                    storage.frombytes(view.tobytes())
            except (BufferError, TypeError, ValueError) as error:
                raise ValueError("embedding buffer could not be copied as float32") from error

        embedding = cls.__new__(cls)
        embedding._initialize(storage)

        return embedding

    def _initialize(self, values: array[float], /) -> None:
        if _FLOAT32_ITEMSIZE != 4:
            raise RuntimeError("this Python platform does not provide 32-bit array('f') storage")

        if not values:
            raise ValueError("an embedding must contain at least one dimension")

        if any(not isfinite(value) for value in values):
            raise ValueError("embedding values must be finite and representable as float32")

        self._values = values

    @property
    def dimensions(self) -> int:
        return len(self._values)

    @property
    def nbytes(self) -> int:
        return len(self._values) * self._values.itemsize

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    @overload
    def __getitem__(self, index: int, /) -> float: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice, /) -> float | tuple[float, ...]:
        if isinstance(index, slice):
            return tuple(self._values[index])

        return self._values[index]

    def __buffer__(self, flags: int, /) -> memoryview:
        if flags & BufferFlags.WRITABLE:
            raise BufferError("embeddings expose read-only buffers")

        return memoryview(self._values).toreadonly()

    def __eq__(self, other: object, /) -> bool:
        return isinstance(other, Embedding) and self._values == other._values

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(dimensions={self.dimensions}, nbytes={self.nbytes})"
