__all__ = ("JsonSchema",)

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, final, override

from hyle._json import decode_json, encode_json


@final
@dataclass(frozen=True, slots=True, init=False, repr=False)
class JsonSchema:
    """Canonical JSON Schema document stored as compact JSON bytes."""

    data: bytes

    def __init__(
        self,
        schema: Self | bool | Mapping[str, object] | str | bytes | bytearray | memoryview,
        /,
    ) -> None:
        if isinstance(schema, JsonSchema):
            data = schema.data

        elif isinstance(schema, bool):
            data = encode_json(schema)

        elif isinstance(schema, Mapping):
            data = encode_json(schema if isinstance(schema, dict) else dict(schema))

        else:
            document = decode_json(schema)

            if not isinstance(document, bool | dict):
                raise ValueError("a JSON Schema must have an object or boolean root")

            data = encode_json(document)

        object.__setattr__(self, "data", data)

    @property
    def size(self) -> int:
        return len(self.data)

    def __bytes__(self) -> bytes:
        return self.data

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"
