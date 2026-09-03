__all__ = ("Binary", "Content", "ResourceRef")

from collections.abc import Buffer
from dataclasses import dataclass
from re import Pattern
from re import compile as compile_pattern
from typing import final, override
from unicodedata import category

_TOKEN = r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"
_MEDIA_TYPE: Pattern[str] = compile_pattern(rf"{_TOKEN}/{_TOKEN}")
_URI_SCHEME: Pattern[str] = compile_pattern(r"[A-Za-z][A-Za-z0-9+.-]*")


class Content:
    """Extension base for model visible content."""

    __slots__ = ()


@final
@dataclass(frozen=True, slots=True, init=False, repr=False)
class Binary(Content):
    """Owned binary content with optional format and display name."""

    data: bytes
    media_type: str | None
    name: str | None

    def __init__(
        self,
        data: Buffer,
        /,
        *,
        media_type: str | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            _validate_name(name, "binary name")

        if isinstance(data, bytes):
            owned = data
        else:
            with memoryview(data) as view:
                owned = view.tobytes()

        object.__setattr__(self, "data", owned)
        object.__setattr__(
            self,
            "media_type",
            None if media_type is None else _normalize_media_type(media_type),
        )
        object.__setattr__(self, "name", name)

    @property
    def size(self) -> int:
        return len(self.data)

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(size={self.size}, "
            f"media_type={self.media_type!r}, name={self.name!r})"
        )


@dataclass(frozen=True, slots=True, init=False)
class ResourceRef(Content):
    uri: str
    media_type: str | None
    name: str | None

    def __init__(
        self,
        uri: str,
        /,
        *,
        media_type: str | None = None,
        name: str | None = None,
    ) -> None:
        _validate_uri(uri)
        if name is not None:
            _validate_name(name, "resource name")

        object.__setattr__(self, "uri", uri)
        object.__setattr__(
            self,
            "media_type",
            None if media_type is None else _normalize_media_type(media_type),
        )
        object.__setattr__(self, "name", name)


def _validate_name(value: str, field: str, /) -> None:
    visible = False
    for character in value:
        char_category = category(character)
        if char_category[0] == "C" or char_category in {"Zl", "Zp"}:
            raise ValueError(f"{field} contains a non-text character")
        if char_category[0] != "Z":
            visible = True
    if not visible:
        raise ValueError(f"{field} must contain a visible character")


def _validate_uri(value: str, /) -> None:
    if not value:
        raise ValueError("resource URI must not be empty")
    if any(category(character)[0] in {"C", "Z"} for character in value):
        raise ValueError("resource URI must not contain whitespace or non-text characters")

    scheme, separator, _ = value.partition(":")
    if not separator or _URI_SCHEME.fullmatch(scheme) is None:
        raise ValueError("resource URI must be absolute and include a valid scheme")


def _normalize_media_type(value: str, /) -> str:
    normalized = value.strip().lower()
    if _MEDIA_TYPE.fullmatch(normalized) is None:
        raise ValueError(f"invalid media type {value!r}")

    media_type, subtype = normalized.split("/", 1)
    if media_type == "*" or subtype == "*":
        raise ValueError(f"media type must identify a specific format, got {value!r}")
    return normalized
