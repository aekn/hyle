__all__ = (
    "Validator",
    "require_bool",
    "require_finite_float",
    "require_float",
    "require_int",
    "require_label",
    "require_non_blank_str",
    "require_non_blank_utf8_str",
    "require_non_empty_str",
    "require_non_negative_float",
    "require_non_negative_int",
    "require_positive_float",
    "require_positive_int",
    "require_str",
    "require_utf8_str",
)

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Self
from unicodedata import category

type _Normalizer[T] = Callable[[object, str], T]
type _Predicate[T] = Callable[[T], bool]
type _Constraint[T] = tuple[_Predicate[T], str]


@dataclass(frozen=True, slots=True)
class Validator[T]:
    """Normalize a runtime value before applying a reusable set of constraints."""

    _normalize: _Normalizer[T]
    _constraints: tuple[_Constraint[T], ...] = ()

    def __call__(self, value: object, name: str, /) -> T:
        normalized = self._normalize(value, name)

        for predicate, message in self._constraints:
            if not predicate(normalized):
                raise ValueError(f"{name} {message}")

        return normalized

    def optional(self, value: object, name: str, /) -> T | None:
        if value is None:
            return None

        return self(value, name)

    def refine(self, predicate: _Predicate[T], message: str, /) -> Self:
        return type(self)(self._normalize, (*self._constraints, (predicate, message)))


def _bool(value: object, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")

    return value


def _int(value: object, name: str, /) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean")

    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    return value


def _float(value: object, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")

    try:
        return float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is too large to represent as a float") from error


def _str(value: object, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    return value


def _non_blank(value: str, /) -> bool:
    return bool(value.strip())


def _normalized(value: str, /) -> bool:
    return value == value.strip()


def _utf8(value: str, /) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False

    return True


def _textual_label(value: str, /) -> bool:
    return all(
        category(character)[0] != "C" and category(character) not in {"Zl", "Zp"}
        for character in value
    )  # fmt: skip


require_bool = Validator(_bool)

require_int = Validator(_int)
require_non_negative_int = require_int.refine(
    lambda value: value >= 0,
    "must be non-negative",
)
require_positive_int = require_int.refine(
    lambda value: value > 0,
    "must be positive",
)

require_float = Validator(_float)
require_finite_float = require_float.refine(
    isfinite,
    "must be finite",
)
require_non_negative_float = require_finite_float.refine(
    lambda value: value >= 0,
    "must be non-negative",
)
require_positive_float = require_finite_float.refine(
    lambda value: value > 0,
    "must be positive",
)

require_str = Validator(_str)
require_non_empty_str = require_str.refine(
    bool,
    "must not be empty",
)
require_non_blank_str = require_str.refine(
    _non_blank,
    "must contain a non-whitespace character",
)
require_utf8_str = require_str.refine(
    _utf8,
    "must be representable as UTF-8",
)
require_non_blank_utf8_str = require_non_blank_str.refine(
    _utf8,
    "must be representable as UTF-8",
)
require_label = require_non_blank_utf8_str.refine(
    _normalized,
    "must already be normalized",
).refine(
    _textual_label,
    "must not contain control, format, private-use, unassigned, or line-separator characters",
)
