__all__ = (
    "JsonInput",
    "JsonScalar",
    "JsonValue",
    "decode_json",
    "decode_json_object",
    "encode_json",
    "json_pointer",
    "normalize_json",
    "normalize_json_object",
    "validate_json",
)

from collections.abc import Iterable
from json import JSONDecodeError, dumps, loads
from math import isfinite
from typing import Never, TypeIs

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonInput = str | bytes | bytearray | memoryview

type _Path = list[str | int]


class _DocumentError(ValueError):
    __slots__ = ()


def decode_json(data: JsonInput, /) -> JsonValue:
    return _decode(data)


def decode_json_object(data: JsonInput, /) -> dict[str, JsonValue]:
    document = _decode(data)

    if not isinstance(document, dict):
        raise ValueError("JSON document must have an object root")

    return document


def validate_json(data: JsonInput, /) -> None:
    _decode(data)


def encode_json(value: object, /) -> bytes:
    try:
        _validate_encodable(value, path=[], active=set())
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error

    return _encode(value)


def normalize_json(data: JsonInput, /) -> bytes:
    return _encode(_decode(data))


def normalize_json_object(data: JsonInput, /) -> bytes:
    return _encode(decode_json_object(data))


def json_pointer(path: Iterable[str | int], /) -> str:
    return "".join(f"/{_escape_pointer(str(part))}" for part in path)


def _decode(data: JsonInput, /) -> JsonValue:
    text = _decode_text(data)

    try:
        document: object = loads(
            text,
            object_pairs_hook=_object,
            parse_constant=_constant,
        )
    except JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    except _DocumentError as error:
        raise ValueError(str(error)) from error
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error
    except ValueError as error:
        raise ValueError(f"invalid JSON: {error}") from error

    try:
        if _validate_decoded(document, path=[]):
            return document
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error

    raise AssertionError("JSON validation returned false")


def _decode_text(data: JsonInput, /) -> str:
    if isinstance(data, str):
        return data

    if isinstance(data, memoryview):
        data = data.tobytes()

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("JSON data must be valid UTF-8") from error


def _encode(value: object, /) -> bytes:
    try:
        return dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            check_circular=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(f"value cannot be represented as strict JSON: {error}") from error


def _validate_decoded(value: object, /, *, path: _Path) -> TypeIs[JsonValue]:
    if _validate_scalar(value, path):
        return True

    if _is_dict(value):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AssertionError("json.loads produced a non-string object key")

            _validate_utf8(key, path, key=True)
            path.append(key)

            try:
                _validate_decoded(item, path=path)
            finally:
                path.pop()

        return True

    if _is_list(value):
        for index, item in enumerate(value):
            path.append(index)

            try:
                _validate_decoded(item, path=path)
            finally:
                path.pop()

        return True

    raise AssertionError(f"json.loads produced unsupported type {type(value).__name__!r}")


def _validate_encodable(
    value: object,
    /,
    *,
    path: _Path,
    active: set[int],
) -> None:
    if _validate_scalar(value, path):
        return

    if _is_dict(value):
        marker = id(value)

        if marker in active:
            raise ValueError(f"circular reference at {_location(path)}")

        active.add(marker)

        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"JSON object key at {_location(path)} must be a string")

                _validate_utf8(key, path, key=True)
                path.append(key)

                try:
                    _validate_encodable(item, path=path, active=active)
                finally:
                    path.pop()
        finally:
            active.remove(marker)

        return

    if _is_list_or_tuple(value):
        marker = id(value)

        if marker in active:
            raise ValueError(f"circular reference at {_location(path)}")

        active.add(marker)

        try:
            for index, item in enumerate(value):
                path.append(index)

                try:
                    _validate_encodable(item, path=path, active=active)
                finally:
                    path.pop()
        finally:
            active.remove(marker)

        return

    raise TypeError(
        f"value at {_location(path)} has unsupported JSON type {type(value).__name__!r}"
    )


def _is_dict(value: object, /) -> TypeIs[dict[object, object]]:
    return isinstance(value, dict)


def _is_list(value: object, /) -> TypeIs[list[object]]:
    return isinstance(value, list)


def _is_list_or_tuple(value: object, /) -> TypeIs[list[object] | tuple[object, ...]]:
    return isinstance(value, list | tuple)


def _validate_scalar(value: object, path: _Path, /) -> bool:
    if value is None or isinstance(value, bool | int):
        return True

    if isinstance(value, str):
        _validate_utf8(value, path)
        return True

    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"non-finite number at {_location(path)}")

        return True

    return False


def _validate_utf8(value: str, path: _Path, /, *, key: bool = False) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        subject = f"JSON object key {value!r}" if key else "string"

        raise ValueError(
            f"{subject} at {_location(path)} cannot be represented as UTF-8"
        ) from error


def _object(pairs: list[tuple[str, object]], /) -> dict[str, object]:
    result: dict[str, object] = {}

    for key, value in pairs:
        if key in result:
            raise _DocumentError(f"duplicate JSON object key {key!r}")

        result[key] = value

    return result


def _constant(value: str, /) -> Never:
    raise _DocumentError(f"non-finite JSON number {value!r} is not permitted")


def _location(path: _Path, /) -> str:
    return json_pointer(path) or "<document>"


def _escape_pointer(value: str, /) -> str:
    return value.replace("~", "~0").replace("/", "~1")
