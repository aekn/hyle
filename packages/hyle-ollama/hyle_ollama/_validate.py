from math import isfinite


def integer(value: object, name: str, /) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError(f"{name} must be an integer")


def finite(value: object, name: str, /) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite number")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def boolean(value: object, name: str, /) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def optional_boolean(value: object, name: str, /) -> None:
    if value is not None:
        boolean(value, name)


def positive_int(value: object, name: str, /) -> int:
    integer(value, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def text(value: object, name: str, /, *, blank: bool) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be UTF-8 text") from error
    if not blank and not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value
