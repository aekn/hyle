__all__ = ("HyleError", "InferenceError", "UnsupportedFeatureError")


class HyleError(Exception):
    """Base class for expected Hyle operational errors."""

    __slots__ = ()


class InferenceError(HyleError):
    """An inference operation failed without producing a usable result."""

    __slots__ = ("transient",)

    def __init__(self, message: str, /, *, transient: bool | None = None) -> None:
        self.transient = transient
        super().__init__(message)


class UnsupportedFeatureError(InferenceError):
    """The selected backend cannot preserve requested semantics."""

    __slots__ = ()

    def __init__(self, message: str, /) -> None:
        super().__init__(message, transient=False)
