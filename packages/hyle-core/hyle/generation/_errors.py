from hyle.errors import HyleError


class StructuredError(HyleError):
    __slots__ = ("kind",)

    def __init__(self, message: str, /, *, kind: str) -> None:
        self.kind = kind
        super().__init__(message)
