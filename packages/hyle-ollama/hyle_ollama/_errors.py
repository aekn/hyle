__all__ = ("OllamaError",)

from hyle.errors import InferenceError


class OllamaError(InferenceError):
    __slots__ = "status_code"

    def __init__(
        self,
        message: str,
        /,
        *,
        status_code: int | None = None,
        transient: bool | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message, transient=transient)
