__all__ = ("RunLimits",)

from dataclasses import dataclass

from hyle._validate import require_non_negative_int, require_positive_float, require_positive_int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunLimits:
    turns: int = 8
    tool_calls: int = 32
    parallel_tools: int = 1

    tool_timeout: float | None = 30.0
    turn_timeout: float | None = None
    timeout: float | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens_per_turn: int | None = None

    def __post_init__(self) -> None:
        for name in ("turns", "parallel_tools"):
            object.__setattr__(
                self,
                name,
                require_positive_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "tool_calls",
            require_non_negative_int(self.tool_calls, "tool_calls"),
        )

        for name in ("tool_timeout", "turn_timeout", "timeout"):
            object.__setattr__(
                self, name, require_positive_float.optional(getattr(self, name), name)
            )

        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "input_tokens_per_turn",
        ):
            object.__setattr__(
                self,
                name,
                require_non_negative_int.optional(getattr(self, name), name),
            )
