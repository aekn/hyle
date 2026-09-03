__all__ = ("OllamaEmbeddingConfig", "OllamaGenerationConfig", "OllamaOptions")

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from hyle_ollama._validate import boolean, finite, integer, optional_boolean, text


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaOptions:
    num_ctx: int | None = None
    num_batch: int | None = None
    num_gpu: int | None = None
    main_gpu: int | None = None
    use_mmap: bool | None = None
    num_thread: int | None = None
    draft_num_predict: int | None = None

    num_keep: int | None = None
    seed: int | None = None
    num_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    repeat_last_n: int | None = None
    temperature: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "num_ctx",
            "num_batch",
            "num_gpu",
            "main_gpu",
            "num_thread",
            "draft_num_predict",
            "num_keep",
            "seed",
            "num_predict",
            "top_k",
            "repeat_last_n",
        ):
            integer(getattr(self, name), name)

        optional_boolean(self.use_mmap, "use_mmap")

        for name in (
            "top_p",
            "min_p",
            "typical_p",
            "temperature",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            finite(getattr(self, name), name)

        if isinstance(self.stop, str):
            raise TypeError("stop must be a sequence of strings")
        object.__setattr__(
            self,
            "stop",
            tuple(text(value, "stop sequence", blank=True) for value in self.stop),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaGenerationConfig:
    think: bool | Literal["low", "medium", "high", "max"] | None = None
    keep_alive: str | int | float | None = None
    truncate: bool | None = None
    shift: bool | None = None
    json_mode: bool = False
    logprobs: bool = False
    top_logprobs: int | None = None
    options: OllamaOptions = OllamaOptions()

    def __post_init__(self) -> None:
        _require_options(self.options)
        _thinking(self.think)
        _keep_alive(self.keep_alive)
        optional_boolean(self.truncate, "truncate")
        optional_boolean(self.shift, "shift")
        boolean(self.json_mode, "json_mode")
        boolean(self.logprobs, "logprobs")

        top = self.top_logprobs
        if top is not None:
            integer(top, "top_logprobs")
            if not 0 <= top <= 20:
                raise ValueError("top_logprobs must be between 0 and 20")
            if not self.logprobs:
                raise ValueError("top_logprobs requires logprobs=True")


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaEmbeddingConfig:
    keep_alive: str | int | float | None = None
    options: OllamaOptions = OllamaOptions()

    def __post_init__(self) -> None:
        _require_options(self.options)
        _keep_alive(self.keep_alive)


def _thinking(value: object, /) -> None:
    if value is None or isinstance(value, bool):
        return
    if not isinstance(value, str):
        raise TypeError("think must be a boolean or string")
    if value not in {"low", "medium", "high", "max"}:
        raise ValueError(f"invalid think value {value!r}")


def _keep_alive(value: object, /) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text(value, "keep_alive", blank=False)
        return
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("keep_alive must be a string or finite number")
    if not isfinite(value):
        raise ValueError("keep_alive must be finite")


def _require_options(value: object, /) -> OllamaOptions:
    if not isinstance(value, OllamaOptions):
        raise TypeError("options must be OllamaOptions")
    return value
