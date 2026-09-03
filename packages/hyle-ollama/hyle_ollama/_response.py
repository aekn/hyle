__all__ = (
    "OllamaEmbeddings",
    "OllamaLogprob",
    "OllamaModelDetails",
    "OllamaModelMetadata",
    "OllamaModelSummary",
    "OllamaProgress",
    "OllamaResponse",
    "OllamaRunningModel",
    "OllamaTimings",
    "OllamaTokenLogprob",
)

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from hyle.embeddings import Embedding, Embeddings
from hyle.generation import Response
from hyle.usage import Usage


@dataclass(frozen=True, slots=True)
class OllamaTokenLogprob:
    token: str
    logprob: float
    token_bytes: bytes = b""


@dataclass(frozen=True, slots=True)
class OllamaLogprob:
    selected: OllamaTokenLogprob
    top: tuple[OllamaTokenLogprob, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "top", tuple(self.top))


@dataclass(frozen=True, slots=True)
class OllamaTimings:
    total_ns: int | None = None
    load_ns: int | None = None
    prompt_eval_ns: int | None = None
    eval_ns: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class OllamaResponse(Response):
    created_at: datetime
    timings: OllamaTimings
    done_reason: str | None = None
    prompt_eval_cached_count: int | None = None
    logprobs: tuple[OllamaLogprob, ...] = ()
    remote_model: str | None = None
    remote_host: str | None = None

    def __post_init__(self) -> None:
        Response.__post_init__(self)
        object.__setattr__(self, "logprobs", tuple(self.logprobs))


@dataclass(frozen=True, slots=True, init=False, repr=False)
class OllamaEmbeddings(Embeddings):
    timings: OllamaTimings

    def __init__(
        self,
        values: Iterable[Embedding],
        /,
        *,
        usage: Usage,
        timings: OllamaTimings,
        model: str | None = None,
    ) -> None:
        Embeddings.__init__(self, values, usage=usage, provider="ollama", model=model)
        object.__setattr__(self, "timings", timings)


@dataclass(frozen=True, slots=True)
class OllamaModelMetadata:
    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: tuple[str, ...] = ()
    parameter_size: str = ""
    quantization_level: str = ""
    context_length: int | None = None
    embedding_length: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", tuple(self.families))


@dataclass(frozen=True, slots=True)
class OllamaModelSummary:
    name: str
    model: str
    modified_at: datetime
    size: int
    digest: str
    metadata: OllamaModelMetadata
    capabilities: tuple[str, ...] = ()
    remote_model: str | None = None
    remote_host: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))


@dataclass(frozen=True, slots=True)
class OllamaRunningModel:
    name: str
    model: str
    size: int
    digest: str
    metadata: OllamaModelMetadata
    expires_at: datetime
    size_vram: int
    context_length: int


@dataclass(frozen=True, slots=True)
class OllamaModelDetails:
    model: str
    metadata: OllamaModelMetadata
    capabilities: tuple[str, ...] = ()
    modified_at: datetime | None = None
    requires: str | None = None
    remote_model: str | None = None
    remote_host: str | None = None
    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    system: str = ""
    renderer: str = ""
    parser: str = ""
    model_info_json: bytes = b"{}"
    projector_info_json: bytes = b"{}"
    tensors_json: bytes = b"[]"
    messages_json: bytes = b"[]"

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))


@dataclass(frozen=True, slots=True)
class OllamaProgress:
    status: str
    digest: str | None = None
    total: int | None = None
    completed: int | None = None
