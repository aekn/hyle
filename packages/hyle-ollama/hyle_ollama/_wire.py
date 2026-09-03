from datetime import datetime

from msgspec import UNSET, Raw, Struct, UnsetType

type UnsetOr[T] = T | UnsetType


class ErrorWire(Struct, omit_defaults=True):
    error: str = ""
    signin_url: str = ""


class ToolFunctionWire(Struct, omit_defaults=True):
    name: str
    parameters: Raw
    description: UnsetOr[str] = UNSET


class ToolWire(Struct, omit_defaults=True):
    function: ToolFunctionWire
    type: str


class ToolCallFunctionWire(Struct, omit_defaults=True):
    index: int
    name: str
    arguments: Raw


class ToolCallWire(Struct, omit_defaults=True):
    function: ToolCallFunctionWire
    id: UnsetOr[str] = UNSET


class MessageWire(Struct, omit_defaults=True):
    role: str
    content: str = ""
    thinking: UnsetOr[str] = UNSET
    images: UnsetOr[tuple[bytes, ...]] = UNSET
    tool_calls: UnsetOr[tuple[ToolCallWire, ...]] = UNSET
    tool_name: UnsetOr[str] = UNSET
    tool_call_id: UnsetOr[str] = UNSET


class ChatRequestWire(Struct, omit_defaults=True):
    model: str
    messages: tuple[MessageWire, ...]
    stream: UnsetOr[bool] = UNSET
    format: UnsetOr[Raw] = UNSET
    keep_alive: UnsetOr[str | int | float] = UNSET
    tools: UnsetOr[tuple[ToolWire, ...]] = UNSET
    options: UnsetOr[dict[str, object]] = UNSET
    think: UnsetOr[bool | str] = UNSET
    truncate: UnsetOr[bool] = UNSET
    shift: UnsetOr[bool] = UNSET
    logprobs: UnsetOr[bool] = UNSET
    top_logprobs: UnsetOr[int] = UNSET


class TokenLogprobWire(Struct, omit_defaults=True):
    token: str
    logprob: float
    bytes: tuple[int, ...] = ()


class LogprobWire(TokenLogprobWire, omit_defaults=True):
    top_logprobs: tuple[TokenLogprobWire, ...] = ()


class ChatRecordWire(Struct, omit_defaults=True):
    model: UnsetOr[str] = UNSET
    remote_model: UnsetOr[str] = UNSET
    remote_host: UnsetOr[str] = UNSET
    created_at: UnsetOr[datetime] = UNSET
    message: UnsetOr[MessageWire] = UNSET
    done: UnsetOr[bool] = UNSET
    done_reason: UnsetOr[str] = UNSET
    logprobs: tuple[LogprobWire, ...] = ()
    total_duration: UnsetOr[int] = UNSET
    load_duration: UnsetOr[int] = UNSET
    prompt_eval_count: UnsetOr[int] = UNSET
    prompt_eval_cached_count: UnsetOr[int] = UNSET
    prompt_eval_duration: UnsetOr[int] = UNSET
    eval_count: UnsetOr[int] = UNSET
    eval_duration: UnsetOr[int] = UNSET
    error: UnsetOr[str] = UNSET


class EmbedRequestWire(Struct, omit_defaults=True):
    model: str
    input: tuple[str, ...]
    truncate: bool
    keep_alive: UnsetOr[str | int | float] = UNSET
    dimensions: UnsetOr[int] = UNSET
    options: UnsetOr[dict[str, object]] = UNSET


class EmbedResponseWire(Struct, omit_defaults=True):
    model: str
    embeddings: tuple[tuple[float, ...], ...]
    total_duration: UnsetOr[int] = UNSET
    load_duration: UnsetOr[int] = UNSET
    prompt_eval_count: UnsetOr[int] = UNSET


class ModelMetadataWire(Struct, omit_defaults=True, frozen=True):
    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: tuple[str, ...] = ()
    parameter_size: str = ""
    quantization_level: str = ""
    context_length: UnsetOr[int] = UNSET
    embedding_length: UnsetOr[int] = UNSET


class ModelSummaryWire(Struct, omit_defaults=True):
    name: str
    model: str
    modified_at: datetime
    size: int
    digest: str
    details: ModelMetadataWire = ModelMetadataWire()
    capabilities: tuple[str, ...] = ()
    remote_model: UnsetOr[str] = UNSET
    remote_host: UnsetOr[str] = UNSET


class ListResponseWire(Struct):
    models: tuple[ModelSummaryWire, ...]


class RunningModelWire(Struct, omit_defaults=True):
    name: str
    model: str
    size: int
    digest: str
    expires_at: datetime
    size_vram: int
    context_length: int
    details: ModelMetadataWire = ModelMetadataWire()


class ProcessResponseWire(Struct):
    models: tuple[RunningModelWire, ...]


class ShowRequestWire(Struct, omit_defaults=True):
    model: str
    verbose: bool = False


class ShowResponseWire(Struct, omit_defaults=True):
    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    system: str = ""
    renderer: str = ""
    parser: str = ""
    details: ModelMetadataWire = ModelMetadataWire()
    remote_model: UnsetOr[str] = UNSET
    remote_host: UnsetOr[str] = UNSET
    capabilities: tuple[str, ...] = ()
    modified_at: UnsetOr[datetime] = UNSET
    requires: UnsetOr[str] = UNSET
    model_info: UnsetOr[Raw] = UNSET
    projector_info: UnsetOr[Raw] = UNSET
    tensors: UnsetOr[Raw] = UNSET
    messages: UnsetOr[Raw] = UNSET


class CopyRequestWire(Struct):
    source: str
    destination: str


class ModelRequestWire(Struct):
    model: str


class TransferRequestWire(Struct, omit_defaults=True):
    model: str
    stream: bool
    insecure: UnsetOr[bool] = UNSET


class ProgressWire(Struct, omit_defaults=True):
    status: str
    digest: UnsetOr[str] = UNSET
    total: UnsetOr[int] = UNSET
    completed: UnsetOr[int] = UNSET


class VersionWire(Struct):
    version: str
