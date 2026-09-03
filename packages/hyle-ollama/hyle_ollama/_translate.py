from collections.abc import Hashable, Iterable

from msgspec import UNSET, Raw, UnsetType

from hyle.content import Binary, Content, ResourceRef
from hyle.embeddings import Embedding
from hyle.errors import UnsupportedFeatureError
from hyle.generation import Msg, Part, Reasoning, Request, Response, StopReason
from hyle.tools import ToolCall, ToolResult, ToolSpec
from hyle.usage import Usage
from hyle_ollama._config import OllamaEmbeddingConfig, OllamaGenerationConfig, OllamaOptions
from hyle_ollama._errors import OllamaError
from hyle_ollama._response import (
    OllamaEmbeddings,
    OllamaLogprob,
    OllamaModelDetails,
    OllamaModelMetadata,
    OllamaModelSummary,
    OllamaProgress,
    OllamaResponse,
    OllamaRunningModel,
    OllamaTimings,
    OllamaTokenLogprob,
)
from hyle_ollama._wire import (
    ChatRecordWire,
    ChatRequestWire,
    CopyRequestWire,
    EmbedRequestWire,
    EmbedResponseWire,
    ListResponseWire,
    LogprobWire,
    MessageWire,
    ModelMetadataWire,
    ModelRequestWire,
    ProcessResponseWire,
    ProgressWire,
    ShowRequestWire,
    ShowResponseWire,
    TokenLogprobWire,
    ToolCallFunctionWire,
    ToolCallWire,
    ToolFunctionWire,
    ToolWire,
    TransferRequestWire,
)

_TEXT_STREAM_KEY = "text"
_REASONING_STREAM_KEY = "reasoning"


def encode_chat_request(
    model: str,
    request: Request,
    config: OllamaGenerationConfig,
    /,
    *,
    stream: bool,
) -> ChatRequestWire:
    if request.schema is not None and config.json_mode:
        raise ValueError("schema and json_mode are mutually exclusive")

    if request.schema is not None:
        format_value: Raw | UnsetType = Raw(request.schema.data)
    elif config.json_mode:
        format_value = Raw(b'"json"')
    else:
        format_value = UNSET

    tools = _encode_tools(request)
    options = _options(config.options, max_output_tokens=request.max_output_tokens)

    return ChatRequestWire(
        model=model,
        messages=tuple(_encode_history(item) for item in request.input),
        stream=stream,
        format=format_value,
        keep_alive=_unset(config.keep_alive),
        tools=UNSET if not tools else tools,
        options=UNSET if not options else options,
        think=_unset(config.think),
        truncate=_unset(config.truncate),
        shift=_unset(config.shift),
        logprobs=True if config.logprobs else UNSET,
        top_logprobs=_unset(config.top_logprobs),
    )


def decode_chat_response(record: ChatRecordWire) -> OllamaResponse:
    if record.error is not UNSET and record.error:
        raise OllamaError(record.error, transient=False)
    if record.done is not True:
        raise OllamaError("chat response is not terminal", transient=False)
    if record.message is UNSET:
        raise OllamaError("chat response has no message", transient=False)

    return build_response(
        record,
        _decode_message(record.message),
        tuple(_decode_logprob(value) for value in record.logprobs),
    )


def decode_stream_parts(
    message: MessageWire,
    /,
) -> Iterable[tuple[Part, Hashable | None]]:
    if message.role.lower() not in {"", "assistant"}:
        raise OllamaError("invalid assistant role", transient=False)

    if message.thinking is not UNSET and message.thinking:
        yield Reasoning(message.thinking), _REASONING_STREAM_KEY
    if message.content:
        yield message.content, _TEXT_STREAM_KEY
    if message.images is not UNSET:
        for image in message.images:
            yield Binary(image), None
    if message.tool_calls is not UNSET:
        for tool_call in message.tool_calls:
            yield _decode_tool_call(tool_call), None


def decode_logprobs(values: tuple[LogprobWire, ...], /) -> tuple[OllamaLogprob, ...]:
    return tuple(_decode_logprob(value) for value in values)


def build_response(
    terminal: ChatRecordWire,
    parts: tuple[Part, ...],
    logprobs: tuple[OllamaLogprob, ...],
    /,
    *,
    remote_model: str | None = None,
    remote_host: str | None = None,
) -> OllamaResponse:
    if terminal.done is not True:
        raise OllamaError("terminal record has done=false", transient=False)
    if terminal.model is UNSET:
        raise OllamaError("terminal record has no model", transient=False)
    if terminal.created_at is UNSET:
        raise OllamaError("terminal record has no created_at", transient=False)

    done_reason = _text_or_none(terminal.done_reason)
    try:
        return OllamaResponse(
            parts=parts,
            usage=Usage(
                input=_none(terminal.prompt_eval_count),
                output=_none(terminal.eval_count),
            ),
            stop_reason=_stop_reason(done_reason, parts),
            provider="ollama",
            model=terminal.model,
            id=None,
            created_at=terminal.created_at,
            timings=OllamaTimings(
                total_ns=_none(terminal.total_duration),
                load_ns=_none(terminal.load_duration),
                prompt_eval_ns=_none(terminal.prompt_eval_duration),
                eval_ns=_none(terminal.eval_duration),
            ),
            done_reason=done_reason,
            prompt_eval_cached_count=_none(terminal.prompt_eval_cached_count),
            logprobs=logprobs,
            remote_model=remote_model or _text_or_none(terminal.remote_model),
            remote_host=remote_host or _text_or_none(terminal.remote_host),
        )
    except (TypeError, ValueError) as error:
        raise OllamaError("invalid chat response", transient=False) from error


def encode_embed_request(
    model: str,
    inputs: tuple[str, ...],
    dimensions: int | None,
    truncate: bool,
    config: OllamaEmbeddingConfig,
    /,
) -> EmbedRequestWire:
    options = _options(config.options)
    return EmbedRequestWire(
        model=model,
        input=inputs,
        truncate=truncate,
        keep_alive=_unset(config.keep_alive),
        dimensions=_unset(dimensions),
        options=UNSET if not options else options,
    )


def decode_embed_response(response: EmbedResponseWire) -> OllamaEmbeddings:
    if not response.embeddings:
        raise OllamaError("embedding response is empty", transient=False)

    try:
        values = tuple(Embedding(vector) for vector in response.embeddings)
        return OllamaEmbeddings(
            values,
            usage=Usage(input=_none(response.prompt_eval_count)),
            timings=OllamaTimings(
                total_ns=_none(response.total_duration),
                load_ns=_none(response.load_duration),
            ),
            model=response.model,
        )
    except (TypeError, ValueError) as error:
        raise OllamaError("invalid embedding response", transient=False) from error


def decode_models(response: ListResponseWire) -> tuple[OllamaModelSummary, ...]:
    return tuple(
        OllamaModelSummary(
            name=model.name,
            model=model.model,
            modified_at=model.modified_at,
            size=model.size,
            digest=model.digest,
            metadata=_metadata(model.details),
            capabilities=model.capabilities,
            remote_model=_text_or_none(model.remote_model),
            remote_host=_text_or_none(model.remote_host),
        )
        for model in response.models
    )


def decode_running(response: ProcessResponseWire) -> tuple[OllamaRunningModel, ...]:
    return tuple(
        OllamaRunningModel(
            name=model.name,
            model=model.model,
            size=model.size,
            digest=model.digest,
            metadata=_metadata(model.details),
            expires_at=model.expires_at,
            size_vram=model.size_vram,
            context_length=model.context_length,
        )
        for model in response.models
    )


def decode_show(model: str, response: ShowResponseWire, /) -> OllamaModelDetails:
    return OllamaModelDetails(
        model=model,
        metadata=_metadata(response.details),
        capabilities=response.capabilities,
        modified_at=_none(response.modified_at),
        requires=_text_or_none(response.requires),
        remote_model=_text_or_none(response.remote_model),
        remote_host=_text_or_none(response.remote_host),
        license=response.license,
        modelfile=response.modelfile,
        parameters=response.parameters,
        template=response.template,
        system=response.system,
        renderer=response.renderer,
        parser=response.parser,
        model_info_json=_raw(response.model_info, b"{}"),
        projector_info_json=_raw(response.projector_info, b"{}"),
        tensors_json=_raw(response.tensors, b"[]"),
        messages_json=_raw(response.messages, b"[]"),
    )


def decode_progress(response: ProgressWire) -> OllamaProgress:
    return OllamaProgress(
        response.status,
        digest=_text_or_none(response.digest),
        total=_none(response.total),
        completed=_none(response.completed),
    )


def encode_show(model: str, verbose: bool, /) -> ShowRequestWire:
    return ShowRequestWire(model, verbose=verbose)


def encode_copy(source: str, destination: str, /) -> CopyRequestWire:
    return CopyRequestWire(source, destination)


def encode_model(model: str, /) -> ModelRequestWire:
    return ModelRequestWire(model)


def encode_transfer(model: str, /, *, insecure: bool | None = None) -> TransferRequestWire:
    return TransferRequestWire(model, stream=False, insecure=_unset(insecure))


def _encode_history(item: Msg | Response | ToolResult, /) -> MessageWire:
    if isinstance(item, Msg):
        return _encode_msg(item)
    if isinstance(item, Response):
        return _encode_response(item)
    return _encode_tool_result(item)


def _encode_msg(message: Msg, /) -> MessageWire:
    if message.role not in {"system", "user", "assistant"}:
        raise UnsupportedFeatureError(f"unsupported Ollama message role {message.role!r}")

    content, images = _encode_content(message.parts)
    return MessageWire(
        role=message.role,
        content=content,
        images=UNSET if not images else images,
    )


def _encode_response(response: Response, /) -> MessageWire:
    text: list[str] = []
    thinking: list[str] = []
    images: list[bytes] = []
    calls: list[ToolCallWire] = []

    for part in response.parts:
        if isinstance(part, str):
            text.append(part)
        elif isinstance(part, Binary):
            images.append(_image(part))
        elif isinstance(part, ResourceRef):
            raise UnsupportedFeatureError("Ollama cannot fetch ResourceRef content")
        elif isinstance(part, Reasoning):
            if part.state is not None:
                raise UnsupportedFeatureError("Ollama cannot replay opaque reasoning state")
            if part.text is not None:
                thinking.append(part.text)
        elif isinstance(part, ToolCall):
            calls.append(_encode_tool_call(part, len(calls)))
        else:
            raise UnsupportedFeatureError(f"unsupported Ollama response part {type(part).__name__}")

    return MessageWire(
        role="assistant",
        content="".join(text),
        thinking=UNSET if not thinking else "".join(thinking),
        images=UNSET if not images else tuple(images),
        tool_calls=UNSET if not calls else tuple(calls),
    )


def _encode_tool_result(result: ToolResult, /) -> MessageWire:
    if result.parts and result.structured_json is not None:
        raise UnsupportedFeatureError("Ollama tool results cannot preserve text and JSON together")

    if result.parts:
        chunks: list[str] = []
        for part in result.parts:
            if not isinstance(part, str):
                raise UnsupportedFeatureError("Ollama tool results support text only")
            chunks.append(part)
        content = "".join(chunks)
    elif result.structured_json is not None:
        content = result.structured_json.decode("utf-8")
    else:
        content = ""

    return MessageWire(
        role="tool",
        content=content,
        tool_name=result.call.name,
        tool_call_id=_unset(result.call.call_id),
    )


def _encode_content(parts: tuple[str | Content, ...], /) -> tuple[str, tuple[bytes, ...]]:
    text: list[str] = []
    images: list[bytes] = []
    for part in parts:
        if isinstance(part, str):
            text.append(part)
        elif isinstance(part, Binary):
            images.append(_image(part))
        elif isinstance(part, ResourceRef):
            raise UnsupportedFeatureError("Ollama cannot fetch ResourceRef content")
        else:
            raise UnsupportedFeatureError(f"unsupported Ollama content {type(part).__name__}")
    return "".join(text), tuple(images)


def _image(value: Binary, /) -> bytes:
    media_type = value.media_type
    if media_type is not None and not media_type.startswith("image/"):
        raise UnsupportedFeatureError(f"Ollama image input does not support {media_type!r}")
    return value.data


def _encode_tools(request: Request, /) -> tuple[ToolWire, ...]:
    if request.tool_choice == "auto":
        return tuple(_encode_tool(tool) for tool in request.tools)
    if request.tool_choice == "required":
        raise UnsupportedFeatureError("Ollama does not support required tool choice")
    raise UnsupportedFeatureError("Ollama does not support specific tool choice")


def _encode_tool(spec: ToolSpec, /) -> ToolWire:
    schema = spec.input_schema.data
    if not schema.startswith(b"{"):
        raise UnsupportedFeatureError("Ollama tool input schema must be an object")
    return ToolWire(
        ToolFunctionWire(
            spec.name,
            Raw(schema),
            description=_unset(spec.description),
        ),
        "function",
    )


def _encode_tool_call(call: ToolCall, index: int, /) -> ToolCallWire:
    return ToolCallWire(
        ToolCallFunctionWire(index, call.name, Raw(call.arguments_json)),
        id=_unset(call.call_id),
    )


def _decode_message(message: MessageWire, /) -> tuple[Part, ...]:
    if message.role.lower() not in {"", "assistant"}:
        raise OllamaError("invalid assistant role", transient=False)

    parts: list[Part] = []
    if message.thinking is not UNSET and message.thinking:
        parts.append(Reasoning(message.thinking))
    if message.content:
        parts.append(message.content)
    if message.images is not UNSET:
        parts.extend(Binary(image) for image in message.images)
    if message.tool_calls is not UNSET:
        parts.extend(_decode_tool_call(call) for call in message.tool_calls)
    return tuple(parts)


def _decode_tool_call(call: ToolCallWire, /) -> ToolCall:
    try:
        return ToolCall(
            call.function.name,
            bytes(call.function.arguments),
            call_id=_text_or_none(call.id),
        )
    except (TypeError, ValueError) as error:
        raise OllamaError("invalid tool call", transient=False) from error


def _decode_logprob(value: LogprobWire, /) -> OllamaLogprob:
    return OllamaLogprob(
        _decode_token(value),
        tuple(_decode_token(candidate) for candidate in value.top_logprobs),
    )


def _decode_token(value: TokenLogprobWire, /) -> OllamaTokenLogprob:
    try:
        token_bytes = bytes(value.bytes)
    except ValueError as error:
        raise OllamaError("invalid token bytes", transient=False) from error
    return OllamaTokenLogprob(value.token, value.logprob, token_bytes)


def _metadata(value: ModelMetadataWire, /) -> OllamaModelMetadata:
    return OllamaModelMetadata(
        parent_model=value.parent_model,
        format=value.format,
        family=value.family,
        families=value.families,
        parameter_size=value.parameter_size,
        quantization_level=value.quantization_level,
        context_length=_none(value.context_length),
        embedding_length=_none(value.embedding_length),
    )


def _options(
    value: OllamaOptions,
    /,
    *,
    max_output_tokens: int | None = None,
) -> dict[str, object]:
    if max_output_tokens is not None and value.num_predict is not None:
        raise ValueError("max_output_tokens and num_predict are mutually exclusive")

    options: dict[str, object] = {}
    for name in (
        "num_ctx",
        "num_batch",
        "num_gpu",
        "main_gpu",
        "use_mmap",
        "num_thread",
        "draft_num_predict",
        "num_keep",
        "seed",
        "num_predict",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "repeat_last_n",
        "temperature",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        current = getattr(value, name)
        if current is not None:
            options[name] = current

    if value.stop:
        options["stop"] = value.stop
    if max_output_tokens is not None:
        options["num_predict"] = max_output_tokens
    return options


def _stop_reason(done_reason: str | None, parts: tuple[Part, ...], /) -> StopReason:
    if any(isinstance(part, ToolCall) for part in parts):
        return "tool_call"
    if done_reason == "stop":
        return "stop"
    if done_reason == "length":
        return "length"
    return "other"


def _unset[T](value: T | None, /) -> T | UnsetType:
    return UNSET if value is None else value


def _none[T](value: T | UnsetType, /) -> T | None:
    return None if value is UNSET else value


def _text_or_none(value: str | UnsetType, /) -> str | None:
    return None if value is UNSET or value == "" else value


def _raw(value: Raw | UnsetType, default: bytes, /) -> bytes:
    return default if value is UNSET else bytes(value)
