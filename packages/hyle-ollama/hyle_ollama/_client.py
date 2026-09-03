__all__ = ("OllamaClient",)

import os
from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Self

import httpx
import msgspec

from hyle_ollama._errors import OllamaError
from hyle_ollama._response import (
    OllamaModelDetails,
    OllamaModelSummary,
    OllamaProgress,
    OllamaRunningModel,
)
from hyle_ollama._translate import (
    decode_models,
    decode_progress,
    decode_running,
    decode_show,
    encode_copy,
    encode_model,
    encode_show,
    encode_transfer,
)
from hyle_ollama._validate import boolean, positive_int, text
from hyle_ollama._wire import (
    ChatRecordWire,
    ChatRequestWire,
    EmbedRequestWire,
    EmbedResponseWire,
    ErrorWire,
    ListResponseWire,
    ProcessResponseWire,
    ProgressWire,
    ShowResponseWire,
    VersionWire,
)

_MIB = 1024 * 1024
_DEFAULT_HOST = "http://localhost:11434"
_MAX_ERROR_BYTES = 64 * 1024
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_ENCODER = msgspec.json.Encoder()
_CHAT_DECODER = msgspec.json.Decoder(ChatRecordWire)
_EMBED_DECODER = msgspec.json.Decoder(EmbedResponseWire)
_LIST_DECODER = msgspec.json.Decoder(ListResponseWire)
_PROCESS_DECODER = msgspec.json.Decoder(ProcessResponseWire)
_SHOW_DECODER = msgspec.json.Decoder(ShowResponseWire)
_PROGRESS_DECODER = msgspec.json.Decoder(ProgressWire)
_VERSION_DECODER = msgspec.json.Decoder(VersionWire)
_ERROR_DECODER = msgspec.json.Decoder(ErrorWire)


class OllamaClient:
    __slots__ = (
        "_closed",
        "_host",
        "_http",
        "_max_record_bytes",
        "_max_request_bytes",
        "_max_response_bytes",
        "_max_stream_bytes",
        "_owns_http",
    )

    def __init__(
        self,
        host: str | None = None,
        /,
        *,
        api_key: str | None = None,
        http: httpx.AsyncClient | None = None,
        max_request_bytes: int = 64 * _MIB,
        max_response_bytes: int = 64 * _MIB,
        max_stream_bytes: int = 256 * _MIB,
        max_record_bytes: int = 8 * _MIB,
    ) -> None:
        base = _host(host or os.getenv("OLLAMA_HOST") or _DEFAULT_HOST)
        if api_key is None and base.host == "ollama.com" and http is None:
            api_key = os.getenv("OLLAMA_API_KEY")
        if api_key is not None:
            api_key = text(api_key, "api_key", blank=False).strip()
        if api_key is not None and http is not None:
            raise ValueError("api_key and http are mutually exclusive")
        if api_key is not None and base.scheme != "https":
            raise ValueError("api_key requires HTTPS")

        self._host = base
        self._max_request_bytes = positive_int(max_request_bytes, "max_request_bytes")
        self._max_response_bytes = positive_int(max_response_bytes, "max_response_bytes")
        self._max_stream_bytes = positive_int(max_stream_bytes, "max_stream_bytes")
        self._max_record_bytes = positive_int(max_record_bytes, "max_record_bytes")
        self._closed = False

        if http is None:
            headers = {"User-Agent": "hyle-ollama"}
            if api_key is not None:
                headers["Authorization"] = f"Bearer {api_key}"
            self._http = httpx.AsyncClient(
                follow_redirects=True,
                timeout=None,
                headers=headers,
            )
            self._owns_http = True
        else:
            self._http = http
            self._owns_http = False

    @property
    def host(self) -> str:
        return str(self._host).rstrip("/")

    @property
    def http(self) -> httpx.AsyncClient:
        return self._http

    @property
    def max_request_bytes(self) -> int:
        return self._max_request_bytes

    @property
    def max_response_bytes(self) -> int:
        return self._max_response_bytes

    @property
    def max_stream_bytes(self) -> int:
        return self._max_stream_bytes

    @property
    def max_record_bytes(self) -> int:
        return self._max_record_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeError("client is closed")
        if self._http.is_closed:
            raise RuntimeError("http client is closed")
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_http:
            await self._http.aclose()

    async def models(self) -> tuple[OllamaModelSummary, ...]:
        response = await _request_json(
            self,
            "GET",
            "/api/tags",
            None,
            _LIST_DECODER,
        )
        return decode_models(response)

    async def running(self) -> tuple[OllamaRunningModel, ...]:
        response = await _request_json(
            self,
            "GET",
            "/api/ps",
            None,
            _PROCESS_DECODER,
        )
        return decode_running(response)

    async def show(self, model: str, /, *, verbose: bool = False) -> OllamaModelDetails:
        model = _model(model)
        boolean(verbose, "verbose")
        response = await _request_json(
            self,
            "POST",
            "/api/show",
            encode_show(model, verbose),
            _SHOW_DECODER,
        )
        return decode_show(model, response)

    async def copy(self, source: str, destination: str, /) -> None:
        await _request_empty(
            self,
            "POST",
            "/api/copy",
            encode_copy(_model(source), _model(destination)),
        )

    async def delete(self, model: str, /) -> None:
        await _request_empty(
            self,
            "DELETE",
            "/api/delete",
            encode_model(_model(model)),
        )

    async def pull(self, model: str, /) -> OllamaProgress:
        response = await _request_json(
            self,
            "POST",
            "/api/pull",
            encode_transfer(_model(model)),
            _PROGRESS_DECODER,
        )
        return decode_progress(response)

    async def push(self, model: str, /, *, insecure: bool = False) -> OllamaProgress:
        boolean(insecure, "insecure")
        response = await _request_json(
            self,
            "POST",
            "/api/push",
            encode_transfer(_model(model), insecure=insecure),
            _PROGRESS_DECODER,
        )
        return decode_progress(response)

    async def version(self) -> str:
        response = await _request_json(
            self,
            "GET",
            "/api/version",
            None,
            _VERSION_DECODER,
        )
        return response.version


async def send_chat(client: OllamaClient, request: ChatRequestWire, /) -> ChatRecordWire:
    return await _request_json(
        client,
        "POST",
        "/api/chat",
        request,
        _CHAT_DECODER,
    )


async def send_embed(client: OllamaClient, request: EmbedRequestWire, /) -> EmbedResponseWire:
    return await _request_json(
        client,
        "POST",
        "/api/embed",
        request,
        _EMBED_DECODER,
    )


async def open_chat_stream(client: OllamaClient, request: ChatRequestWire, /) -> httpx.Response:
    body = _encode(client, request)
    response = await _send(client, "POST", "/api/chat", body, ndjson=True)
    if response.is_error:
        try:
            await _raise_status(response)
        finally:
            await response.aclose()
    return response


async def iter_ndjson(
    client: OllamaClient,
    response: httpx.Response,
    /,
) -> AsyncGenerator[bytes]:
    total = 0
    buffer = bytearray()

    try:
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > client.max_stream_bytes:
                raise OllamaError("stream too large", transient=False)
            buffer.extend(chunk)

            while True:
                newline = buffer.find(b"\n")
                if newline < 0:
                    if len(buffer) > client.max_record_bytes:
                        raise OllamaError("stream record too large", transient=False)
                    break

                record = bytes(buffer[:newline])
                del buffer[: newline + 1]
                if record.endswith(b"\r"):
                    record = record[:-1]
                if not record:
                    continue
                if len(record) > client.max_record_bytes:
                    raise OllamaError("stream record too large", transient=False)
                yield record

        if buffer:
            if len(buffer) > client.max_record_bytes:
                raise OllamaError("stream record too large", transient=False)
            record = bytes(buffer[:-1] if buffer.endswith(b"\r") else buffer)
            if record:
                yield record
    except httpx.TimeoutException as error:
        raise OllamaError("response timed out", transient=True) from error
    except httpx.RequestError as error:
        raise OllamaError("transport failed", transient=True) from error


def is_direct_cloud(client: OllamaClient, /) -> bool:
    return httpx.URL(client.host).host == "ollama.com"


async def _request_json[T](
    client: OllamaClient,
    method: str,
    path: str,
    request: object | None,
    decoder: msgspec.json.Decoder[T],
    /,
) -> T:
    body = None if request is None else _encode(client, request)
    response = await _send(client, method, path, body, ndjson=False)
    try:
        if response.is_error:
            await _raise_status(response)
        data = await _read(response, client.max_response_bytes)
    finally:
        await response.aclose()

    try:
        return decoder.decode(data)
    except msgspec.DecodeError as error:
        raise OllamaError("invalid JSON response", transient=False) from error


async def _request_empty(
    client: OllamaClient,
    method: str,
    path: str,
    request: object,
    /,
) -> None:
    response = await _send(client, method, path, _encode(client, request), ndjson=False)
    try:
        if response.is_error:
            await _raise_status(response)
        await _read(response, client.max_response_bytes)
    finally:
        await response.aclose()


def _encode(client: OllamaClient, value: object, /) -> bytes:
    try:
        data = _ENCODER.encode(value)
    except (msgspec.EncodeError, TypeError, ValueError) as error:
        raise OllamaError("invalid request", transient=False) from error
    if len(data) > client.max_request_bytes:
        raise OllamaError("request too large", transient=False)
    return data


async def _send(
    client: OllamaClient,
    method: str,
    path: str,
    content: bytes | None,
    /,
    *,
    ndjson: bool,
) -> httpx.Response:
    if client.closed:
        raise RuntimeError("client is closed")
    if client.http.is_closed:
        raise RuntimeError("http client is closed")
    headers = {"Accept": "application/x-ndjson" if ndjson else "application/json"}
    if content is not None:
        headers["Content-Type"] = "application/json"
    request = client.http.build_request(
        method,
        httpx.URL(client.host).copy_with(path=path),
        content=content,
        headers=headers,
    )
    try:
        return await client.http.send(request, stream=True)
    except httpx.ConnectError as error:
        raise OllamaError("connection failed", transient=True) from error
    except httpx.TimeoutException as error:
        raise OllamaError("request timed out", transient=True) from error
    except httpx.RequestError as error:
        raise OllamaError("transport failed", transient=True) from error


async def _read(
    response: httpx.Response,
    limit: int,
    /,
    *,
    too_large: str = "response too large",
    status_code: int | None = None,
) -> bytes:
    data = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            if len(data) + len(chunk) > limit:
                raise OllamaError(
                    too_large,
                    status_code=status_code,
                    transient=(
                        status_code in _TRANSIENT_STATUS if status_code is not None else False
                    ),
                )
            data.extend(chunk)
    except httpx.TimeoutException as error:
        raise OllamaError(
            "response timed out",
            status_code=status_code,
            transient=True,
        ) from error
    except httpx.RequestError as error:
        raise OllamaError(
            "transport failed",
            status_code=status_code,
            transient=True,
        ) from error
    return bytes(data)


async def _raise_status(response: httpx.Response, /) -> None:
    transient = response.status_code in _TRANSIENT_STATUS
    body = await _read(
        response,
        _MAX_ERROR_BYTES,
        too_large="error response too large",
        status_code=response.status_code,
    )

    message = ""
    if body:
        try:
            decoded = _ERROR_DECODER.decode(body)
        except msgspec.DecodeError:
            try:
                message = body.decode("utf-8").strip()
            except UnicodeDecodeError:
                message = ""
        else:
            message = decoded.error.strip()
            if not message and decoded.signin_url:
                message = "authentication required"

    if not message:
        message = response.reason_phrase.lower() or f"http {response.status_code}"
    elif len(message) > 1024:
        message = f"{message[:1021]}..."

    raise OllamaError(
        message,
        status_code=response.status_code,
        transient=transient,
    )


def _host(value: str, /) -> httpx.URL:
    value = text(value, "host", blank=False).strip()
    if "://" not in value:
        scheme = "https" if value.rstrip("/") == "ollama.com" else "http"
        value = f"{scheme}://{value}"

    try:
        url = httpx.URL(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid host") from error

    if url.scheme not in {"http", "https"} or not url.host:
        raise ValueError("host must be an absolute HTTP URL")
    if url.query or url.fragment:
        raise ValueError("host must not include a query or fragment")
    if url.path not in {"", "/"}:
        raise ValueError("host must not include a path")
    return url.copy_with(path="/")


def _model(value: str, /) -> str:
    return text(value, "model", blank=False).strip()
