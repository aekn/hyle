import json

import httpx
import pytest

from hyle_ollama import OllamaClient, OllamaError


@pytest.mark.asyncio
async def test_borrowed_http_client_remains_caller_owned() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    client = OllamaClient(http=http)
    await client.aclose()

    assert client.closed
    assert not http.is_closed
    await http.aclose()


def test_api_key_and_borrowed_http_are_mutually_exclusive() -> None:
    http = httpx.AsyncClient()
    try:
        with pytest.raises(ValueError, match="api_key and http are mutually exclusive"):
            OllamaClient(api_key="secret", http=http)
    finally:
        import asyncio

        asyncio.run(http.aclose())


def test_host_is_normalized_and_path_is_rejected() -> None:
    client = OllamaClient("localhost:11434")
    try:
        assert client.host == "http://localhost:11434"
    finally:
        import asyncio

        asyncio.run(client.aclose())

    with pytest.raises(ValueError, match="must not include a path"):
        OllamaClient("http://localhost:11434/api")


@pytest.mark.asyncio
async def test_http_error_uses_provider_message_and_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "server busy"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(http=http)
        with pytest.raises(OllamaError, match="server busy") as captured:
            await client.version()

    assert captured.value.status_code == 429
    assert captured.value.transient is True


@pytest.mark.asyncio
async def test_invalid_json_response_is_protocol_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(OllamaError, match="invalid JSON response"):
            await OllamaClient(http=http).version()


@pytest.mark.asyncio
async def test_request_and_response_bounds_are_enforced() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"version": "x" * 100})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(http=http, max_response_bytes=8)
        with pytest.raises(OllamaError, match="response too large"):
            await client.version()

    async def should_not_send(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request reached transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(should_not_send)) as http:
        client = OllamaClient(http=http, max_request_bytes=4)
        with pytest.raises(OllamaError, match="request too large"):
            await client.delete("model")


@pytest.mark.asyncio
async def test_model_management_requests_are_native_and_typed() -> None:
    seen: list[tuple[str, str, object | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body: object | None = None
        if request.content:
            body = json.loads(request.content)
        seen.append((request.method, request.url.path, body))

        match request.url.path:
            case "/api/tags":
                return httpx.Response(200, json={"models": []})
            case "/api/ps":
                return httpx.Response(200, json={"models": []})
            case "/api/show":
                return httpx.Response(200, json={"details": {}, "model_info": {}})
            case "/api/copy" | "/api/delete":
                return httpx.Response(200)
            case "/api/pull" | "/api/push":
                return httpx.Response(200, json={"status": "success"})
            case "/api/version":
                return httpx.Response(200, json={"version": "0.14.0"})
            case _:
                return httpx.Response(404, json={"error": "not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(http=http)
        assert await client.models() == ()
        assert await client.running() == ()
        details = await client.show("qwen3", verbose=True)
        assert details.model == "qwen3"
        await client.copy("qwen3", "qwen3-copy")
        await client.delete("qwen3-copy")
        assert (await client.pull("qwen3")).status == "success"
        assert (await client.push("user/model", insecure=True)).status == "success"
        assert await client.version() == "0.14.0"

    assert ("POST", "/api/pull", {"model": "qwen3", "stream": False}) in seen
    assert (
        "POST",
        "/api/push",
        {"model": "user/model", "stream": False, "insecure": True},
    ) in seen


def test_bare_cloud_host_uses_https() -> None:
    client = OllamaClient("ollama.com")
    try:
        assert client.host == "https://ollama.com"
    finally:
        import asyncio

        asyncio.run(client.aclose())


def test_api_key_rejects_cleartext_host() -> None:
    with pytest.raises(ValueError, match="api_key requires HTTPS"):
        OllamaClient("http://ollama.com", api_key="secret")


@pytest.mark.asyncio
async def test_json_operations_advertise_json_not_ndjson() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/json"
        return httpx.Response(200, json={"version": "0.14.0"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await OllamaClient(http=http).version() == "0.14.0"


@pytest.mark.asyncio
async def test_model_metadata_and_remote_fields_decode() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3:4b",
                            "model": "qwen3:4b",
                            "modified_at": "2026-09-02T12:00:00Z",
                            "size": 42,
                            "digest": "sha256:test",
                            "details": {
                                "family": "qwen3",
                                "families": ["qwen3"],
                                "parameter_size": "4B",
                                "quantization_level": "Q4_K_M",
                                "context_length": 32768,
                            },
                            "capabilities": ["completion", "tools"],
                            "remote_model": "qwen3:4b",
                            "remote_host": "https://ollama.com",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        (model,) = await OllamaClient(http=http).models()

    assert model.metadata.family == "qwen3"
    assert model.metadata.context_length == 32768
    assert model.capabilities == ("completion", "tools")
    assert model.remote_host == "https://ollama.com"


@pytest.mark.asyncio
async def test_show_tolerates_cloud_response_without_model_info() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"details": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        details = await OllamaClient(http=http).show("cloud-model")

    assert details.model == "cloud-model"
    assert details.model_info_json == b"{}"
