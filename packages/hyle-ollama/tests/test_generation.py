import json

import httpx
import pytest

from hyle.content import Binary, ResourceRef
from hyle.errors import UnsupportedFeatureError
from hyle.generation import Msg, Reasoning, Request, Response
from hyle.schema import JsonSchema
from hyle.tools import ToolCall, ToolResult, ToolSpec
from hyle.usage import Usage
from hyle_ollama import (
    OllamaClient,
    OllamaError,
    OllamaGenerationConfig,
    OllamaModel,
    OllamaOptions,
)


def _response(message: dict[str, object], *, done_reason: str = "stop") -> dict[str, object]:
    return {
        "model": "qwen3",
        "created_at": "2026-09-02T12:00:00Z",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 7,
        "prompt_eval_cached_count": 2,
        "eval_count": 3,
        "total_duration": 100,
        "load_duration": 10,
    }


@pytest.mark.asyncio
async def test_generation_translates_native_controls_without_merging() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "keep_alive": "5m",
            "options": {"num_ctx": 16384, "temperature": 0.0},
            "think": "high",
            "truncate": False,
            "shift": True,
            "logprobs": True,
            "top_logprobs": 3,
        }
        return httpx.Response(200, json=_response({"role": "assistant", "content": "hi"}))

    config = OllamaGenerationConfig(
        think="high",
        keep_alive="5m",
        truncate=False,
        shift=True,
        logprobs=True,
        top_logprobs=3,
        options=OllamaOptions(num_ctx=16_384, temperature=0.0),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = OllamaModel("qwen3", client=OllamaClient(http=http), config=config)
        response = await model.generate(Request("hello"))

    assert response.require_text() == "hi"
    assert response.usage == Usage(7, 3)
    assert response.prompt_eval_cached_count == 2
    assert response.timings.total_ns == 100


@pytest.mark.asyncio
async def test_images_reasoning_tool_ids_and_tool_results_round_trip() -> None:
    call = ToolCall("lookup", '{"query":"x"}', call_id="call-1")
    prior = Response(
        parts=(Reasoning("checking"), "calling", call),
        usage=Usage(1, 1),
        stop_reason="tool_call",
    )
    result = ToolResult(call, "found")

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "user", "content": "look", "images": ["aW1n"]},
            {
                "role": "assistant",
                "content": "calling",
                "thinking": "checking",
                "tool_calls": [
                    {
                        "function": {
                            "index": 0,
                            "name": "lookup",
                            "arguments": {"query": "x"},
                        },
                        "id": "call-1",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "found",
                "tool_name": "lookup",
                "tool_call_id": "call-1",
            },
        ]
        return httpx.Response(
            200,
            json=_response(
                {
                    "role": "assistant",
                    "thinking": "done thinking",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "function": {
                                "index": 0,
                                "name": "lookup",
                                "arguments": {"query": "y"},
                            },
                        }
                    ],
                }
            ),
        )

    history = (
        Msg.user("look", Binary(b"img", media_type="image/png")),
        prior,
        result,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        response = await OllamaModel("qwen3", client=OllamaClient(http=http)).generate(
            Request(history)
        )

    assert response.parts[0] == Reasoning("done thinking")
    assert response.tool_calls == (ToolCall("lookup", '{"query":"y"}', call_id="call-2"),)
    assert response.stop_reason == "tool_call"


@pytest.mark.asyncio
async def test_tools_and_structured_output_use_raw_native_json() -> None:
    spec = ToolSpec(
        name="lookup",
        input_schema=JsonSchema(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        ),
    )
    schema = JsonSchema({"type": "object", "properties": {"answer": {"type": "string"}}})

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["parameters"]["type"] == "object"
        assert body["format"]["properties"]["answer"]["type"] == "string"
        return httpx.Response(200, json=_response({"role": "assistant", "content": "{}"}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = OllamaModel("qwen3", client=OllamaClient(http=http))
        await model.generate(Request("answer", tools=(spec,), schema=schema))


@pytest.mark.asyncio
async def test_tool_schema_requires_object_root() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    spec = ToolSpec(name="lookup", input_schema=JsonSchema(True))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = OllamaModel("qwen3", client=OllamaClient(http=http))
        with pytest.raises(UnsupportedFeatureError, match="tool input schema must be an object"):
            await model.generate(Request("x", tools=(spec,)))

    assert calls == 0


@pytest.mark.asyncio
async def test_unsupported_semantics_are_rejected_before_transport() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    spec = ToolSpec(name="lookup", input_schema=JsonSchema({"type": "object"}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = OllamaModel("qwen3", client=OllamaClient(http=http))
        with pytest.raises(UnsupportedFeatureError, match="required tool choice"):
            await model.generate(Request("x", tools=(spec,), tool_choice="required"))
        with pytest.raises(UnsupportedFeatureError, match="ResourceRef"):
            await model.generate(Request(Msg.user(ResourceRef("https://example.com/a"))))
        with pytest.raises(UnsupportedFeatureError, match="opaque reasoning state"):
            await model.generate(
                Request(
                    Response(
                        parts=(Reasoning(state=b"state", scope="other"),),
                        usage=Usage(),
                        stop_reason="stop",
                    )
                )
            )

    assert calls == 0


@pytest.mark.asyncio
async def test_max_output_tokens_and_num_predict_have_no_precedence_rule() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    config = OllamaGenerationConfig(options=OllamaOptions(num_predict=32))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = OllamaModel("qwen3", client=OllamaClient(http=http), config=config)
        with pytest.raises(ValueError, match="mutually exclusive"):
            await model.generate(Request("x", max_output_tokens=64))

    assert calls == 0


@pytest.mark.asyncio
async def test_direct_cloud_structured_output_is_rejected() -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    try:
        client = OllamaClient("https://ollama.com", http=http)
        model = OllamaModel("gpt-oss:120b", client=client)
        with pytest.raises(UnsupportedFeatureError, match="Cloud"):
            await model.generate(Request("x", schema=JsonSchema({"type": "object"})))
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_assistant_role_is_normalized_like_ollama() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_response({"role": "ASSISTANT", "content": "ok"}),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        response = await OllamaModel("qwen3", client=OllamaClient(http=http)).generate(
            Request("hello")
        )

    assert response.require_text() == "ok"


@pytest.mark.asyncio
async def test_malformed_chat_response_is_provider_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3",
                "created_at": "2026-09-02T12:00:00Z",
                "done": True,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        model = OllamaModel("qwen3", client=OllamaClient(http=http))
        with pytest.raises(OllamaError, match="chat response has no message"):
            await model.generate(Request("hello"))
