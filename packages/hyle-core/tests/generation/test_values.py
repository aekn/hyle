from typing import Any

import pytest

from hyle.content import Binary
from hyle.generation import Msg, PartBuffer, Reasoning, Request, Response
from hyle.schema import JsonSchema
from hyle.tools import ToolCall, ToolResult
from hyle.usage import Usage


def test_msg_uses_plain_strings() -> None:
    image = Binary(b"x", media_type="image/png")
    msg = Msg.user("hello", image)
    assert msg.role == "user"
    assert msg.parts == ("hello", image)


def test_response_is_replayable_model_turn() -> None:
    call = ToolCall("lookup", call_id="1")
    response = Response(
        parts=(Reasoning("checking"), "answer", call),
        usage=Usage(4, 2),
        stop_reason="tool_call",
        provider="test",
        model="model",
        id="response-1",
    )
    assert response.text == "answer"
    assert response.require_text() == "answer"
    assert response.tool_calls == (call,)


def test_response_require_text_rejects_blank_or_missing() -> None:
    for parts in ((), ("   ",), (Binary(b"x"),)):
        response = Response(parts=parts, usage=Usage(), stop_reason="stop")
        with pytest.raises(ValueError, match="non-blank text"):
            response.require_text()


def test_reasoning_state_is_scoped_and_owned() -> None:
    state = bytearray(b"state")
    reasoning = Reasoning("thinking", state=state, scope="provider")
    state[:] = b"other"
    assert reasoning.state == b"state"
    assert reasoning.scope == "provider"


def test_reasoning_requires_state_and_scope_together() -> None:
    with pytest.raises(ValueError, match="together"):
        Reasoning(state=b"state")


def test_part_buffer_merges_adjacent_and_keyed_fragments() -> None:
    buffer = PartBuffer()
    buffer.append("a")
    buffer.append("b")
    buffer.append("A", key="left")
    buffer.append("B", key="right")
    buffer.append("1", key="left")
    buffer.append("2", key="right")
    assert buffer.finish() == ("ab", "A1", "B2")


def test_request_normalizes_string_and_instructions() -> None:
    request = Request("hello", instructions="be concise", max_output_tokens=64)
    assert request.input == (Msg.system("be concise"), Msg.user("hello"))
    assert request.max_output_tokens == 64


def test_request_accepts_semantic_history_directly() -> None:
    first = Response(parts=("answer",), usage=Usage(1, 1), stop_reason="stop")
    call = ToolCall("lookup", call_id="1")
    result = ToolResult(call, "ok")
    request = Request((Msg.user("question"), first, result))
    assert request.input == (Msg.user("question"), first, result)


def test_public_generation_values_reject_invalid_runtime_parts() -> None:
    invalid: Any = object()

    with pytest.raises(TypeError, match="message parts"):
        Msg.user(invalid)

    with pytest.raises(TypeError, match="response parts"):
        Response(parts=(invalid,), usage=Usage(), stop_reason="stop")

    with pytest.raises(TypeError, match="request input"):
        Request([invalid])

    call = ToolCall("lookup")
    with pytest.raises(TypeError, match="tool result parts"):
        ToolResult(call, invalid)


def test_generation_metadata_rejects_invalid_runtime_values() -> None:
    invalid: Any = object()

    with pytest.raises(TypeError, match="instructions"):
        Request("hello", instructions=invalid)

    with pytest.raises(TypeError, match="schema"):
        Request("hello", schema=invalid)

    with pytest.raises(TypeError, match="usage"):
        Response(parts=("x",), usage=invalid, stop_reason="stop")

    with pytest.raises(ValueError, match="stop reason"):
        Response(parts=("x",), usage=Usage(), stop_reason=invalid)

    schema = JsonSchema({"type": "object"})
    assert Request("hello", schema=schema).schema is schema
