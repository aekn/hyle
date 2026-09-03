from datetime import UTC, datetime

import msgspec

from hyle_ollama._wire import (
    ChatRequestWire,
    EmbedRequestWire,
    MessageWire,
    ModelMetadataWire,
    ModelSummaryWire,
    ToolFunctionWire,
    ToolWire,
    TransferRequestWire,
)


def test_shared_model_metadata_default_is_immutable() -> None:
    first = ModelSummaryWire(
        "a",
        "a",
        datetime(2026, 9, 2, 12, tzinfo=UTC),
        1,
        "digest",
    )
    second = ModelSummaryWire(
        "b",
        "b",
        first.modified_at,
        2,
        "digest",
    )

    assert first.details is second.details
    assert isinstance(first.details, ModelMetadataWire)


def test_unset_fields_are_omitted_but_false_and_zero_are_preserved() -> None:
    request = ChatRequestWire(
        "qwen3",
        (MessageWire("user", "hello"),),
        stream=False,
        truncate=False,
        shift=False,
        options={"temperature": 0.0, "main_gpu": 0, "use_mmap": False},
    )
    encoded = msgspec.json.encode(request)
    decoded = msgspec.json.decode(encoded)

    assert decoded["stream"] is False
    assert decoded["truncate"] is False
    assert decoded["shift"] is False
    assert decoded["options"] == {"temperature": 0.0, "main_gpu": 0, "use_mmap": False}
    assert "think" not in decoded


def test_embed_request_preserves_explicit_false() -> None:
    request = EmbedRequestWire("embeddinggemma", ("hello",), False)
    decoded = msgspec.json.decode(msgspec.json.encode(request))
    assert decoded == {"model": "embeddinggemma", "input": ["hello"], "truncate": False}


def test_transfer_request_preserves_explicit_non_streaming() -> None:
    request = TransferRequestWire("qwen3", False)
    assert msgspec.json.decode(msgspec.json.encode(request)) == {
        "model": "qwen3",
        "stream": False,
    }


def test_tool_type_is_not_omitted() -> None:
    request = ToolWire(
        ToolFunctionWire("lookup", msgspec.Raw(b'{"type":"object"}')),
        "function",
    )
    decoded = msgspec.json.decode(msgspec.json.encode(request))
    assert decoded["type"] == "function"
