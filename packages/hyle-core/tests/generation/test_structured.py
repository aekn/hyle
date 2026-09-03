from dataclasses import dataclass

import pytest

from hyle.generation import Response, Structured, StructuredError
from hyle.usage import Usage


@dataclass
class Answer:
    value: int


def _response(text: str) -> Response:
    return Response(parts=(text,), usage=Usage(), stop_reason="stop")


def test_structured_owns_schema_and_validation() -> None:
    output = Structured(Answer)
    assert b'"value"' in output.schema.data
    assert output(_response('{"value":42}')) == Answer(42)


def test_structured_rejects_non_json_and_invalid_values() -> None:
    output = Structured(Answer)
    with pytest.raises(StructuredError):
        output(_response("not json"))
    with pytest.raises(StructuredError):
        output(_response('{"value":"no"}'))
