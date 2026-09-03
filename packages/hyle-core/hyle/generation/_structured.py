__all__ = ("Structured",)

from typing import final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypeForm

from hyle._json import validate_json
from hyle.generation._errors import StructuredError
from hyle.generation._response import Response
from hyle.schema import JsonSchema


@final
class Structured[OutputT]:
    __slots__ = ("_adapter", "_schema", "_strict")

    def __init__(self, output_type: TypeForm[OutputT], /, *, strict: bool = True) -> None:
        adapter: TypeAdapter[OutputT] = TypeAdapter(output_type)
        self._adapter = adapter
        self._schema = JsonSchema(adapter.json_schema(mode="validation"))
        self._strict = strict

    @property
    def schema(self) -> JsonSchema:
        return self._schema

    @property
    def strict(self) -> bool:
        return self._strict

    def __call__(self, response: Response, /) -> OutputT:
        try:
            text = response.require_text()
        except ValueError as error:
            raise StructuredError(str(error), kind="text") from None

        try:
            validate_json(text)
        except ValueError as error:
            raise StructuredError(str(error), kind="json") from None

        try:
            return self._adapter.validate_json(text, strict=self._strict)
        except ValidationError as error:
            raise StructuredError(
                f"structured response failed validation ({error.error_count()} errors)",
                kind="validation",
            ) from None
