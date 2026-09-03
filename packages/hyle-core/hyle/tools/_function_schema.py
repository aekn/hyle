__all__ = ("CompiledFunction", "compile_function")

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, Signature, getdoc, signature
from typing import (
    Any,
    TypeIs,
    final,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticSchemaGenerationError,
    TypeAdapter,
    ValidationError,
    create_model,
)

from hyle.content import Content
from hyle.schema import JsonSchema
from hyle.tools._errors import ToolArgumentError, ToolIssue, ToolOutputError
from hyle.tools._tool import ToolCall, ToolResult, ToolSpec, require_tool_name

_OBJECT_ADAPTER = TypeAdapter(object)
_RAW_BYTES = (bytes, bytearray, memoryview)


@final
@dataclass(frozen=True, slots=True)
class CompiledFunction:
    spec: ToolSpec
    arguments_model: type[BaseModel]
    result_adapter: TypeAdapter[Any]

    positional_only: tuple[str, ...]
    keyword: tuple[str, ...]

    validate_result: bool
    strict: bool

    def read_arguments(self, call: ToolCall, /) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if call.name != self.spec.name:
            raise ValueError(f"tool {self.spec.name!r} cannot execute a call for {call.name!r}")

        try:
            arguments = self.arguments_model.model_validate_json(
                call.arguments_json,
                strict=self.strict,
            )
        except ValidationError as error:
            raise ToolArgumentError(self.spec.name, _validation_issues(error)) from None

        provided = arguments.model_fields_set
        last_positional = -1

        for index, name in enumerate(self.positional_only):
            if name in provided:
                last_positional = index

        args = tuple(
            getattr(arguments, name) for name in self.positional_only[: last_positional + 1]
        )
        kwargs = {name: getattr(arguments, name) for name in self.keyword if name in provided}

        return args, kwargs

    def write_result(self, value: object, call: ToolCall, /) -> ToolResult:
        if self.validate_result:
            try:
                value = self.result_adapter.validate_python(value, strict=self.strict)
            except ValidationError as error:
                raise ToolOutputError(self.spec.name, _validation_issues(error)) from None

        return _tool_result(call, value, self.result_adapter)


def compile_function(
    function: Callable[..., object],
    /,
    *,
    name: str | None,
    description: str | None,
    strict: bool,
) -> CompiledFunction:
    tool_name = _tool_name(function, name)
    call_signature = _signature(function, tool_name)

    _reject_unbound_method(function, tool_name, call_signature)

    fields: dict[str, Any] = {}
    positional_only: list[str] = []
    keyword: list[str] = []

    for parameter in call_signature.parameters.values():
        _validate_parameter(tool_name, parameter)

        annotation = Any if parameter.annotation is Parameter.empty else parameter.annotation
        default = ... if parameter.default is Parameter.empty else parameter.default

        fields[parameter.name] = annotation, default

        if parameter.kind is Parameter.POSITIONAL_ONLY:
            positional_only.append(parameter.name)
        else:
            keyword.append(parameter.name)

    arguments_model = create_model(
        "ToolArguments",
        __config__=ConfigDict(
            cache_strings="keys",
            extra="forbid",
            hide_input_in_errors=True,
            loc_by_alias=True,
            protected_namespaces=(),
            validate_by_alias=True,
            validate_by_name=False,
        ),
        **fields,
    )

    input_document = arguments_model.model_json_schema(by_alias=True, mode="validation")
    input_document.pop("title", None)

    _strip_generated_parameter_titles(arguments_model, input_document)

    return_annotation = call_signature.return_annotation
    annotated_result = return_annotation is not Signature.empty
    if annotated_result:
        try:
            result_adapter = TypeAdapter(return_annotation)
            validate_result = True
        except PydanticSchemaGenerationError:
            result_adapter = _OBJECT_ADAPTER
            validate_result = False
    else:
        result_adapter = _OBJECT_ADAPTER
        validate_result = False

    return CompiledFunction(
        spec=ToolSpec(
            name=tool_name,
            input_schema=JsonSchema(input_document),
            description=_description(function, description),
        ),
        arguments_model=arguments_model,
        result_adapter=result_adapter,
        positional_only=tuple(positional_only),
        keyword=tuple(keyword),
        validate_result=validate_result,
        strict=strict,
    )


def _strip_generated_parameter_titles(
    arguments_model: type[BaseModel],
    document: dict[str, Any],
    /,
) -> None:
    properties = document.get("properties")

    if not _is_dict(properties):
        return

    for name, field in arguments_model.model_fields.items():
        if field.title is not None:
            continue

        alias = field.validation_alias
        visible_name = alias if isinstance(alias, str) else name
        property_schema = properties.get(visible_name)

        if _is_dict(property_schema):
            property_schema.pop("title", None)


def _signature(
    function: Callable[..., object],
    tool_name: str,
    /,
) -> Signature:
    try:
        return signature(function, eval_str=True)
    except (NameError, TypeError, ValueError) as error:
        raise TypeError(f"signature for tool {tool_name!r} could not be resolved") from error


def _reject_unbound_method(
    function: Callable[..., object],
    tool_name: str,
    call_signature: Signature,
    /,
) -> None:
    first = next(iter(call_signature.parameters.values()), None)

    if first is None or first.name not in {"self", "cls"}:
        return

    qualified_name = getattr(function, "__qualname__", "")
    owner, separator, _ = qualified_name.rpartition(".")

    if separator and not owner.endswith(".<locals>"):
        raise TypeError(
            f"tool {tool_name!r} appears to be an unbound method; "
            "pass a bound instance or class method"
        )


def _validate_parameter(tool_name: str, parameter: Parameter, /) -> None:
    if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
        raise TypeError(
            f"tool {tool_name!r} parameter {parameter.name!r} uses "
            f"{parameter.kind.description}; "
            "function tools require a fixed signature"
        )

    if parameter.name.startswith("_"):
        raise TypeError(
            f"tool {tool_name!r} parameter {parameter.name!r} begins with an underscore"
        )

    if hasattr(BaseModel, parameter.name):
        raise TypeError(
            f"tool {tool_name!r} parameter {parameter.name!r} conflicts with the Pydantic model API"
        )


def _tool_name(function: Callable[..., object], explicit: str | None, /) -> str:
    if explicit is not None:
        return require_tool_name(explicit)

    inferred = getattr(function, "__name__", None)

    if not isinstance(inferred, str):
        raise TypeError("callable tools without a function name require name=")

    return require_tool_name(inferred)


def _description(function: Callable[..., object], explicit: str | None, /) -> str | None:
    if explicit is not None:
        return explicit

    docstring = getdoc(function)

    if docstring is None:
        return None

    paragraph, _, _ = docstring.partition("\n\n")

    value = " ".join(line.strip() for line in paragraph.splitlines()).strip()

    return value or None


def _validation_issues(error: ValidationError, /) -> tuple[ToolIssue, ...]:
    return tuple(
        ToolIssue(
            path=tuple(detail["loc"]),
            message=detail["msg"],
            code=detail["type"],
        )
        for detail in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    )


def _tool_result(
    call: ToolCall,
    value: object,
    adapter: TypeAdapter[Any],
    /,
) -> ToolResult:
    if isinstance(value, ToolResult):
        if value.call != call:
            raise ToolOutputError(call.name, detail="returned a result for different tool call")
        return value

    if isinstance(value, str | Content):
        return ToolResult(call, value)

    if isinstance(value, _RAW_BYTES):
        raise ToolOutputError(call.name, detail="returned raw bytes; return Binary(...) instead")

    try:
        document = adapter.dump_python(value, mode="json", warnings="error")
        return ToolResult.json(call, document)
    except TypeError, ValueError:
        raise ToolOutputError(call.name) from None


def _is_dict(value: object, /) -> TypeIs[dict[object, object]]:
    return isinstance(value, dict)
