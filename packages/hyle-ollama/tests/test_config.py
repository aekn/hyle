from typing import Any

import pytest

from hyle_ollama import (
    OllamaEmbeddingConfig,
    OllamaGenerationConfig,
    OllamaOptions,
)


def _invalid(value: object) -> Any:
    return value


def test_options_snapshot_stop_sequences() -> None:
    stop = ["END"]
    options = OllamaOptions(stop=_invalid(stop))
    stop.append("MORE")
    assert options.stop == ("END",)


def test_options_keep_sentinel_friendly_integers() -> None:
    options = OllamaOptions(num_predict=-1, main_gpu=0, temperature=0.0, use_mmap=False)
    assert options.num_predict == -1
    assert options.main_gpu == 0
    assert options.temperature == 0.0
    assert options.use_mmap is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_options_reject_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="temperature must be finite"):
        OllamaOptions(temperature=value)


def test_options_reject_bool_as_integer() -> None:
    with pytest.raises(TypeError, match="num_ctx must be an integer"):
        OllamaOptions(num_ctx=_invalid(True))


def test_generation_config_validates_thinking_and_logprobs() -> None:
    assert OllamaGenerationConfig(think="max").think == "max"

    with pytest.raises(ValueError, match="invalid think value"):
        OllamaGenerationConfig(think=_invalid("extreme"))
    with pytest.raises(TypeError, match="think must be a boolean or string"):
        OllamaGenerationConfig(think=_invalid(1))
    with pytest.raises(ValueError, match="between 0 and 20"):
        OllamaGenerationConfig(logprobs=True, top_logprobs=21)
    with pytest.raises(ValueError, match="requires logprobs=True"):
        OllamaGenerationConfig(top_logprobs=1)


def test_keep_alive_rejects_bool_and_blank_text() -> None:
    with pytest.raises(TypeError, match="string or finite number"):
        OllamaGenerationConfig(keep_alive=_invalid(True))
    with pytest.raises(ValueError, match="must not be blank"):
        OllamaEmbeddingConfig(keep_alive="   ")


def test_config_requires_options_value() -> None:
    with pytest.raises(TypeError, match="options must be OllamaOptions"):
        OllamaGenerationConfig(options=_invalid(object()))
    with pytest.raises(TypeError, match="options must be OllamaOptions"):
        OllamaEmbeddingConfig(options=_invalid(object()))


def test_stop_sequences_do_not_treat_text_as_a_sequence() -> None:
    with pytest.raises(TypeError, match="stop must be a sequence of strings"):
        OllamaOptions(stop=_invalid("END"))
