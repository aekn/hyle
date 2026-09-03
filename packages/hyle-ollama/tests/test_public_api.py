import hyle_ollama


def test_provider_root_is_deliberate() -> None:
    expected = {
        "OllamaClient",
        "OllamaEmbedder",
        "OllamaEmbeddingConfig",
        "OllamaEmbeddings",
        "OllamaError",
        "OllamaGenerationConfig",
        "OllamaLogprob",
        "OllamaModel",
        "OllamaModelDetails",
        "OllamaModelMetadata",
        "OllamaModelSummary",
        "OllamaOptions",
        "OllamaProgress",
        "OllamaResponse",
        "OllamaRunningModel",
        "OllamaStream",
        "OllamaTimings",
        "OllamaTokenLogprob",
    }
    assert set(hyle_ollama.__all__) == expected
    for name in expected:
        assert getattr(hyle_ollama, name) is not None
