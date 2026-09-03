from hyle import Model, Msg, Request, Response, Structured, run, tool
from hyle.content import Binary, Content, ResourceRef
from hyle.embeddings import Embeddable, Embedder, Embedding, Embeddings
from hyle.generation import Part, PartBuffer, Reasoning, Stream, StreamingModel
from hyle.runtime import RunEvent, RunLimits, RunResult, RunStream, run_stream
from hyle.schema import JsonSchema
from hyle.tools import (
    FunctionTool,
    Tool,
    ToolBatchError,
    ToolCall,
    ToolFailed,
    ToolFinished,
    ToolResult,
    ToolSpec,
    ToolStarted,
    execute_tool,
    execute_tools,
)
from hyle.usage import Usage
from hyle_ollama import (
    OllamaClient,
    OllamaEmbedder,
    OllamaEmbeddingConfig,
    OllamaEmbeddings,
    OllamaError,
    OllamaGenerationConfig,
    OllamaModel,
    OllamaModelDetails,
    OllamaOptions,
    OllamaResponse,
    OllamaStream,
)


def main() -> None:
    public: tuple[object, ...] = (
        Binary,
        Content,
        Embeddable,
        Embedder,
        Embedding,
        Embeddings,
        FunctionTool,
        JsonSchema,
        Model,
        Msg,
        OllamaClient,
        OllamaEmbedder,
        OllamaEmbeddingConfig,
        OllamaEmbeddings,
        OllamaError,
        OllamaGenerationConfig,
        OllamaModel,
        OllamaModelDetails,
        OllamaOptions,
        OllamaResponse,
        OllamaStream,
        Part,
        PartBuffer,
        Reasoning,
        Request,
        ResourceRef,
        Response,
        RunEvent,
        RunLimits,
        RunResult,
        RunStream,
        Stream,
        StreamingModel,
        Structured,
        Tool,
        ToolBatchError,
        ToolCall,
        ToolFailed,
        ToolFinished,
        ToolResult,
        ToolSpec,
        ToolStarted,
        Usage,
        execute_tool,
        execute_tools,
        run,
        run_stream,
        tool,
    )
    print(f"imported {len(public)} public symbols")


if __name__ == "__main__":
    main()
