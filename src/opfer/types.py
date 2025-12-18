from __future__ import annotations

import base64
import hashlib
from collections.abc import (
    Awaitable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import cache
from typing import (
    Annotated,
    Any,
    Callable,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from PIL import Image
from pydantic import BaseModel, Field, PlainSerializer, TypeAdapter

from opfer.internal.inspect import FuncSchema

type JsonValue = (
    None | str | bool | int | float | Sequence[JsonValue] | Mapping[str, JsonValue]
)

type JsonComparableValue = None | str | int | float

type JsonSchema = Mapping[str, JsonValue] | bool


@dataclass(frozen=True)
class Blob:
    mime_type: str
    data: bytes

    @property
    @cache
    def content_md5(self):
        """RFC1864: base64-encoded MD5 digests"""
        return base64.b64encode(hashlib.md5(self.data).digest())


@dataclass
class File:
    name: str
    mime_type: str
    data: bytes
    metadata: dict[str, str] | None = field(default=None)

    @property
    @cache
    def content_md5(self):
        """RFC1864: base64-encoded MD5 digests"""
        return base64.b64encode(hashlib.md5(self.data).digest())


class DataClass(BaseModel):
    pass


class MediaResolutionLevel(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MediaResolution(DataClass):
    level: MediaResolutionLevel | None = Field(default=None)
    num_tokens: int | None = Field(default=None)


class PartText(DataClass):
    text: str | None = Field(default=None)


class PartThought(DataClass):
    text: str | None = Field(default=None)
    is_summary: bool | None = Field(default=None)


class PartBlob(DataClass):
    mime_type: str
    uri: str
    content_md5: str


class PartFunctionCall(DataClass):
    id: str
    name: str
    arguments: dict[str, Any] | None = Field(default=None)


class PartFunctionResponsePartBlob(DataClass):
    mime_type: str
    uri: str
    content_md5: str


class PartFunctionResponsePart(DataClass):
    blob: PartFunctionResponsePartBlob | None = Field(default=None)


class PartFunctionResponse(DataClass):
    id: str
    name: str
    response: dict[str, Any] | None = Field(default=None)
    parts: list[PartFunctionResponsePart] | None = Field(default=None)


class Part(DataClass):
    thought_signature: bytes | None = Field(default=None)
    media_resolution: MediaResolution | None = Field(default=None)
    text: PartText | None = Field(default=None)
    thought: PartThought | None = Field(default=None)
    blob: PartBlob | None = Field(default=None)
    function_call: PartFunctionCall | None = Field(default=None)
    function_response: PartFunctionResponse | None = Field(default=None)

    @property
    def type(
        self,
    ) -> PartText | PartBlob | PartThought | PartFunctionCall | PartFunctionResponse:
        match self:
            case Part(text=text) if text is not None:
                return text
            case Part(blob=blob) if blob is not None:
                return blob
            case Part(thought=thought) if thought is not None:
                return thought
            case Part(function_call=function_call) if function_call is not None:
                return function_call
            case Part(function_response=function_response) if (
                function_response is not None
            ):
                return function_response
        raise ValueError("part has no valid type.")


class Role(str, Enum):
    USER = "USER"
    MODEL = "MODEL"


class Content(DataClass):
    role: Role | None
    parts: list[Part] | None

    @property
    def text(self) -> str:
        if not self.parts:
            return ""
        texts = []
        for part in self.parts:
            match part.type:
                case PartText():
                    if part.text:
                        texts.append(part.text.text)
        return "\n".join(texts)


type PartLike = str | Image.Image | Part
type ContentLike = Content | PartLike | Sequence[PartLike]
type ContentListLike = ContentLike | Sequence[ContentLike]

ContentListAdapter = TypeAdapter(Sequence[Content])


class ModelConfig(DataClass):
    provider: str
    name: str
    temperature: float | None = Field(default=None)
    max_output_tokens: int | None = Field(default=None)
    stop_sequences: list[str] | None = Field(default=None)
    top_p: float | None = Field(default=None)
    top_k: float | None = Field(default=None)
    response_logprobs: bool | None = Field(default=None)
    logprobs: int | None = Field(default=None)
    presence_penalty: float | None = Field(default=None)
    frequency_penalty: float | None = Field(default=None)
    seed: int | None = Field(default=None)


class TokenModality(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    DOCUMENT = "DOCUMENT"


class ModalityTokenCount(DataClass):
    modality: TokenModality
    tokens: int


class Usage(DataClass):
    total_tokens: int | None
    input_tokens: int | None
    input_tokens_details: list[ModalityTokenCount] | None = Field(default=None)
    output_tokens: int | None
    thought_tokens: int | None = Field(default=None)
    cached_tokens: int | None = Field(default=None)
    cached_tokens_details: list[ModalityTokenCount] | None = Field(default=None)


ModalityTokenCountList = TypeAdapter(list[ModalityTokenCount])

type Datetime = Annotated[
    datetime, PlainSerializer(lambda dt: dt.isoformat(), return_type=str)
]


class AgentResponseMetadata(DataClass):
    id: str | None
    provider: str
    model: str
    timestamp: Datetime
    usage: Usage


class AgentResponse[T](DataClass):
    metadata: AgentResponseMetadata
    instruction: Content | None
    input: list[Content]
    output: list[Content]
    finish_reason: str
    parsed: T | None = Field(default=None)


class AgentTurnResult(DataClass):
    steps: list[AgentResponse]


class AgentOutput[T](DataClass):
    turns: list[AgentTurnResult]

    @property
    def final_output(self) -> T:
        if not self.turns:
            raise ValueError("no turns available in agent output.")
        final_turn = self.turns[-1]
        if not final_turn.steps:
            raise ValueError("no steps available in final turn.")
        final_step = final_turn.steps[-1]
        if final_step.parsed is None:
            raise ValueError("final step parsed output is None.")
        return final_step.parsed

    @property
    def final_output_text(self) -> str:
        if not self.turns:
            raise ValueError("no turns available in agent output.")
        final_turn = self.turns[-1]
        if not final_turn.steps:
            raise ValueError("no steps available in final turn.")
        final_step = final_turn.steps[-1]
        texts = []
        for content in final_step.output:
            texts.append(content.text)
        return "\n".join(texts)


class ToolOutput(TypedDict):
    output: JsonValue


class ToolOutputError(TypedDict):
    error: JsonValue


type ToolResult = ToolOutput | ToolOutputError


@runtime_checkable
class Tool[**I](Protocol):
    @property
    def schema(self) -> FuncSchema: ...

    async def call(
        self,
        call_id: str,
        *args: I.args,
        **kwargs: I.kwargs,
    ) -> ToolResult: ...


type ToolLike = Tool | Callable[..., Awaitable[Any]]


class Agent[T](Protocol):
    @property
    def id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def description(self) -> str | None: ...

    @property
    def instruction(self) -> ContentLike: ...

    @property
    def model(self) -> ModelConfig: ...

    @property
    def output_type(self) -> type[T] | None: ...

    @property
    def tools(self) -> Sequence[Tool]: ...

    async def run(
        self,
        input: ContentListLike,
        max_turns: int | None = None,
    ) -> AgentOutput[T]: ...

    async def invoke_tools(
        self, calls: list[PartFunctionCall]
    ) -> list[PartFunctionResponse]: ...

    def as_tool(
        self,
        description: str,
        name: str | None = None,
    ) -> Tool[[str]]: ...


class Chat(Protocol):
    async def send[T](
        self,
        agent: Agent[T],
        input: list[Content],
        instruction: Content | None = None,
    ) -> AgentResponse: ...

    @property
    def history(self) -> list[Content]: ...


class ModelProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def chat(self) -> Chat: ...


class ModelProviderRegistry(Protocol):
    def register(self, provider: ModelProvider) -> None: ...

    def get(self, name: str) -> ModelProvider: ...


class ArtifactStorage(Protocol):
    async def exists(self, uri: str) -> bool: ...
    async def download(self, uri: str) -> File: ...
    async def upload(self, file: File) -> str: ...


class BlobStorage(Protocol):
    async def exists(self, uri: str) -> bool: ...
    async def download(self, uri: str) -> Blob: ...
    async def upload(self, blob: Blob) -> str: ...
