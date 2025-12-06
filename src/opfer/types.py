from __future__ import annotations

from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import (
    Annotated,
    Any,
    Mapping,
    Sequence,
)

from PIL import Image
from pydantic import BaseModel, Field, PlainSerializer, TypeAdapter

from opfer.blob import Blob, download_blob, get_blob_storage

type JsonValue = (
    None | str | bool | int | float | Sequence[JsonValue] | Mapping[str, JsonValue]
)

type JsonComparableValue = None | str | int | float

type JsonSchema = Mapping[str, JsonValue] | bool


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
    url: str
    content_md5: str


async def new_part_image(image: Image.Image, format: str = "webp") -> Part:
    buf = BytesIO()
    image.save(buf, format=format)
    data = buf.getvalue()
    blob = Blob(
        mime_type=f"image/{format.lower()}",
        data=data,
    )
    resolver = get_blob_storage()
    url = await resolver.upload(blob)
    return Part(
        blob=PartBlob(
            mime_type=blob.mime_type,
            url=url,
            content_md5=blob.content_md5.decode(),
        )
    )


async def download_part_blob(blob: PartBlob | PartFunctionResponsePartBlob) -> Blob:
    downloaded = await download_blob(blob.url)
    assert (
        downloaded.mime_type == blob.mime_type
    ), f"downloaded mime type {downloaded.mime_type} does not match, expected {blob.mime_type}"
    assert (
        downloaded.content_md5 == blob.content_md5.encode()
    ), f"downloaded content_md5 {downloaded.content_md5} does not match, expected {blob.content_md5}"
    return downloaded


class PartFunctionCall(DataClass):
    id: str
    name: str
    arguments: dict[str, Any] | None = Field(default=None)


class PartFunctionResponsePartBlob(DataClass):
    mime_type: str
    url: str
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
            case Part(
                function_response=function_response
            ) if function_response is not None:
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
type ContentLike = Content | PartLike | list[PartLike]
type ContentListLike = ContentLike | list[ContentLike]

ContentListAdapter = TypeAdapter(list[Content])


async def as_instruction_part(part: PartLike) -> Part:
    match part:
        case str():
            return Part(text=PartText(text=part))
        case Image.Image():
            return await new_part_image(part)
        case Part():
            match part.type:
                case PartText():
                    return part
                case PartBlob():
                    raise ValueError("instruction part does not support image.")
                case PartFunctionResponse():
                    raise ValueError(
                        "instruction part does not support function response."
                    )
                case PartThought() | PartFunctionCall():
                    raise ValueError("instruction part does not support model part.")


async def as_instruction_content(content: ContentLike) -> Content:
    match content:
        case Content():
            if content.role is None or content.role != Role.USER:
                raise ValueError("content is none or not a user content.")
            return content
        case str() | Image.Image() | Part():
            return Content(role=Role.USER, parts=[await as_instruction_part(content)])
        case list():
            parts = [await as_instruction_part(p) for p in content]
            return Content(role=Role.USER, parts=parts)


async def as_part(part: PartLike) -> Part:
    match part:
        case str():
            return Part(text=PartText(text=part))
        case Image.Image():
            return await new_part_image(part)
        case Part():
            return part


async def _pop_front_content(
    contents: list[PartLike] | list[ContentLike],
) -> tuple[Content | None, list[ContentLike]]:
    if not contents:
        return None, []
    first, *rest = contents
    match first:
        case str() | Image.Image():
            role = Role.USER
        case Part():
            match first.type:
                case PartText() | PartBlob() | PartFunctionResponse():
                    role = Role.USER
                case PartThought() | PartFunctionCall():
                    role = Role.MODEL
        case Content():
            return first, rest
        case list():
            # part list
            first_part, *_ = first
            match first_part:
                case str() | Image.Image():
                    role = Role.USER
                case Part():
                    match first_part.type:
                        case PartText() | PartBlob() | PartFunctionResponse():
                            role = Role.USER
                        case PartThought() | PartFunctionCall():
                            role = Role.MODEL
            return Content(role=role, parts=[await as_part(p) for p in first]), rest
    # collect same role parts
    same_role_parts = [first]
    for content in rest:
        match content:
            case str():
                if role == Role.USER:
                    same_role_parts.append(content)
                else:
                    break
            case Part():
                match content.type:
                    case PartText() | PartBlob() | PartFunctionResponse():
                        if role == Role.USER:
                            same_role_parts.append(content)
                        else:
                            break
                    case PartThought() | PartFunctionCall():
                        if role == Role.MODEL:
                            same_role_parts.append(content)
                        else:
                            break
            case Content() | list():
                break
    remaining_contents = rest[len(same_role_parts) - 1 :]
    return (
        Content(
            role=role,
            parts=[await as_part(p) for p in same_role_parts],
        ),
        remaining_contents,
    )


async def as_content_list(contents: ContentListLike) -> list[Content]:
    match contents:
        case str():
            return [Content(role=Role.USER, parts=[Part(text=PartText(text=contents))])]
        case Image.Image():
            part = await new_part_image(contents)
            return [Content(role=Role.USER, parts=[part])]
        case Part():
            match contents.type:
                case PartText() | PartBlob() | PartFunctionResponse():
                    return [Content(role=Role.USER, parts=[contents])]
                case PartThought() | PartFunctionCall():
                    return [Content(role=Role.MODEL, parts=[contents])]
        case Content():
            return [contents]
        case list():
            result: list[Content] = []
            remaining_contents: list[PartLike] | list[ContentLike] = contents
            while remaining_contents:
                content, remaining_contents = await _pop_front_content(
                    remaining_contents,
                )
                if content is None:
                    break
                result.append(content)
            return result


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


class AgentResponse[TOutput](DataClass):
    metadata: AgentResponseMetadata
    instruction: Content | None
    input: list[Content]
    output: list[Content]
    finish_reason: str
    parsed: TOutput | None = Field(default=None)


class AgentTurnResult(DataClass):
    steps: list[AgentResponse]


class AgentOutput[TOutput](DataClass):
    turns: list[AgentTurnResult]

    @property
    def final_output(self) -> TOutput:
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
