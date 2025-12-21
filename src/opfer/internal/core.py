# from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import uuid
from collections.abc import Awaitable, Buffer, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Callable, Protocol, assert_never

from PIL import Image
from pydantic import ValidationError

from opfer.errors import (
    MaxStepsExceeded,
    ModelBehaviorError,
    ModelIncompleteError,
    ModelRefusalError,
    ToolError,
)
from opfer.internal import attributes, operations
from opfer.internal.inspect import FuncSchema, get_func_schema
from opfer.internal.tracing import tracer
from opfer.types import (
    Agent,
    AgentOutput,
    AgentResponse,
    AgentTurnResult,
    Artifact,
    ArtifactStorage,
    ArtifactUpload,
    Content,
    ContentLike,
    ContentListAdapter,
    ContentListLike,
    JsonValue,
    ModalityTokenCountList,
    ModelConfig,
    ModelProvider,
    ModelProviderRegistry,
    Part,
    PartBlob,
    PartFunctionCall,
    PartFunctionResponse,
    PartFunctionResponsePartBlob,
    PartLike,
    PartText,
    PartThought,
    Role,
    Tool,
    ToolInterceptor,
    ToolInterceptorContext,
    ToolLike,
    ToolResult,
)

_context_artifact_repository = ContextVar[ArtifactStorage]("artifact_storage")
_context_model_provider_registry = ContextVar[ModelProviderRegistry](
    "model_provider_registry"
)


def get_artifact_storage() -> ArtifactStorage:
    return _context_artifact_repository.get()


def set_artifact_storage(repo: ArtifactStorage) -> Token[ArtifactStorage]:
    return _context_artifact_repository.set(repo)


def reset_artifact_storage(token: Token[ArtifactStorage]) -> None:
    _context_artifact_repository.reset(token)


async def upload_artifact(upload: ArtifactUpload) -> Artifact:
    return await get_artifact_storage().upload(upload)


async def get_artifact(uri: str) -> Artifact:
    return await get_artifact_storage().get(uri)


@dataclass
class _InMemoryArtifact(Artifact):
    _uri: str
    _mime_type: str
    _size: int
    _content: bytes
    _content_md5: bytes
    _display_name: str | None
    _metadata: dict[str, str] | None

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def display_name(self) -> str | None:
        return self._display_name

    @property
    def mime_type(self) -> str:
        return self._mime_type

    @property
    def metadata(self) -> dict[str, str] | None:
        return self._metadata

    @property
    def size(self) -> int:
        return self._size

    @property
    def content_md5(self) -> bytes:
        return self._content_md5

    async def download_as_bytes(self) -> bytes:
        return self._content


class InMemoryArtifactStorage:
    data: dict[str, _InMemoryArtifact]

    def __init__(self):
        self.data = {}

    async def exists(self, uri: str) -> bool:
        return uri in self.data

    async def get(self, uri: str) -> Artifact:
        return self.data[uri]

    async def upload(self, upload: ArtifactUpload) -> Artifact:
        content_md5 = base64.b64encode(hashlib.md5(upload.content).digest())
        size = len(upload.content)
        uri = f"artifacts://{content_md5.decode()}"
        self.data[uri] = file = _InMemoryArtifact(
            _uri=uri,
            _mime_type=upload.mime_type,
            _size=size,
            _content=upload.content,
            _content_md5=content_md5,
            _display_name=upload.display_name,
            _metadata=upload.metadata,
        )
        return file


def get_model_provider_registry() -> ModelProviderRegistry:
    return _context_model_provider_registry.get()


def set_model_provider_registry(
    registry: ModelProviderRegistry,
) -> Token[ModelProviderRegistry]:
    return _context_model_provider_registry.set(registry)


def reset_model_provider_registry(
    token: Token[ModelProviderRegistry],
) -> None:
    _context_model_provider_registry.reset(token)


class DefaultModelProviderRegistry:
    _providers: dict[str, ModelProvider]

    def __init__(self):
        self._providers = {}

    def register(self, provider: ModelProvider) -> None:
        if provider.name in self._providers:
            raise ValueError(f"ModelProvider '{provider.name}' already registered")
        self._providers[provider.name] = provider

    def get(self, name: str) -> ModelProvider:
        if name not in self._providers:
            raise ValueError(f"ModelProvider '{name}' not registered")
        return self._providers[name]


async def get_part_blob(blob: PartBlob | PartFunctionResponsePartBlob) -> Artifact:
    downloaded = await get_artifact(blob.uri)
    assert downloaded.mime_type == blob.mime_type, (
        f"downloaded mime type {downloaded.mime_type} does not match, expected {blob.mime_type}"
    )
    assert downloaded.content_md5 == blob.content_md5.encode(), (
        f"downloaded content_md5 {downloaded.content_md5} does not match, expected {blob.content_md5}"
    )
    return downloaded


async def image_as_part(image: Image.Image, format: str = "webp") -> Part:
    buf = io.BytesIO()
    image.save(buf, format=format)
    data = buf.getvalue()
    blob = ArtifactUpload(
        display_name=f"image.{format.lower()}",
        mime_type=f"image/{format.lower()}",
        content=data,
    )
    upload = await upload_artifact(blob)
    return Part(
        blob=PartBlob(
            mime_type=blob.mime_type,
            uri=upload.uri,
            content_md5=upload.content_md5.decode(),
        )
    )


async def as_instruction_part(part: PartLike) -> Part:
    match part:
        case str():
            return Part(text=PartText(text=part))
        case Image.Image():
            return await image_as_part(part)
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
        case Sequence():
            parts = [await as_instruction_part(p) for p in content]
            return Content(role=Role.USER, parts=parts)


async def as_part(part: PartLike) -> Part:
    match part:
        case str():
            return Part(text=PartText(text=part))
        case Image.Image():
            return await image_as_part(part)
        case Part():
            return part


async def _pop_front_content(
    contents: Sequence[PartLike] | Sequence[ContentLike],
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
        case Sequence():
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
            case Content() | Sequence():
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
            part = await image_as_part(contents)
            return [Content(role=Role.USER, parts=[part])]
        case Part():
            match contents.type:
                case PartText() | PartBlob() | PartFunctionResponse():
                    return [Content(role=Role.USER, parts=[contents])]
                case PartThought() | PartFunctionCall():
                    return [Content(role=Role.MODEL, parts=[contents])]
        case Content():
            return [contents]
        case Sequence():
            result: list[Content] = []
            remaining_contents: Sequence[PartLike] | Sequence[ContentLike] = contents
            while remaining_contents:
                content, remaining_contents = await _pop_front_content(
                    remaining_contents,
                )
                if content is None:
                    break
                result.append(content)
            return result


def _func_schema_as_tool_definition(schema: FuncSchema) -> JsonValue:
    # TODO: implement
    return None


class _Hasher(Protocol):
    def update(self, buffer: Buffer, /) -> None: ...


def _hash_func(hasher: _Hasher, func: Callable):
    code = func.__code__
    attributes = [
        code.co_code,
        code.co_consts,
        code.co_names,
        code.co_varnames,
    ]

    for attr in attributes:
        hasher.update(str(attr).encode())


_current_tool_call_id = ContextVar[str]("current_tool_call_id")


def get_current_tool_call_id() -> str:
    return _current_tool_call_id.get()


@contextmanager
def as_current_tool_call_id(call_id: str):
    t = _current_tool_call_id.set(call_id)
    try:
        yield call_id
    finally:
        _current_tool_call_id.reset(t)


@dataclass
class _Tool[**I, O](Tool[I]):
    _func: Callable[I, Awaitable[O]]
    _schema: FuncSchema
    _interceptors: list[ToolInterceptor[I, O]]
    _md5: bytes

    def __init__(
        self,
        func: Callable[I, Awaitable[O]],
        name: str | None = None,
        description: str | None = None,
        interceptors: Sequence[ToolInterceptor[I, O]] | None = None,
    ):
        self._func = func
        self._schema = get_func_schema(
            func,
            name=name,
            description=description,
            # allow_positional_args=False,
        )
        self._interceptors = list(interceptors or [])

        hasher = hashlib.md5()
        _hash_func(hasher, func)
        self._md5 = base64.b64encode(hasher.digest())

    @property
    def schema(self) -> FuncSchema:
        return self._schema

    @property
    def md5(self) -> bytes:
        return self._md5

    async def __call__(self, *args: I.args, **kwargs: I.kwargs) -> O:
        call_id = f"direct_call_{uuid.uuid4().hex[:16]}"
        with (
            as_current_tool_call_id(call_id),
            tracer.span(
                f"call tool {self._schema.name}",
                attributes={
                    attributes.OPERATION_NAME: "tool_call",
                    attributes.TOOL_NAME: self._schema.name,
                    attributes.TOOL_DESCRIPTION: self._schema.description or "",
                    attributes.TOOL_DEFINITION: _func_schema_as_tool_definition(
                        self._schema
                    ),
                    attributes.TOOL_CALL_ID: call_id,
                },
            ) as s,
        ):
            _args = {
                **{
                    k: v
                    for k, v in zip(self._schema.input_model.model_fields.keys(), args)
                },
                **kwargs,
            }
            input = self._schema.input_model.model_validate(_args)
            s.set_attribute(
                attributes.TOOL_CALL_INPUT,
                input.model_dump(),
            )
            ctx = ToolInterceptorContext(
                call_id=call_id,
                schema=self._schema,
                input=input,
            )
            fn = self._func
            for interceptor in self._interceptors:
                fn = interceptor(fn, ctx)
            output = await fn(*args, **kwargs)
            output_json = self._schema.output_type.dump_python(output)
            s.set_attribute(
                attributes.TOOL_CALL_OUTPUT,
                output_json,
            )
            return output

    async def call(self, call_id: str, *args: I.args, **kwargs: I.kwargs) -> ToolResult:
        with (
            as_current_tool_call_id(call_id),
            tracer.span(
                f"call tool {self._schema.name}",
                attributes={
                    attributes.OPERATION_NAME: "tool_call",
                    attributes.TOOL_NAME: self._schema.name,
                    attributes.TOOL_DESCRIPTION: self._schema.description or "",
                    attributes.TOOL_DEFINITION: _func_schema_as_tool_definition(
                        self._schema
                    ),
                    attributes.TOOL_CALL_ID: call_id,
                },
            ) as s,
        ):
            try:
                _args = {
                    **{
                        k: v
                        for k, v in zip(
                            self._schema.input_model.model_fields.keys(), args
                        )
                    },
                    **kwargs,
                }
                input = self._schema.input_model.model_validate(_args)
                s.set_attribute(
                    attributes.TOOL_CALL_INPUT,
                    input.model_dump(),
                )
            except ValidationError as e:
                s.record_exception(e)
                return {"error": "invalid input: " + repr(e)}
            ctx = ToolInterceptorContext(
                call_id=call_id,
                schema=self._schema,
                input=input,
            )
            fn = self._func
            for interceptor in self._interceptors:
                fn = interceptor(fn, ctx)
            try:
                output = await fn(*args, **kwargs)
            except ToolError as e:
                s.record_exception(e)
                return {"error": str(e)}
            try:
                output_json = self._schema.output_type.dump_python(output)
                s.set_attribute(
                    attributes.TOOL_CALL_OUTPUT,
                    output_json,
                )
                return {"output": output_json}
            except ValidationError as e:
                s.record_exception(e)
                return {"error": "internal error: invalid output: " + repr(e)}


def tool[**I, O](
    name: str | None = None,
    description: str | None = None,
    interceptors: Sequence[ToolInterceptor[I, O]] | None = None,
):
    def decorator(func: Callable[I, Awaitable[O]]) -> _Tool[I, O]:
        return _Tool[I, O](
            func,
            name=name,
            description=description,
            interceptors=interceptors,
        )

    return decorator


_tool_cache_attr = "_opfer_tool_cache"


def _as_tool(fn: ToolLike) -> Tool:
    if isinstance(fn, Tool):
        return fn
    cache = getattr(fn, _tool_cache_attr, None)
    if cache is not None:
        if not isinstance(cache, _Tool):
            raise ValueError(f"invalid tool cache type: {type(cache)} of {fn}")
        return cache
    tool_instance = _Tool(fn)
    setattr(fn, _tool_cache_attr, tool_instance)
    return tool_instance


def _hash_part_like(hasher: _Hasher, part: PartLike):
    match part:
        case str():
            hasher.update(part.encode("utf-8"))
        case Image.Image():
            hasher.update(part.tobytes())
        case Part():
            hasher.update(
                part.model_dump_json(
                    indent=None,
                    exclude_unset=True,
                    exclude_none=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            )
        case _:
            assert_never(part)


def _hash_content_like(hasher: _Hasher, content: ContentLike):
    match content:
        case Content():
            hasher.update(
                content.model_dump_json(
                    indent=None,
                    exclude_unset=True,
                    exclude_none=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            )
        case Sequence():
            for item in content:
                _hash_part_like(hasher, item)
        case str() | Image.Image() | Part():
            _hash_part_like(hasher, content)
        case _:
            assert_never(content)


class _Agent[T]:
    _id: str
    _display_name: str
    _description: str | None
    _instruction: ContentLike
    _model: ModelConfig
    _output_type: type[T] | None
    _tools: list[Tool]
    _md5: bytes

    def __init__(
        self,
        *,
        id: str,
        display_name: str,
        model: ModelConfig,
        instruction: ContentLike,
        description: str | None = None,
        output_type: type[T] | None = None,
        tools: Sequence[ToolLike] | None = None,
    ):
        self._id = id
        self._display_name = display_name
        self._description = description
        self._instruction = instruction
        self._model = model
        self._output_type = output_type
        self._tools = [_as_tool(t) for t in tools or []]

        hasher = hashlib.md5()
        hasher.update(self._id.encode("utf-8"))
        _hash_content_like(hasher, self._instruction)
        hasher.update(repr(self._output_type).encode("utf-8"))
        hasher.update(
            self._model.model_dump_json(
                indent=None,
                exclude_unset=True,
                exclude_none=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        for tool in self._tools:
            hasher.update(tool.md5)
        self._md5 = base64.b64encode(hasher.digest())

    @property
    def id(self) -> str:
        return self._id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def instruction(self) -> ContentLike:
        return self._instruction

    @property
    def model(self) -> ModelConfig:
        return self._model

    @property
    def output_type(self) -> type[T] | None:
        return self._output_type

    @property
    def tools(self) -> Sequence[Tool]:
        return self._tools

    @property
    def md5(self) -> bytes:
        return self._md5

    async def run(
        self,
        input: ContentListLike,
        max_turns: int | None = None,
    ) -> AgentOutput[T]:
        with tracer.span(
            operations.OPFER_AGENT_RUN,
            attributes={
                attributes.OPERATION_NAME: operations.OPFER_AGENT_RUN,
                attributes.DESCRIPTION: f"Running agent {self.display_name} ({self.id})",
                attributes.AI_AGENT_ID: self._id,
                attributes.AI_AGENT_NAME: self._display_name,
                attributes.AI_AGENT_DESCRIPTION: self._description,
                attributes.AI_AGENT_RUN_MAX_TURNS: max_turns,
            },
        ):
            return await _run_agent(
                agent=self,
                input=input,
                max_turns=max_turns,
            )

    def as_tool(
        self,
        description: str,
        name: str | None = None,
    ) -> Tool[[str]]:
        return _agent_as_tool(self, name, description)

    def find_tool_by_name(self, name: str) -> Tool | None:
        for tool in self._tools:
            if tool.schema.name == name:
                return tool

    async def invoke_tools(
        self, calls: list[PartFunctionCall]
    ) -> list[PartFunctionResponse]:
        responses: list[PartFunctionResponse] = []
        async with asyncio.TaskGroup() as tg:
            tools = [self.find_tool_by_name(call.name) for call in calls]
            missing_tools = [
                call.name for call, tool in zip(calls, tools) if tool is None
            ]
            if missing_tools:
                raise ValueError(f"tools not found: {', '.join(missing_tools)}.")
            tools = [i for i in tools if i is not None]
            tasks = [
                tg.create_task(tool.call(call_id=call.id, **(call.arguments or {})))
                for call, tool in zip(calls, tools)
            ]
            for call, task in zip(calls, tasks):
                response = await task
                responses.append(
                    PartFunctionResponse(
                        id=call.id,
                        name=call.name,
                        response=dict(response),
                    )
                )
        return responses


def agent[T = str](
    *,
    id: str,
    display_name: str,
    model: ModelConfig,
    instruction: ContentLike,
    description: str | None = None,
    output_type: type[T] | None = None,
    tools: Sequence[ToolLike] | None = None,
) -> Agent[T]:
    return _Agent[T](
        id=id,
        display_name=display_name,
        model=model,
        instruction=instruction,
        description=description,
        output_type=output_type,
        tools=tools,
    )


def _agent_as_tool[T](
    agent: Agent[T],
    name: str | None = None,
    description: str | None = None,
) -> _Tool[[str], T]:
    async def tool_func(input: str) -> T:
        output = await agent.run(input)

        # print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))
        return output.final_output

    return tool(
        name=name or f"ask_to_subagent_{agent.id}",
        description=description,
    )(tool_func)


async def _run_agent_turn[T](
    turn: int,
    agent: Agent[T],
    input: list[Content],
    instruction: Content | None,
    max_steps: int | None = None,
) -> AgentTurnResult:
    provider_registry = get_model_provider_registry()
    provider = provider_registry.get(agent.model.provider)

    chat = await provider.chat()

    responses: list[AgentResponse] = []
    step = 0

    current_input = input

    while True:
        with tracer.span(
            operations.OPFER_AGENT_STEP,
            attributes={
                attributes.OPERATION_NAME: operations.OPFER_AGENT_STEP,
                attributes.DESCRIPTION: f"Running agent {agent.display_name} ({agent.id}): turn {turn}: step {step}",
                attributes.AI_AGENT_RUN_STEP_NUMBER: step,
            },
        ):
            # TODO: if step reaches max, disable tool calling.

            with tracer.span(
                operations.OPFER_AGENT_CHAT,
                attributes={
                    attributes.OPERATION_NAME: operations.OPFER_AGENT_CHAT,
                    attributes.AI_INSTRUCTION: instruction.model_dump_json(
                        ensure_ascii=False,
                        exclude_unset=True,
                        exclude_none=True,
                    )
                    if instruction
                    else None,
                    attributes.AI_INPUT: ContentListAdapter.dump_json(
                        chat.history + current_input,
                        ensure_ascii=False,
                        exclude_unset=True,
                        exclude_none=True,
                    ).decode(),
                    attributes.AI_REQUEST_PROVIDER: agent.model.provider,
                    attributes.AI_REQUEST_MODEL: agent.model.name,
                    attributes.AI_REQUEST_TEMPERATURE: agent.model.temperature,
                    attributes.AI_REQUEST_MAX_OUTPUT_TOKENS: agent.model.max_output_tokens,
                    attributes.AI_REQUEST_STOP_SEQUENCES: agent.model.stop_sequences,
                    attributes.AI_REQUEST_TOP_P: agent.model.top_p,
                    attributes.AI_REQUEST_TOP_K: agent.model.top_k,
                    attributes.AI_REQUEST_PRESENCE_PENALTY: agent.model.presence_penalty,
                    attributes.AI_REQUEST_FREQUENCY_PENALTY: agent.model.frequency_penalty,
                    attributes.AI_REQUEST_SEED: agent.model.seed,
                    # attributes.AI_REQUEST_TOOL_DEFINITIONS:  # TODO
                },
            ) as s:
                try:
                    response = await chat.send(
                        agent=agent,
                        input=current_input,
                        instruction=instruction,
                    )
                    s.set_attributes(
                        {
                            attributes.AI_RESPONSE_ID: response.metadata.id,
                            attributes.AI_RESPONSE_PROVIDER: response.metadata.provider,
                            attributes.AI_RESPONSE_MODEL: response.metadata.model,
                            attributes.AI_RESPONSE_TIMESTAMP: response.metadata.timestamp.isoformat(),
                            attributes.AI_RESPONSE_FINISH_REASON: response.finish_reason,
                            attributes.AI_USAGE_INPUT_TOKENS: response.metadata.usage.input_tokens,
                            attributes.AI_USAGE_INPUT_TOKENS_DETAILS: ModalityTokenCountList.dump_json(
                                response.metadata.usage.input_tokens_details,
                                ensure_ascii=False,
                                exclude_unset=True,
                                exclude_none=True,
                            ).decode()
                            if response.metadata.usage.input_tokens_details
                            else None,
                            attributes.AI_USAGE_OUTPUT_TOKENS: response.metadata.usage.output_tokens,
                            attributes.AI_USAGE_THOUGHT_TOKENS: response.metadata.usage.thought_tokens,
                            attributes.AI_USAGE_CACHED_TOKENS: response.metadata.usage.cached_tokens,
                            attributes.AI_USAGE_CACHED_TOKENS_DETAILS: ModalityTokenCountList.dump_json(
                                response.metadata.usage.cached_tokens_details,
                                ensure_ascii=False,
                                exclude_unset=True,
                                exclude_none=True,
                            ).decode()
                            if response.metadata.usage.cached_tokens_details
                            else None,
                            attributes.AI_USAGE_TOTAL_TOKENS: response.metadata.usage.total_tokens,
                            attributes.AI_OUTPUT: ContentListAdapter.dump_json(
                                response.output,
                                ensure_ascii=False,
                                exclude_unset=True,
                                exclude_none=True,
                            ).decode(),
                        }
                    )
                except ModelRefusalError as e:
                    s.set_attributes(
                        {
                            attributes.AI_RESPONSE_ERROOR_TYPE: "refusal",
                            attributes.AI_RESPONSE_REFUSAL_REASON: e.reason,
                            attributes.AI_RESPONSE_REFUSAL_MESSAGE: e.message,
                        }
                    )
                    raise
                except ModelIncompleteError as e:
                    s.set_attributes(
                        {
                            attributes.AI_RESPONSE_ERROOR_TYPE: "incomplete",
                            attributes.AI_RESPONSE_INCOMPLETE_REASON: e.reason,
                            attributes.AI_RESPONSE_INCOMPLETE_MESSAGE: e.message,
                        }
                    )
                    raise
                except ModelBehaviorError:
                    s.set_attribute(attributes.AI_RESPONSE_ERROOR_TYPE, "internal")
                    raise
            responses.append(response)

            function_calls = [
                p.function_call
                for c in response.output
                for p in c.parts or []
                if p.function_call is not None
            ]

            if not function_calls:
                break

            step += 1
            if max_steps is not None and step >= max_steps:
                raise MaxStepsExceeded("max steps exceeded")

            function_responses = await agent.invoke_tools(function_calls)

            current_input = [
                Content(
                    role=Role.USER,
                    parts=[Part(function_response=i) for i in function_responses],
                )
            ]

    return AgentTurnResult(
        steps=responses,
    )


async def _run_agent[T](
    agent: Agent[T],
    input: ContentListLike,
    max_turns: int | None = None,
) -> AgentOutput[T]:
    turn = 0
    responses: list[AgentTurnResult] = []

    input_normalized = await as_content_list(input)
    instruction = await as_instruction_content(agent.instruction)

    while turn < (max_turns or 10):
        with tracer.span(
            operations.OPFER_AGENT_TURN,
            attributes={
                attributes.OPERATION_NAME: operations.OPFER_AGENT_TURN,
                attributes.DESCRIPTION: f"Running agent {agent.display_name} ({agent.id}): turn {turn}",
                attributes.AI_AGENT_RUN_TURN_NUMBER: turn,
            },
        ):
            response = await _run_agent_turn(
                turn,
                agent=agent,
                input=input_normalized,
                instruction=instruction,
            )
        responses.append(response)
        turn += 1
        # TODO: support multi turn
        break

    return AgentOutput(
        turns=responses,
    )
