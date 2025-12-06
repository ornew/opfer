# from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence, TypedDict

from pydantic import ValidationError

from opfer import attributes
from opfer.errors import (
    MaxStepsExceeded,
    ModelBehaviorError,
    ModelIncompleteError,
    ModelRefusalError,
    ToolError,
)
from opfer.inspect import FuncSchema, get_func_schema
from opfer.provider import get_model_provider_registry
from opfer.tracing import tracer
from opfer.types import (
    AgentOutput,
    AgentResponse,
    AgentTurnResult,
    Content,
    ContentLike,
    ContentListAdapter,
    ContentListLike,
    JsonValue,
    ModalityTokenCountList,
    ModelConfig,
    Part,
    PartFunctionCall,
    PartFunctionResponse,
    Role,
    as_content_list,
    as_instruction_content,
)


class ToolOutput(TypedDict):
    output: JsonValue


class ToolOutputError(TypedDict):
    error: JsonValue


type ToolResult = ToolOutput | ToolOutputError


def _func_schema_as_tool_definition(schema: FuncSchema) -> JsonValue:
    # TODO: implement
    return None


@dataclass
class Tool[**I, O]:
    _func: Callable[I, Awaitable[O]]
    schema: FuncSchema

    def __init__(
        self,
        func: Callable[I, Awaitable[O]],
        name: str | None = None,
        description: str | None = None,
    ):
        self._func = func
        self.schema = get_func_schema(
            func,
            name=name,
            description=description,
            # allow_positional_args=False,
        )

    async def call(self, call_id: str, *args: I.args, **kwargs: I.kwargs) -> ToolResult:
        with tracer.span(
            f"call tool {self.schema.name}",
            attributes={
                attributes.OPERATION_NAME: "tool_call",
                attributes.TOOL_NAME: self.schema.name,
                attributes.TOOL_DESCRIPTION: self.schema.description or "",
                attributes.TOOL_DEFINITION: _func_schema_as_tool_definition(
                    self.schema
                ),
                attributes.TOOL_CALL_ID: call_id,
            },
        ) as s:
            try:
                _args = {
                    **{
                        k: v
                        for k, v in zip(
                            self.schema.input_model.model_fields.keys(), args
                        )
                    },
                    **kwargs,
                }
                input = self.schema.input_model.model_validate(_args)
                s.set_attribute(
                    attributes.TOOL_CALL_INPUT,
                    input.model_dump(),
                )
            except ValidationError as e:
                s.record_exception(e)
                return {"error": "Invalid input: " + repr(e)}
            try:
                output = await self._func(*args, **kwargs)
                output_json = self.schema.output_type.dump_python(output)
                s.set_attribute(
                    attributes.TOOL_CALL_OUTPUT,
                    output_json,
                )
                return {"output": output_json}
            except ToolError as e:
                s.record_exception(e)
                return {"error": str(e)}


def tool(name: str | None = None, description: str | None = None):
    def decorator[**I, O](func: Callable[I, Awaitable[O]]) -> Tool[I, O]:
        return Tool[I, O](func, name=name, description=description)

    return decorator


type ToolLike = Tool | Callable[..., Awaitable[Any]]

_tool_cache_attr = "_opfer_tool_cache"


def as_tool(fn: ToolLike) -> Tool:
    if isinstance(fn, Tool):
        return fn
    cache = getattr(fn, _tool_cache_attr, None)
    if cache is not None:
        if not isinstance(cache, Tool):
            raise ValueError(f"invalid tool cache type: {type(cache)} of {fn}")
        return cache
    tool_instance = Tool(fn)
    setattr(fn, _tool_cache_attr, tool_instance)
    return tool_instance


class Agent[TOutput = str]:
    id: str
    display_name: str
    description: str | None
    instruction: ContentLike
    model: ModelConfig
    output_type: type[TOutput] | None

    tools: list[Tool]

    def __init__(
        self,
        *,
        id: str,
        display_name: str,
        model: ModelConfig,
        instruction: ContentLike,
        description: str | None = None,
        output_type: type[TOutput] | None = None,
        tools: Sequence[ToolLike] | None = None,
    ):
        self.id = id
        self.display_name = display_name
        self.description = description
        self.instruction = instruction
        self.model = model
        self.output_type = output_type
        self.tools = [as_tool(t) for t in tools or []]

    async def run(
        self,
        input: ContentListLike,
        max_turns: int | None = None,
    ) -> AgentOutput[TOutput]:
        with tracer.span(
            f"agent run: {self.id}",
            attributes={
                attributes.OPERATION_NAME: "agent_run",
                attributes.AI_AGENT_ID: self.id,
                attributes.AI_AGENT_NAME: self.display_name,
                attributes.AI_AGENT_DESCRIPTION: self.description,
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
    ) -> Tool:
        return agent_as_tool(self, name, description)

    def find_tool_by_name(self, name: str) -> Tool | None:
        for tool in self.tools:
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


def agent_as_tool[T](
    agent: Agent[T],
    name: str | None = None,
    description: str | None = None,
) -> Tool[[str], T]:
    async def tool_func(input: str) -> T:
        output = await agent.run(input)

        # print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))
        return output.final_output

    return tool(
        name=name or f"ask_to_subagent_{agent.id}",
        description=description,
    )(tool_func)


async def _run_agent_turn[TOutput](
    agent: Agent[TOutput],
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
            f"agent step {agent.id} ({step})",
            attributes={
                attributes.OPERATION_NAME: "agent_run_step",
                attributes.AI_AGENT_RUN_STEP_NUMBER: step,
            },
        ):
            # TODO: if step reaches max, disable tool calling.

            with tracer.span(
                f"agent chat {agent.model.provider}/{agent.model.name}",
                attributes={
                    attributes.OPERATION_NAME: "agent_chat",
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


async def _run_agent[TOutput](
    agent: Agent[TOutput],
    input: ContentListLike,
    max_turns: int | None = None,
) -> AgentOutput[TOutput]:
    turn = 0
    responses: list[AgentTurnResult] = []

    input_normalized = await as_content_list(input)
    instruction = await as_instruction_content(agent.instruction)

    while turn < (max_turns or 10):
        with tracer.span(
            f"agent turn: {agent.id} ({turn})",
            attributes={
                attributes.OPERATION_NAME: "agent_run_turn",
                attributes.AI_AGENT_RUN_TURN_NUMBER: turn,
            },
        ):
            response = await _run_agent_turn(
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
