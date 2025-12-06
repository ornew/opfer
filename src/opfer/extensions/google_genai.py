from datetime import datetime
from typing import Any

import google.genai
import google.genai.client
import google.genai.errors
import google.genai.types
from pydantic import BaseModel

from opfer.blob import Blob, upload_blob
from opfer.core import Agent, Tool
from opfer.errors import (
    ModelBehaviorError,
    ModelIncompleteError,
    ModelRefusalError,
    RetryableError,
    UserError,
)
from opfer.logging import logger
from opfer.provider import Chat, ModelProvider
from opfer.retry import retry
from opfer.types import (
    AgentResponse,
    AgentResponseMetadata,
    Content,
    MediaResolution,
    MediaResolutionLevel,
    ModalityTokenCount,
    Part,
    PartBlob,
    PartFunctionCall,
    PartFunctionResponse,
    PartFunctionResponsePart,
    PartFunctionResponsePartBlob,
    PartText,
    PartThought,
    Role,
    TokenModality,
    Usage,
    download_part_blob,
)


def as_modality_token_counts(
    details: list[google.genai.types.ModalityTokenCount],
):
    res: list[ModalityTokenCount] = []
    for mtc in details:
        modality = TokenModality.UNSPECIFIED
        match mtc.modality:
            case google.genai.types.MediaModality.TEXT:
                modality = TokenModality.TEXT
            case google.genai.types.MediaModality.IMAGE:
                modality = TokenModality.IMAGE
            case google.genai.types.MediaModality.AUDIO:
                modality = TokenModality.AUDIO
            case google.genai.types.MediaModality.VIDEO:
                modality = TokenModality.VIDEO
            case google.genai.types.MediaModality.DOCUMENT:
                modality = TokenModality.DOCUMENT
        res.append(
            ModalityTokenCount(
                modality=modality,
                tokens=mtc.token_count or 0,
            )
        )
    return res


class GoogleModelProvider(ModelProvider):
    client: google.genai.client.AsyncClient

    def __init__(
        self,
        client: google.genai.client.AsyncClient,
        name: str | None = None,
    ):
        self.client = client
        self._name = name

    @property
    def name(self) -> str:
        return self._name or "google"

    async def chat(self) -> Chat:
        return GoogleAgentProviderChat(provider=self)


class GoogleAgentProviderChat(Chat):
    def __init__(self, provider: GoogleModelProvider):
        self._provider = provider
        self._cached_encoded_history: list[google.genai.types.ContentUnion] = []
        self._history: list[Content] = []

    @property
    def history(self) -> list[Content]:
        return self._history

    async def send[TOutput](
        self,
        agent: Agent[TOutput],
        input: list[Content],
        instruction: Content | None = None,
    ) -> AgentResponse:
        system_instruction_encoded = (
            await self.encode_content(instruction) if instruction is not None else None
        )
        contents = await self.encode_content_list(input)
        self._cached_encoded_history.extend(contents)
        self._history = self._history + list(input)  # No extend

        raw_response = await self._generate(
            agent=agent,
            contents=self._cached_encoded_history,
            system_instruction=system_instruction_encoded,
        )

        usage = raw_response.usage_metadata

        if usage is None:
            raise ModelBehaviorError("no usage metadata in response")

        input_tokens_details = as_modality_token_counts(
            usage.prompt_tokens_details or []
        )
        cached_tokens_details = as_modality_token_counts(
            usage.cache_tokens_details or []
        )
        metadata = AgentResponseMetadata(
            provider=self._provider.name,
            id=raw_response.response_id,
            model=raw_response.model_version or agent.model.name,
            timestamp=raw_response.create_time or datetime.now(),
            usage=Usage(
                total_tokens=usage.total_token_count,
                input_tokens=usage.prompt_token_count,
                input_tokens_details=input_tokens_details,
                output_tokens=usage.candidates_token_count,
                thought_tokens=usage.thoughts_token_count,
                cached_tokens=usage.cached_content_token_count,
                cached_tokens_details=cached_tokens_details,
            ),
        )

        # TODO:
        # hook.on_model_response(metadata)

        prompt_feedback = raw_response.prompt_feedback

        if prompt_feedback is not None:
            block_reason = (
                prompt_feedback.block_reason
                or google.genai.types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
            )
            raise ModelRefusalError(
                reason=block_reason.name,
                message=f"prompt blocked: {prompt_feedback.block_reason_message or "no message"}, safety_ratings={prompt_feedback.safety_ratings}",
            )

        if raw_response.candidates is None or len(raw_response.candidates) == 0:
            raise ModelBehaviorError("no candidates in response")

        if len(raw_response.candidates) > 1:
            logger.warning("more than one candidate in response, will use first")

        candidate = raw_response.candidates[0]

        finish_message = candidate.finish_message
        finish_reason = (
            candidate.finish_reason
            or google.genai.types.FinishReason.FINISH_REASON_UNSPECIFIED
        )

        match finish_reason:
            case google.genai.types.FinishReason.STOP:
                pass
            case google.genai.types.FinishReason.MAX_TOKENS:
                raise ModelIncompleteError(
                    reason=finish_reason.name,
                    message=f"max tokens reached: {finish_message}",
                )
            case _:
                raise ModelRefusalError(
                    reason=finish_reason.name,
                    message=f"{finish_message or "no message"}",
                )

        if candidate.content is None:
            raise ModelBehaviorError("no content in candidate")

        output = await self.decode_content(candidate.content)

        if output.parts is None:
            raise ModelBehaviorError("no parts in output")

        parsed: Any = output.text
        if agent.output_type is not None:
            if isinstance(agent.output_type, BaseModel):
                try:
                    parsed = agent.output_type.model_validate_json(output.text)
                except Exception as e:
                    raise ModelBehaviorError(
                        f"failed to parse model output: {e}"
                    ) from e
            else:
                raise UserError("invalid agent output type")

        response = AgentResponse(
            metadata=metadata,
            instruction=instruction,
            input=self._history,
            output=[output],
            finish_reason=finish_reason.name,
            parsed=parsed,
        )

        self._cached_encoded_history.append(candidate.content)
        self._history.append(output)

        return response

    def as_function_declaration(
        self, tool: Tool
    ) -> google.genai.types.FunctionDeclaration:
        return google.genai.types.FunctionDeclaration(
            name=tool.schema.name,
            description=tool.schema.description,
            parameters_json_schema=google.genai.types.Schema.model_validate(
                tool.schema.input_schema,
            ),
        )

    @retry(errors=(RetryableError,))
    async def _generate[TOutput](
        self,
        agent: Agent[TOutput],
        contents: google.genai.types.ContentListUnion,
        system_instruction: google.genai.types.ContentUnion | None,
    ):
        config = agent.model
        mime_type = None
        json_schema = None
        if agent.output_type is not None:
            if isinstance(agent.output_type, BaseModel):
                mime_type = "application/json"
                json_schema = agent.output_type.model_json_schema()
            else:
                raise UserError("invalid agent output type")

        tools: list[google.genai.types.Tool] = []
        if agent.tools:
            tools.append(
                google.genai.types.Tool(
                    function_declarations=[
                        self.as_function_declaration(tool) for tool in agent.tools
                    ],
                )
            )

        try:
            output = await self._provider.client.models.generate_content(
                model=config.name,
                contents=contents,
                config=google.genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    candidate_count=1,
                    stop_sequences=config.stop_sequences,
                    response_logprobs=config.response_logprobs,
                    logprobs=config.logprobs,
                    presence_penalty=config.presence_penalty,
                    frequency_penalty=config.frequency_penalty,
                    seed=config.seed,
                    automatic_function_calling=google.genai.types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                    response_mime_type=mime_type,
                    response_json_schema=json_schema,
                    thinking_config=google.genai.types.ThinkingConfig(
                        include_thoughts=True,
                    ),
                    tools=tools,
                ),
            )
            if (
                output
                and output.candidates
                and output.candidates[0].content
                and output.candidates[0].content.parts is None
            ):
                logger.warning(
                    "model no returned parts in response, this might Gemini API bug, we retry request."
                )
                raise RetryableError("empty parts")
            return output
        except google.genai.errors.APIError as e:
            if e.code in {
                429,
                500,
                502,
                503,
                504,
            }:
                logger.warning(
                    f"transient error from Gemini API (code {e.code}), retrying: {e}"
                )
                raise RetryableError() from e
            raise

    def encode_role(self, role: Role) -> str:
        match role:
            case Role.USER:
                return "user"
            case Role.MODEL:
                return "model"

    def encode_part_media_resolution_level(
        self, mr: MediaResolutionLevel
    ) -> google.genai.types.PartMediaResolutionLevel:
        match mr:
            case MediaResolutionLevel.UNSPECIFIED:
                return google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_UNSPECIFIED
            case MediaResolutionLevel.LOW:
                return google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
            case MediaResolutionLevel.MEDIUM:
                return (
                    google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM
                )
            case MediaResolutionLevel.HIGH:
                return google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH

    async def encode_part(self, part: Part) -> google.genai.types.Part:
        match t := part.type:
            case PartText():
                return google.genai.types.Part(
                    text=t.text,
                    thought_signature=part.thought_signature,
                )
            case PartThought():
                return google.genai.types.Part(
                    text=t.text,
                    thought=True,
                    thought_signature=part.thought_signature,
                )
            case PartBlob():
                blob = await download_part_blob(t)
                return google.genai.types.Part(
                    thought_signature=part.thought_signature,
                    inline_data=google.genai.types.Blob(
                        data=blob.data,
                        mime_type=blob.mime_type,
                    ),
                    media_resolution=google.genai.types.PartMediaResolution(
                        level=self.encode_part_media_resolution_level(
                            part.media_resolution.level
                        )
                        if part.media_resolution.level is not None
                        else None,
                        num_tokens=part.media_resolution.num_tokens,
                    )
                    if part.media_resolution is not None
                    else None,
                )
            case PartFunctionCall():
                return google.genai.types.Part(
                    thought_signature=part.thought_signature,
                    function_call=google.genai.types.FunctionCall(
                        id=t.id,
                        name=t.name,
                        args=t.arguments,
                    )
                    if part is not None
                    else None,
                )
            case PartFunctionResponse():
                return google.genai.types.Part(
                    function_response=google.genai.types.FunctionResponse(
                        id=t.id,
                        name=t.name,
                        response=t.response,
                        parts=[
                            google.genai.types.FunctionResponsePart(
                                inline_data=await self.encode_function_response_blob(
                                    part.blob
                                )
                                if part.blob is not None
                                else None,
                            )
                            for part in t.parts
                        ]
                        if t.parts is not None
                        else None,
                    )
                    if part is not None
                    else None,
                )

    async def encode_function_response_blob(
        self, blob: PartFunctionResponsePartBlob
    ) -> google.genai.types.FunctionResponseBlob:
        resolved = await download_part_blob(blob)
        return google.genai.types.FunctionResponseBlob(
            data=resolved.data,
            mime_type=resolved.mime_type,
        )

    async def encode_content(
        self,
        content: Content,
    ) -> google.genai.types.Content:
        return google.genai.types.Content(
            role=self.encode_role(content.role) if content.role is not None else None,
            parts=[await self.encode_part(p) for p in content.parts]
            if content.parts is not None
            else None,
        )

    async def encode_content_list(
        self,
        contents: list[Content],
    ) -> list[google.genai.types.ContentUnion]:
        return [await self.encode_content(c) for c in contents]

    def decode_role(self, role: str) -> Role:
        match role:
            case "user":
                return Role.USER
            case "model":
                return Role.MODEL
            case _:
                raise ValueError(f"unknown role: {role}")

    async def decode_part(self, part: google.genai.types.Part) -> Part:
        if part.thought:
            return Part(
                thought_signature=part.thought_signature,
                thought=PartThought(
                    text=part.text,
                    is_summary=True,
                ),
            )
        if part.text is not None:
            return Part(
                thought_signature=part.thought_signature,
                text=PartText(
                    text=part.text,
                ),
            )
        if (d := part.inline_data) is not None:
            if d.mime_type is None:
                raise RuntimeError("inline data mime type is None")
            if d.mime_type.startswith("image/"):
                blob = Blob(
                    mime_type=d.mime_type,
                    data=d.data or b"",
                )
                blob_url = await upload_blob(blob)
                return Part(
                    thought_signature=part.thought_signature,
                    media_resolution=MediaResolution(
                        level=self.decode_part_media_resolution_level(
                            part.media_resolution.level
                        )
                        if part.media_resolution is not None
                        and part.media_resolution.level is not None
                        else None,
                        num_tokens=part.media_resolution.num_tokens,
                    )
                    if part.media_resolution is not None
                    else None,
                    blob=PartBlob(
                        mime_type=d.mime_type,
                        url=blob_url,
                        content_md5=blob.content_md5.decode(),
                    ),
                )
            raise RuntimeError(f"unsupported inline data mime type: {d.mime_type}")
        if part.function_call is not None:
            return Part(
                thought_signature=part.thought_signature,
                function_call=PartFunctionCall(
                    id=part.function_call.id or "",
                    name=part.function_call.name or "",
                    arguments=part.function_call.args,
                ),
            )
        if part.function_response is not None:

            async def _as_part_blob(
                blob: google.genai.types.FunctionResponseBlob,
            ) -> PartFunctionResponsePartBlob:
                if blob.mime_type is None:
                    raise RuntimeError("function response blob mime type is None")
                resolved_blob = Blob(
                    mime_type=blob.mime_type,
                    data=blob.data or b"",
                )
                blob_url = await upload_blob(resolved_blob)
                return PartFunctionResponsePartBlob(
                    mime_type=blob.mime_type,
                    url=blob_url,
                    content_md5=resolved_blob.content_md5.decode(),
                )

            return Part(
                function_response=PartFunctionResponse(
                    id=part.function_response.id or "",
                    name=part.function_response.name or "",
                    response=part.function_response.response,
                    parts=[
                        PartFunctionResponsePart(
                            blob=await _as_part_blob(p.inline_data)
                            if p.inline_data is not None
                            else None,
                        )
                        for p in part.function_response.parts
                    ]
                    if part.function_response.parts is not None
                    else None,
                ),
            )
        raise RuntimeError("unknown part type")

    def decode_part_media_resolution_level(
        self, mr: google.genai.types.PartMediaResolutionLevel
    ) -> MediaResolutionLevel:
        match mr:
            case google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_UNSPECIFIED:
                return MediaResolutionLevel.UNSPECIFIED
            case google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW:
                return MediaResolutionLevel.LOW
            case google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM:
                return MediaResolutionLevel.MEDIUM
            case google.genai.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH:
                return MediaResolutionLevel.HIGH

    async def decode_content(self, content: google.genai.types.Content) -> Content:
        return Content(
            role=self.decode_role(content.role) if content.role is not None else None,
            parts=[await self.decode_part(p) for p in content.parts]
            if content.parts is not None
            else None,
        )
