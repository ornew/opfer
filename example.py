from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager

from PIL import Image

from opfer.artifacts import (
    File,
    download_artifact,
    reset_artifact_storage,
    set_artifact_storage,
    upload_artifact,
)
from opfer.blob import Blob, reset_blob_storage, set_blob_storage
from opfer.core import Agent
from opfer.logging import logger
from opfer.provider import (
    DefaultModelProviderRegistry,
    get_model_provider_registry,
    reset_model_provider_registry,
    set_model_provider_registry,
)
from opfer.tracing import (
    get_current_span,
    trace,
    tracer,
)
from opfer.types import ModelConfig
from opfer.workflow import Workflow


class InMemoryBlobStorage:
    data: dict[str, Blob]

    def __init__(self):
        self.data = {}

    async def exists(self, url: str) -> bool:
        return url in self.data

    async def download(self, url: str) -> Blob:
        return self.data[url]

    async def upload(self, blob: Blob) -> str:
        url = f"blob://{blob.content_md5.decode()}"
        self.data[url] = blob
        return url


class InMemoryArtifactStorage:
    data: dict[str, File]

    def __init__(self):
        self.data = {}

    async def exists(self, url: str) -> bool:
        return url in self.data

    async def download(self, url: str) -> File:
        return self.data[url]

    async def upload(self, file: File) -> str:
        url = f"artifacts://{file.name}"
        self.data[url] = file
        return url


async def add(a: float, b: float) -> float:
    return a + b


async def sub(a: float, b: float) -> float:
    return a - b


async def mul(a: float, b: float) -> float:
    s = get_current_span()
    if s:
        s.add_event("Called mul function", {"a": a, "b": b})
    return a * b


async def div(a: float, b: float) -> float:
    return a / b


math_v1 = Agent(
    id="math_v1",
    display_name="Math Agent",
    instruction="""
You are a math agent.
""",
    tools=[add, sub, mul, div],
    model=ModelConfig(
        provider="google",
        name="gemini-2.5-flash-lite",
    ),
)


assistant_v1 = Agent(
    id="assistant_v1",
    display_name="Super Helpful Assistant",
    instruction="""
You MUST ALWAYS respond after tool calls. Don't empty response.
You are a helpful assistant.
""",
    tools=[
        math_v1.as_tool(
            description="Use this tool to answer math questions.",
        ),
    ],
    model=ModelConfig(
        provider="google",
        name="gemini-2.5-flash-lite",
    ),
)


async def load_image() -> str:
    filename = "3.png"
    image = Image.open(filename)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    file = File(
        name=filename,
        data=buf.getvalue(),
        mime_type="image/pil",
    )
    url = await upload_artifact(file)
    return url


async def help_about_image(image_url: str, question: str):
    image_file = await download_artifact(image_url)
    image = Image.open(io.BytesIO(image_file.data))
    res = await assistant_v1.run(
        [
            image,
            question,
        ]
    )
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
    return res.final_output


async def my_workflow():
    async with Workflow() as w:
        url = w.run(load_image())
        answer = url.then(
            lambda u: help_about_image(u, "この画像の数字を３倍すると何になる？")
        )
        return await answer


@asynccontextmanager
async def trace_opentelemetry():
    import opentelemetry.sdk.resources
    import opentelemetry.sdk.trace
    import opentelemetry.sdk.trace.export
    import opentelemetry.trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    from opfer.extensions.opentelemetry import OtelSpanProcessor

    resource = opentelemetry.sdk.resources.Resource.create(
        attributes={
            opentelemetry.sdk.resources.SERVICE_NAME: "opfer-example",
        }
    )

    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
    processor = opentelemetry.sdk.trace.export.BatchSpanProcessor(otlp_exporter)
    tracer_provider = opentelemetry.sdk.trace.TracerProvider(resource=resource)
    tracer_provider.add_span_processor(processor)
    opentelemetry.trace.set_tracer_provider(tracer_provider)
    otel_tracer = opentelemetry.trace.get_tracer(__name__)
    tracer.add_span_processor(OtelSpanProcessor(otel_tracer))
    yield
    tracer_provider.shutdown()


@asynccontextmanager
async def configure_logging():
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)
    yield


@asynccontextmanager
async def configure_opfer_base():
    # since 3.14
    # with (
    #     set_model_provider_registry(...),
    #     set_artifact_storage(...),
    #     set_blob_storage(...),
    # ):
    #     yield
    a = set_model_provider_registry(DefaultModelProviderRegistry())
    b = set_artifact_storage(InMemoryArtifactStorage())
    c = set_blob_storage(InMemoryBlobStorage())
    yield
    reset_model_provider_registry(a)
    reset_artifact_storage(b)
    reset_blob_storage(c)


@asynccontextmanager
async def register_google_genai():
    import google.genai

    from opfer.extensions.google_genai import GoogleModelProvider

    google_genai_client = google.genai.Client().aio
    async with google_genai_client:
        model_provider_registry = get_model_provider_registry()
        model_provider_registry.register(
            GoogleModelProvider(client=google_genai_client)
        )
        yield google_genai_client


async def main():
    async with (
        configure_logging(),
        trace_opentelemetry(),
        configure_opfer_base(),
        register_google_genai(),
    ):
        print("Running workflow...")
        with trace():
            result = await my_workflow()
        print(f"Workflow result: {result}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
