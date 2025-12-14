# Opfer

Opfer is a Python framework for building AI agent applications with workflow orchestration, tool integration, and observability.

## Features

- **Agent System**: Define AI agents with custom tools and instructions
- **Workflow Orchestration**: Build complex workflows with async task composition
- **Multi-Provider Support**: Integrate with various AI model providers (Google GenAI, etc.)
- **Observability**: Built-in tracing with OpenTelemetry support
- **Artifact Management**: Handle files and binary data with flexible storage backends
- **Agent Composition**: Use agents as tools in other agents

## Requirements

- Python 3.13+

## Installation

```bash
pip install opfer
```

### Optional Dependencies

For Google GenAI support:
```bash
pip install opfer[google_genai]
```

For OpenTelemetry tracing:
```bash
pip install opfer[opentelemetry]
```

## Quick Start

### Basic Agent

```python
from opfer.core import Agent
from opfer.types import ModelConfig

async def add(a: float, b: float) -> float:
    return a + b

async def multiply(a: float, b: float) -> float:
    return a * b

math_agent = Agent(
    id="math_agent",
    display_name="Math Agent",
    instruction="You are a math agent.",
    tools=[add, multiply],
    model=ModelConfig(
        provider="google",
        name="gemini-2.5-flash-lite",
    ),
)
```

### Agent Composition

Agents can use other agents as tools:

```python
assistant = Agent(
    id="assistant",
    display_name="Super Helpful Assistant",
    instruction="You are a helpful assistant.",
    tools=[
        math_agent.as_tool(
            description="Use this tool to answer math questions.",
        ),
    ],
    model=ModelConfig(
        provider="google",
        name="gemini-2.5-flash-lite",
    ),
)
```

### Workflow

Build complex workflows with async task composition:

```python
from opfer.workflow import Workflow

async def load_data() -> str:
    # Load and process data
    return "processed_data"

async def analyze(data: str) -> str:
    # Analyze data
    return f"Analysis of {data}"

async def my_workflow():
    async with Workflow() as w:
        data = w.run(load_data())
        result = data.then(lambda d: analyze(d))
        return await result
```

### Artifact Management

Handle files and binary data with artifact storage:

```python
import io

from PIL import Image

from opfer.artifacts import File, upload_artifact, download_artifact

async def process_image():
    # Load image
    image = Image.open("image.png")
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    # Upload as artifact
    file = File(
        name="image.png",
        data=buf.getvalue(),
        mime_type="image/png",
    )
    url = await upload_artifact(file)

    # Download artifact
    downloaded = await download_artifact(url)
    return downloaded
```

### Configuration

Set up storage backends and model providers:

```python
from contextlib import asynccontextmanager

from opfer.provider import DefaultModelProviderRegistry, set_model_provider_registry
from opfer.artifacts import set_artifact_storage
from opfer.blob import set_blob_storage

set_model_provider_registry(DefaultModelProviderRegistry())
set_artifact_storage(InMemoryArtifactStorage())
set_blob_storage(InMemoryBlobStorage())
```

### Observability

Enable tracing:

```python
from opfer.tracing import trace, tracer, get_current_span

# Add custom events to spans
async def monitored_function(a: float, b: float) -> float:
    span = get_current_span()
    if span:
        span.add_event("Function called", {"a": a, "b": b})
    return a * b

# Trace entire workflows
async def main():
    with trace():
        result = await my_workflow()
```

## Complete Example

See [examples/basic/example.py](./examples/basic/example.py) for a complete working example that demonstrates:

- Creating agents with custom tools
- Agent composition (assistant using math agent)
- Workflow orchestration with async tasks
- Image processing with artifact storage
- OpenTelemetry integration for tracing
- Google GenAI provider registration

To run the example:

```bash
python examples/basic/example.py
```

## Architecture

### Core Components

- **Agent**: AI agents with custom tools and instructions
- **Workflow**: Async task orchestration with composable operations
- **Provider**: Abstraction for AI model providers
- **Artifacts**: File and data management with pluggable storage
- **Blob**: Binary data storage with content-addressable URLs
- **Tracing**: Observability with span-based tracing

### Storage Backends

Opfer provides flexible storage backends for artifacts and blobs:

- In-memory storage (for development/testing)
- Custom storage implementations (implement the storage interface)

### Model Providers

Currently supported:
- Google GenAI (via `google-genai` package)
