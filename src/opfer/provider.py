from __future__ import annotations

from contextvars import ContextVar, Token
from typing import (
    TYPE_CHECKING,
    Protocol,
)

from opfer.types import (
    AgentResponse,
    Content,
)

if TYPE_CHECKING:
    from opfer.core import Agent


class Chat(Protocol):
    async def send[TOutput](
        self,
        agent: Agent[TOutput],
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


_context_model_provider_registry = ContextVar[ModelProviderRegistry](
    "model_provider_registry"
)


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
