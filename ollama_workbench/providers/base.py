"""Unified provider abstraction layer for all AI providers."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Generator
from abc import ABC, abstractmethod
import time


@dataclass
class ProviderResponse:
    """Unified response from any AI provider."""
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    error: Optional[str] = None
    raw: Any = None  # provider-specific raw response

    @property
    def ok(self) -> bool:
        return self.error is None and self.text != ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseProvider(ABC):
    """Abstract base for all AI providers."""

    name: str = "base"

    @abstractmethod
    def call(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.7, max_tokens: int = 4000,
             stream: bool = False, **kwargs) -> ProviderResponse:
        """Send a request to the provider and return a unified response."""
        ...

    @abstractmethod
    def get_models(self) -> List[str]:
        """Return list of available model names."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider is reachable / configured."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_providers: Dict[str, BaseProvider] = {}


def get_provider(name: str) -> BaseProvider:
    """Get a provider instance by name. Cached after first creation."""
    if name not in _providers:
        if name == "ollama":
            from .provider_ollama import OllamaProvider
            _providers[name] = OllamaProvider()
        elif name == "openai":
            from .provider_openai import OpenAIProvider
            _providers[name] = OpenAIProvider()
        elif name == "groq":
            from .provider_groq import GroqProvider
            _providers[name] = GroqProvider()
        elif name == "mistral":
            from .provider_mistral import MistralProvider
            _providers[name] = MistralProvider()
        else:
            raise ValueError(f"Unknown provider: {name}")
    return _providers[name]
