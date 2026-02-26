"""OpenAI provider wrapper using the unified abstraction layer."""

import time
import logging
from typing import Dict, List, Any

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Provider wrapper for the OpenAI API."""

    name = "openai"

    def _get_api_key(self) -> str:
        """Resolve the OpenAI API key from stored configuration."""
        from .ollama_utils import load_api_keys
        return load_api_keys().get("openai_api_key", "")

    def call(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.7, max_tokens: int = 4000,
             stream: bool = False, **kwargs) -> ProviderResponse:
        """Call OpenAI and return a ProviderResponse."""
        from .openai_utils import call_openai_api

        api_key = kwargs.get("openai_api_key") or self._get_api_key()
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        presence_penalty = kwargs.get("presence_penalty", 0.0)

        start = time.time()
        try:
            result = call_openai_api(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=False,
                openai_api_key=api_key,
            )
            latency = time.time() - start

            if result is None:
                return ProviderResponse(
                    error="OpenAI API returned None",
                    latency=latency,
                )

            return ProviderResponse(
                text=result,
                latency=latency,
                raw=result,
            )
        except Exception as e:
            latency = time.time() - start
            logger.error("OpenAIProvider.call failed: %s", e)
            return ProviderResponse(error=str(e), latency=latency)

    def get_models(self) -> List[str]:
        """Return available OpenAI models (fetched from API with fallback)."""
        from .openai_utils import get_openai_models
        return get_openai_models()

    def is_available(self) -> bool:
        """Check whether an OpenAI API key is configured."""
        return bool(self._get_api_key())
