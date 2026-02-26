"""Mistral provider wrapper using the unified abstraction layer."""

import time
import logging
from typing import Dict, List, Any

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


class MistralProvider(BaseProvider):
    """Provider wrapper for the Mistral API."""

    name = "mistral"

    def _get_api_key(self) -> str:
        """Resolve the Mistral API key from stored configuration."""
        from .ollama_utils import load_api_keys
        return load_api_keys().get("mistral_api_key", "")

    def call(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.7, max_tokens: int = 4000,
             stream: bool = False, **kwargs) -> ProviderResponse:
        """Call Mistral and return a ProviderResponse."""
        from .mistral_utils import call_mistral_api

        api_key = kwargs.get("mistral_api_key") or self._get_api_key()

        start = time.time()
        try:
            result = call_mistral_api(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                mistral_api_key=api_key,
            )
            latency = time.time() - start

            if result is None:
                return ProviderResponse(
                    error="Mistral API returned None",
                    latency=latency,
                )

            return ProviderResponse(
                text=result,
                latency=latency,
                raw=result,
            )
        except Exception as e:
            latency = time.time() - start
            logger.error("MistralProvider.call failed: %s", e)
            return ProviderResponse(error=str(e), latency=latency)

    def get_models(self) -> List[str]:
        """Return available Mistral models (fetched from API with fallback)."""
        from .mistral_utils import get_mistral_models
        return get_mistral_models()

    def is_available(self) -> bool:
        """Check whether a Mistral API key is configured."""
        return bool(self._get_api_key())
