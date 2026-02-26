"""Groq provider wrapper using the unified abstraction layer."""

import time
import logging
from typing import Dict, List, Any

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    """Provider wrapper for the Groq API."""

    name = "groq"

    def _get_api_key(self) -> str:
        """Resolve the Groq API key from stored configuration."""
        from .ollama_utils import load_api_keys
        return load_api_keys().get("groq_api_key", "")

    def call(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.7, max_tokens: int = 4000,
             stream: bool = False, **kwargs) -> ProviderResponse:
        """Call Groq and return a ProviderResponse."""
        from .groq_utils import call_groq_api

        api_key = kwargs.get("groq_api_key") or self._get_api_key()

        start = time.time()
        try:
            result = call_groq_api(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=api_key,
            )
            latency = time.time() - start

            if result is None:
                return ProviderResponse(
                    error="Groq API returned None",
                    latency=latency,
                )

            return ProviderResponse(
                text=result,
                latency=latency,
                raw=result,
            )
        except Exception as e:
            latency = time.time() - start
            logger.error("GroqProvider.call failed: %s", e)
            return ProviderResponse(error=str(e), latency=latency)

    def get_models(self) -> List[str]:
        """Return available Groq models (fetched from API with fallback)."""
        from .groq_utils import get_groq_models
        return get_groq_models()

    def is_available(self) -> bool:
        """Check whether a Groq API key is configured."""
        return bool(self._get_api_key())
