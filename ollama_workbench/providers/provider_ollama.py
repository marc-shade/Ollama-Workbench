"""Ollama provider wrapper using the unified abstraction layer."""

import time
import logging
from typing import Dict, List, Any

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert a list of chat messages into a single prompt string.

    Ollama's generate endpoint expects a plain prompt, not a messages list.
    We join the messages with role labels so the model has context.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class OllamaProvider(BaseProvider):
    """Provider wrapper for the local Ollama server."""

    name = "ollama"

    def call(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.7, max_tokens: int = 4000,
             stream: bool = False, **kwargs) -> ProviderResponse:
        """Call Ollama via ``call_ollama_endpoint`` and return a ProviderResponse."""
        from .ollama_utils import call_ollama_endpoint

        prompt = _messages_to_prompt(messages)
        presence_penalty = kwargs.get("presence_penalty", 0.0)
        frequency_penalty = kwargs.get("frequency_penalty", 0.0)

        start = time.time()
        try:
            result = call_ollama_endpoint(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stream=False,
            )
            latency = time.time() - start

            # call_ollama_endpoint returns a 5-tuple:
            # (response_text, context, eval_count, eval_duration, metrics_dict)
            response_text, context, eval_count, eval_duration, metrics = result

            return ProviderResponse(
                text=response_text or "",
                output_tokens=eval_count or 0,
                latency=latency,
                raw={
                    "context": context,
                    "eval_count": eval_count,
                    "eval_duration": eval_duration,
                    "metrics": metrics,
                },
            )
        except Exception as e:
            latency = time.time() - start
            logger.error("OllamaProvider.call failed: %s", e)
            return ProviderResponse(error=str(e), latency=latency)

    def get_models(self) -> List[str]:
        """Return models available on the Ollama server."""
        try:
            from .ollama_utils import get_available_models
            return get_available_models()
        except Exception as e:
            logger.error("OllamaProvider.get_models failed: %s", e)
            return []

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            import requests
            from ollama_workbench.core.config import get_config
            config = get_config()
            host = config.get("OLLAMA_HOST", "http://localhost:11434")
            if not host.startswith("http://") and not host.startswith("https://"):
                host = f"http://{host}"
            resp = requests.get(f"{host}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
