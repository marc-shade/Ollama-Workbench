"""
Enhanced Ollama Client with Modern Features
- Async/await support for better performance
- Connection pooling for efficiency
- Retry mechanisms with exponential backoff
- Enhanced error handling and logging
- Performance monitoring and metrics
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
import httpx
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure structured logging
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enumeration of supported model providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GROQ = "groq"
    MISTRAL = "mistral"


@dataclass
class ModelInfo:
    """Enhanced model information with metadata"""
    name: str
    provider: ModelProvider
    size: Optional[str] = None
    family: Optional[str] = None
    parameter_count: Optional[str] = None
    quantization: Optional[str] = None
    capabilities: Optional[List[str]] = None
    context_length: Optional[int] = None
    last_updated: Optional[str] = None


@dataclass
class GenerationMetrics:
    """Metrics for model generation performance"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float
    tokens_per_second: float
    first_token_latency: Optional[float] = None
    model_name: str = ""
    timestamp: float = 0.0


class EnhancedOllamaClient:
    """
    Enhanced Ollama client with modern async patterns and reliability features
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        max_retries: int = 3,
        connection_pool_size: int = 10
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.connection_pool_size = connection_pool_size
        
        # Initialize clients
        self._sync_client = ollama.Client(host=base_url)
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Performance tracking
        self._metrics: List[GenerationMetrics] = []
        self._connection_pool_stats = {
            "active_connections": 0,
            "total_requests": 0,
            "failed_requests": 0
        }
        
        logger.info(f"Initialized EnhancedOllamaClient with base_url={base_url}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_async_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._cleanup_async_client()

    async def _initialize_async_client(self):
        """Initialize async HTTP client with connection pooling"""
        if self._http_client is None:
            limits = httpx.Limits(
                max_keepalve_connections=self.connection_pool_size,
                max_connections=self.connection_pool_size * 2
            )
            
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=limits,
                http2=True  # Enable HTTP/2 for better performance
            )
            
            logger.debug("Initialized async HTTP client with connection pooling")

    async def _cleanup_async_client(self):
        """Cleanup async HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.debug("Cleaned up async HTTP client")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, httpx.HTTPError))
    )
    async def list_models_async(self) -> List[ModelInfo]:
        """
        Asynchronously list available models with enhanced metadata
        """
        try:
            await self._initialize_async_client()
            
            response = await self._http_client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                model_info = ModelInfo(
                    name=model_data.get("name", ""),
                    provider=ModelProvider.OLLAMA,
                    size=self._format_size(model_data.get("size", 0)),
                    family=model_data.get("details", {}).get("family", ""),
                    parameter_count=model_data.get("details", {}).get("parameter_size", ""),
                    quantization=model_data.get("details", {}).get("quantization_level", ""),
                    context_length=model_data.get("details", {}).get("context_length"),
                    last_updated=model_data.get("modified_at", "")
                )
                models.append(model_info)
            
            logger.info(f"Retrieved {len(models)} models from Ollama")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            self._connection_pool_stats["failed_requests"] += 1
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate_async(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Asynchronously generate text with performance monitoring
        """
        start_time = time.time()
        first_token_time = None
        prompt_tokens = len(prompt.split())  # Rough estimation
        
        try:
            await self._initialize_async_client()
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }
            
            self._connection_pool_stats["total_requests"] += 1
            
            if stream:
                return self._stream_generate_async(payload, start_time, prompt_tokens, model)
            else:
                response = await self._http_client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                end_time = time.time()
                result = response.json()
                
                # Calculate metrics
                completion_tokens = len(result.get("response", "").split())
                total_tokens = prompt_tokens + completion_tokens
                generation_time = end_time - start_time
                tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
                
                metrics = GenerationMetrics(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    generation_time=generation_time,
                    tokens_per_second=tokens_per_second,
                    model_name=model,
                    timestamp=start_time
                )
                
                self._metrics.append(metrics)
                logger.info(f"Generated {completion_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
                
                return result
                
        except Exception as e:
            logger.error(f"Generation failed for model {model}: {e}")
            self._connection_pool_stats["failed_requests"] += 1
            raise

    async def _stream_generate_async(
        self,
        payload: Dict[str, Any],
        start_time: float,
        prompt_tokens: int,
        model: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming generation with metrics"""
        completion_tokens = 0
        first_token_time = None
        
        try:
            async with self._http_client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            
                            # Track first token latency
                            if first_token_time is None and chunk.get("response"):
                                first_token_time = time.time() - start_time
                            
                            if chunk.get("response"):
                                completion_tokens += len(chunk["response"].split())
                            
                            yield chunk
                            
                            if chunk.get("done", False):
                                # Final metrics calculation
                                end_time = time.time()
                                total_tokens = prompt_tokens + completion_tokens
                                generation_time = end_time - start_time
                                tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
                                
                                metrics = GenerationMetrics(
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=total_tokens,
                                    generation_time=generation_time,
                                    tokens_per_second=tokens_per_second,
                                    first_token_latency=first_token_time,
                                    model_name=model,
                                    timestamp=start_time
                                )
                                
                                self._metrics.append(metrics)
                                logger.info(f"Streamed {completion_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
                                break
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse streaming response: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    async def health_check_async(self) -> Dict[str, Any]:
        """
        Async health check with detailed status information
        """
        try:
            await self._initialize_async_client()
            
            start_time = time.time()
            response = await self._http_client.get(f"{self.base_url}/api/tags")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                models_count = len(response.json().get("models", []))
                
                return {
                    "status": "healthy",
                    "response_time": response_time,
                    "models_available": models_count,
                    "connection_pool_stats": self._connection_pool_stats.copy(),
                    "base_url": self.base_url,
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time": response_time,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics"""
        if not self._metrics:
            return {"error": "No metrics available"}
        
        recent_metrics = self._metrics[-100:]  # Last 100 generations
        
        avg_tokens_per_second = sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
        avg_generation_time = sum(m.generation_time for m in recent_metrics) / len(recent_metrics)
        total_tokens = sum(m.total_tokens for m in recent_metrics)
        
        first_token_latencies = [m.first_token_latency for m in recent_metrics if m.first_token_latency]
        avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0
        
        return {
            "total_generations": len(self._metrics),
            "recent_generations": len(recent_metrics),
            "average_tokens_per_second": avg_tokens_per_second,
            "average_generation_time": avg_generation_time,
            "average_first_token_latency": avg_first_token_latency,
            "total_tokens_processed": total_tokens,
            "connection_pool_stats": self._connection_pool_stats.copy(),
            "timestamp": time.time()
        }

    def clear_metrics(self):
        """Clear performance metrics history"""
        self._metrics.clear()
        logger.info("Performance metrics cleared")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in bytes to human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"


# Global client instance with connection pooling
_global_client: Optional[EnhancedOllamaClient] = None


async def get_enhanced_client() -> EnhancedOllamaClient:
    """Get or create global enhanced Ollama client"""
    global _global_client
    
    if _global_client is None:
        _global_client = EnhancedOllamaClient()
        await _global_client._initialize_async_client()
    
    return _global_client


@asynccontextmanager
async def ollama_client_context():
    """Context manager for enhanced Ollama client"""
    client = await get_enhanced_client()
    try:
        yield client
    finally:
        # Cleanup handled by global client lifecycle
        pass


# Backward compatibility functions
def call_ollama_api_enhanced(model: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Enhanced synchronous wrapper for backward compatibility
    """
    async def _async_call():
        async with ollama_client_context() as client:
            return await client.generate_async(model, prompt, **kwargs)
    
    try:
        # Run async function in event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _async_call())
                return future.result()
        else:
            return loop.run_until_complete(_async_call())
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(_async_call())


# Export main classes and functions
__all__ = [
    "EnhancedOllamaClient",
    "ModelInfo", 
    "GenerationMetrics",
    "ModelProvider",
    "get_enhanced_client",
    "ollama_client_context",
    "call_ollama_api_enhanced"
]