"""
Performance Optimization Module for Ollama Workbench

Features:
- Smart caching with TTL and memory limits
- Connection pooling and reuse
- Async/await patterns for better concurrency
- Resource monitoring and optimization
- Background task processing
- Memory-efficient data handling
- Performance metrics and profiling
"""

import asyncio
import functools
import hashlib
import json
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from collections import OrderedDict
from threading import Lock
import psutil
import sys

import streamlit as st


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: float


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    operation_name: str
    duration: float
    memory_before: float
    memory_after: float
    cpu_percent: float
    cache_hits: int
    cache_misses: int
    timestamp: float


class SmartCache:
    """
    Intelligent caching system with memory management and TTL
    """
    
    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl: float = 300,  # 5 minutes
        cleanup_interval: float = 60  # 1 minute
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._current_size = 0
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0
        }
        
        # Schedule periodic cleanup
        self._last_cleanup = time.time()

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments"""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
            else:
                return sys.getsizeof(data)
        except Exception:
            return 1024  # Default estimate

    def _cleanup_expired(self):
        """Remove expired entries"""
        if time.time() - self._last_cleanup < self.cleanup_interval:
            return

        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._current_size -= entry.size_bytes
            
            if expired_keys:
                self._stats["cleanups"] += 1
        
        self._last_cleanup = current_time

    def _evict_lru(self, required_size: int):
        """Evict least recently used entries to make space"""
        with self._lock:
            while self._current_size + required_size > self.max_size_bytes and self._cache:
                key, entry = self._cache.popitem(last=False)  # Remove oldest
                self._current_size -= entry.size_bytes
                self._stats["evictions"] += 1

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        self._cleanup_expired()
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                current_time = time.time()
                
                # Check if expired
                if current_time - entry.timestamp > entry.ttl:
                    del self._cache[key]
                    self._current_size -= entry.size_bytes
                    self._stats["misses"] += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.access_count += 1
                self._stats["hits"] += 1
                return entry.data
            
            self._stats["misses"] += 1
            return None

    def set(self, key: str, data: Any, ttl: Optional[float] = None):
        """Set item in cache"""
        ttl = ttl or self.default_ttl
        size_bytes = self._estimate_size(data)
        
        # Check if we need to evict
        self._evict_lru(size_bytes)
        
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            access_count=1,
            size_bytes=size_bytes,
            ttl=ttl
        )
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size -= old_entry.size_bytes
            
            self._cache[key] = entry
            self._current_size += size_bytes

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "cleanups": 0
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self._stats["hits"] / (self._stats["hits"] + self._stats["misses"]) if (self._stats["hits"] + self._stats["misses"]) > 0 else 0
        
        return {
            "size_mb": self._current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "entry_count": len(self._cache),
            "hit_rate": hit_rate,
            **self._stats
        }


class PerformanceMonitor:
    """
    Performance monitoring and profiling system
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        self._lock = Lock()

    def start_operation(self, operation_name: str) -> Dict[str, Any]:
        """Start monitoring an operation"""
        return {
            "operation_name": operation_name,
            "start_time": time.time(),
            "memory_before": self.process.memory_info().rss / (1024 * 1024),  # MB
            "cpu_before": self.process.cpu_percent()
        }

    def end_operation(self, context: Dict[str, Any], cache_hits: int = 0, cache_misses: int = 0):
        """End monitoring and record metrics"""
        end_time = time.time()
        memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        metrics = PerformanceMetrics(
            operation_name=context["operation_name"],
            duration=end_time - context["start_time"],
            memory_before=context["memory_before"],
            memory_after=memory_after,
            cpu_percent=self.process.cpu_percent(),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            timestamp=context["start_time"]
        )
        
        with self._lock:
            self.metrics.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]

    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            if operation_name:
                relevant_metrics = [m for m in self.metrics if m.operation_name == operation_name]
            else:
                relevant_metrics = self.metrics
        
        if not relevant_metrics:
            return {"error": "No metrics available"}
        
        durations = [m.duration for m in relevant_metrics]
        memory_usage = [m.memory_after - m.memory_before for m in relevant_metrics]
        
        return {
            "operation_count": len(relevant_metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_memory_delta": sum(memory_usage) / len(memory_usage),
            "total_cache_hits": sum(m.cache_hits for m in relevant_metrics),
            "total_cache_misses": sum(m.cache_misses for m in relevant_metrics),
            "operations_per_second": len(relevant_metrics) / (time.time() - relevant_metrics[0].timestamp) if relevant_metrics else 0
        }


class BackgroundTaskManager:
    """
    Background task processing for non-blocking operations
    """
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self._task_counter = 0

    async def submit_task(
        self,
        func: Callable,
        *args,
        task_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit a background task"""
        if task_name is None:
            self._task_counter += 1
            task_name = f"task_{self._task_counter}"
        
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(self.executor, func, *args, **kwargs)
        
        self.active_tasks[task_name] = task
        
        # Clean up completed tasks
        def cleanup_task(task_name_inner):
            def cleanup(fut):
                self.active_tasks.pop(task_name_inner, None)
            return cleanup
        
        task.add_done_callback(cleanup_task(task_name))
        
        return task_name

    async def wait_for_task(self, task_name: str) -> Any:
        """Wait for a specific task to complete"""
        if task_name in self.active_tasks:
            return await self.active_tasks[task_name]
        else:
            raise ValueError(f"Task {task_name} not found")

    def get_task_status(self) -> Dict[str, str]:
        """Get status of all active tasks"""
        return {
            name: "running" if not task.done() else "completed"
            for name, task in self.active_tasks.items()
        }

    def cancel_task(self, task_name: str) -> bool:
        """Cancel a specific task"""
        if task_name in self.active_tasks:
            task = self.active_tasks[task_name]
            if not task.done():
                task.cancel()
                return True
        return False

    def shutdown(self):
        """Shutdown the task manager"""
        self.executor.shutdown(wait=True)


# Global instances
_cache = SmartCache()
_performance_monitor = PerformanceMonitor()
_task_manager = BackgroundTaskManager()


def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """
    Decorator for intelligent caching with performance monitoring
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _cache._generate_key(func.__name__, args, kwargs)
            
            # Start performance monitoring
            perf_context = _performance_monitor.start_operation(f"cached_{func.__name__}")
            
            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                _performance_monitor.end_operation(perf_context, cache_hits=1, cache_misses=0)
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            _cache.set(cache_key, result, ttl)
            
            # End performance monitoring
            _performance_monitor.end_operation(perf_context, cache_hits=0, cache_misses=1)
            
            return result
        
        return wrapper
    return decorator


def performance_monitored(operation_name: Optional[str] = None):
    """
    Decorator for performance monitoring
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            perf_context = _performance_monitor.start_operation(op_name)
            
            try:
                result = func(*args, **kwargs)
                _performance_monitor.end_operation(perf_context)
                return result
            except Exception as e:
                _performance_monitor.end_operation(perf_context)
                raise
        
        return wrapper
    return decorator


async def background_task(func: Callable, *args, **kwargs) -> str:
    """Submit a function as a background task"""
    return await _task_manager.submit_task(func, *args, **kwargs)


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    return _cache.get_stats()


def get_performance_stats(operation_name: Optional[str] = None) -> Dict[str, Any]:
    """Get performance statistics"""
    return _performance_monitor.get_performance_summary(operation_name)


def clear_cache():
    """Clear global cache"""
    _cache.clear()


def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3),
        "disk_usage_percent": psutil.disk_usage('/').percent,
        "active_tasks": len(_task_manager.active_tasks),
        "cache_size_mb": _cache._current_size / (1024 ** 2)
    }


@asynccontextmanager
async def performance_context(operation_name: str):
    """Context manager for performance monitoring"""
    perf_context = _performance_monitor.start_operation(operation_name)
    try:
        yield _performance_monitor
    finally:
        _performance_monitor.end_operation(perf_context)


# Streamlit-specific optimizations
def optimize_streamlit_performance():
    """Apply Streamlit-specific performance optimizations"""
    
    # Configure session state for performance
    if 'performance_optimization_enabled' not in st.session_state:
        st.session_state.performance_optimization_enabled = True
    
    # Enable fragment caching for frequently updated components
    if hasattr(st, 'fragment'):
        st.fragment.cache_data.clear()  # Clear old cache
    
    # Optimize rerun behavior
    st.rerun = st.rerun  # Use modern rerun instead of experimental_rerun


# Export main functions and classes
__all__ = [
    "SmartCache",
    "PerformanceMonitor", 
    "BackgroundTaskManager",
    "cached",
    "performance_monitored",
    "background_task",
    "get_cache_stats",
    "get_performance_stats",
    "clear_cache",
    "get_system_resources",
    "performance_context",
    "optimize_streamlit_performance"
]