# observability/opik_integration.py

"""
Opik integration module for Ollama Workbench
Provides comprehensive observability for LLM operations, RAG systems, and agentic workflows.
"""

import os
import logging
import functools
# import asyncio  # Not currently used
import json
import time
import threading
from collections import defaultdict, deque
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Performance and alerting thresholds
DEFAULT_THRESHOLDS = {
    "slow_response_time": 5.0,
    "very_slow_response_time": 10.0,
    "high_error_rate": 0.05,  # 5%
    "very_high_error_rate": 0.10,  # 10%
    "low_tokens_per_second": 10.0,
    "very_low_tokens_per_second": 5.0
}

@dataclass
class OperationMetrics:
    """Data class for operation metrics"""
    operation_id: str
    operation_type: str
    model_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    token_count: Optional[int] = None
    tokens_per_second: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def finalize(self, end_time: Optional[float] = None):
        """Finalize metrics calculation"""
        self.end_time = end_time or time.time()
        self.duration = self.end_time - self.start_time
        if self.token_count and self.duration > 0:
            self.tokens_per_second = self.token_count / self.duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return asdict(self)

# Import Opik with graceful fallback
try:
    from opik import configure, track, opik_context
    OPIK_AVAILABLE = True
    logger.info("Opik successfully imported")
except ImportError as e:
    logger.warning(f"Opik not available: {e}")
    OPIK_AVAILABLE = False
    
    # Create fallback decorators that do nothing
    def track(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    
    class MockOpikContext:
        def update_current_trace(self, **kwargs):
            pass
        
        def start_span(self, name):  # pylint: disable=unused-argument
            return self
        
        def set_output(self, data):  # pylint: disable=unused-argument
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    opik_context = MockOpikContext()


class OpikIntegration:
    """Enhanced main class for managing comprehensive Opik integration"""
    
    def __init__(self):
        self.enabled = False
        self.project_name = "ollama-workbench"
        self.api_key = None
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        
        # Performance monitoring
        self._metrics_buffer = deque(maxlen=10000)  # Keep last 10k operations
        self._error_buffer = deque(maxlen=1000)     # Keep last 1k errors
        self._active_operations = {}  # Track ongoing operations
        self._metrics_lock = threading.Lock()
        
        # Alerting system
        self._alert_callbacks = []
        self._alert_cooldown = {}  # Prevent alert spam
        self._last_health_check = 0
        
        # Batch processing for performance
        self._batch_buffer = []
        self._batch_size = 100
        self._batch_timeout = 30  # seconds
        self._last_batch_send = time.time()
        
        self._initialize()
    
    def _initialize(self):
        """Initialize comprehensive Opik configuration"""
        if not OPIK_AVAILABLE:
            logger.warning("Opik integration disabled - package not available")
            return
        
        # Check for API key in environment or config
        self.api_key = os.getenv('OPIK_API_KEY')
        opik_url = os.getenv('OPIK_URL')
        workspace = os.getenv('OPIK_WORKSPACE')
        
        # Configure Opik if credentials are available
        try:
            if self.api_key:
                configure_kwargs = {
                    'api_key': self.api_key,
                    'project_name': self.project_name
                }
                
                if opik_url:
                    configure_kwargs['url'] = opik_url
                if workspace:
                    configure_kwargs['workspace'] = workspace
                    
                configure(**configure_kwargs)
                logger.info(f"Opik configured for project: {self.project_name}")
            else:
                logger.info("Opik available but no API key provided - will configure on first use")
                
            self.enabled = True
            
            # Start background monitoring
            self._start_background_monitoring()
            
        except Exception as e:
            logger.error(f"Failed to initialize Opik: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if Opik integration is enabled"""
        return self.enabled and OPIK_AVAILABLE
    
    def configure_project(self, project_name: str, api_key: Optional[str] = None):
        """Reconfigure Opik with new settings"""
        self.project_name = project_name
        if api_key:
            self.api_key = api_key
            os.environ['OPIK_API_KEY'] = api_key
        
        self._initialize()
    
    def create_trace_decorator(self, 
                             name: str, 
                             capture_input: bool = True, 
                             capture_output: bool = True,
                             tags: Optional[List[str]] = None):
        """Create a trace decorator for functions"""
        def decorator(func):
            if not self.is_enabled():
                return func
            
            @track(
                name=name,
                capture_input=capture_input,
                capture_output=capture_output
            )
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if tags:
                    opik_context.update_current_trace(tags=tags)
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def add_metadata(self, metadata: Dict[str, Any]):
        """Add metadata to current trace"""
        if self.is_enabled():
            opik_context.update_current_trace(metadata=metadata)
    
    def add_tags(self, tags: List[str]):
        """Add tags to current trace"""
        if self.is_enabled():
            opik_context.update_current_trace(tags=tags)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log error with context to current trace"""
        if self.is_enabled():
            error_metadata = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.now().isoformat()
            }
            if context:
                error_metadata.update(context)
            
            opik_context.update_current_trace(
                tags=["error"],
                metadata=error_metadata
            )
    
    def start_span(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Start a new span for tracking sub-operations"""
        if not self.is_enabled():
            return MockOpikContext()
        
        span = opik_context.start_span(name=name)
        if metadata:
            span.update_current_trace(metadata=metadata)
        return span
    
    def log_performance_metrics(self, 
                               response_time: float,
                               token_count: Optional[int] = None,
                               model_name: Optional[str] = None,
                               operation_type: str = "unknown",
                               operation_id: Optional[str] = None,
                               **kwargs):
        """Log comprehensive performance metrics to current trace and monitoring system"""
        metrics = {
            "response_time_seconds": response_time,
            "operation_type": operation_type,
            "timestamp": datetime.now().isoformat(),
            "operation_id": operation_id or f"{operation_type}_{int(time.time() * 1000)}"
        }
        
        if token_count:
            metrics["token_count"] = token_count
            metrics["tokens_per_second"] = token_count / response_time if response_time > 0 else 0
        
        if model_name:
            metrics["model_name"] = model_name
        
        # Add any additional metrics
        metrics.update(kwargs)
        
        # Store in metrics buffer for analysis
        with self._metrics_lock:
            operation_metrics = OperationMetrics(
                operation_id=metrics["operation_id"],
                operation_type=operation_type,
                model_name=model_name or "unknown",
                start_time=time.time() - response_time,
                end_time=time.time(),
                duration=response_time,
                token_count=token_count,
                tokens_per_second=metrics.get("tokens_per_second"),
                metadata=kwargs
            )
            operation_metrics.finalize()
            self._metrics_buffer.append(operation_metrics)
        
        # Check for performance alerts
        self._check_performance_alerts(metrics)
        
        # Log to Opik if enabled
        if self.is_enabled():
            opik_context.update_current_trace(
                metadata={"performance_metrics": metrics}
            )
        
        # Add to batch for efficient processing
        self._add_to_batch("performance_metrics", metrics)
    
    def _start_background_monitoring(self):
        """Start background thread for continuous monitoring"""
        def monitor():
            while self.enabled:
                try:
                    self._process_batch_metrics()
                    self._run_health_checks()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(60)  # Back off on error
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("Background monitoring started")
    
    def _process_batch_metrics(self):
        """Process batched metrics efficiently"""
        current_time = time.time()
        
        if (len(self._batch_buffer) >= self._batch_size or 
            current_time - self._last_batch_send > self._batch_timeout):
            
            if self._batch_buffer:
                try:
                    # Process batch - could send to external analytics, etc.
                    batch_size = len(self._batch_buffer)
                    logger.debug("Processing batch of %d metrics", batch_size)
                    
                    # Clear batch
                    self._batch_buffer.clear()
                    self._last_batch_send = current_time
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
    
    def _add_to_batch(self, metric_type: str, data: Dict[str, Any]):
        """Add metric to batch buffer"""
        self._batch_buffer.append({
            "type": metric_type,
            "data": data,
            "timestamp": time.time()
        })
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and trigger alerts"""
        response_time = metrics.get("response_time_seconds", 0)
        tokens_per_second = metrics.get("tokens_per_second", 0)
        model_name = metrics.get("model_name", "unknown")
        
        alerts = []
        
        # Check response time thresholds
        if response_time > self.thresholds["very_slow_response_time"]:
            alerts.append({
                "level": "critical",
                "type": "slow_response",
                "message": f"Very slow response: {response_time:.2f}s",
                "model": model_name,
                "value": response_time,
                "threshold": self.thresholds["very_slow_response_time"]
            })
        elif response_time > self.thresholds["slow_response_time"]:
            alerts.append({
                "level": "warning",
                "type": "slow_response",
                "message": f"Slow response: {response_time:.2f}s",
                "model": model_name,
                "value": response_time,
                "threshold": self.thresholds["slow_response_time"]
            })
        
        # Check throughput thresholds
        if tokens_per_second > 0:
            if tokens_per_second < self.thresholds["very_low_tokens_per_second"]:
                alerts.append({
                    "level": "critical",
                    "type": "low_throughput",
                    "message": f"Very low throughput: {tokens_per_second:.2f} tokens/sec",
                    "model": model_name,
                    "value": tokens_per_second,
                    "threshold": self.thresholds["very_low_tokens_per_second"]
                })
            elif tokens_per_second < self.thresholds["low_tokens_per_second"]:
                alerts.append({
                    "level": "warning",
                    "type": "low_throughput",
                    "message": f"Low throughput: {tokens_per_second:.2f} tokens/sec",
                    "model": model_name,
                    "value": tokens_per_second,
                    "threshold": self.thresholds["low_tokens_per_second"]
                })
        
        # Trigger alerts with cooldown
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert with cooldown to prevent spam"""
        alert_key = f"{alert['type']}_{alert['model']}_{alert['level']}"
        current_time = time.time()
        cooldown_period = 300 if alert['level'] == 'warning' else 180  # 5min for warning, 3min for critical
        
        if (alert_key not in self._alert_cooldown or 
            current_time - self._alert_cooldown[alert_key] > cooldown_period):
            
            self._alert_cooldown[alert_key] = current_time
            
            # Log alert
            logger.warning(f"Performance Alert: {alert['message']}", extra=alert)
            
            # Call registered alert callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for alerts"""
        self._alert_callbacks.append(callback)
    
    def _run_health_checks(self):
        """Run periodic health checks"""
        current_time = time.time()
        
        # Run health check every 5 minutes
        if current_time - self._last_health_check < 300:
            return
        
        self._last_health_check = current_time
        
        try:
            # Calculate error rates and performance metrics
            with self._metrics_lock:
                recent_metrics = [m for m in self._metrics_buffer 
                                if current_time - m.start_time < 3600]  # Last hour
                
                if not recent_metrics:
                    return
                
                total_ops = len(recent_metrics)
                failed_ops = len([m for m in recent_metrics if not m.success])
                error_rate = failed_ops / total_ops if total_ops > 0 else 0
                
                avg_response_time = sum(m.duration for m in recent_metrics if m.duration) / total_ops
                avg_tokens_per_second = sum(m.tokens_per_second for m in recent_metrics 
                                          if m.tokens_per_second) / len([m for m in recent_metrics 
                                                                        if m.tokens_per_second])
                
                health_metrics = {
                    "timestamp": current_time,
                    "total_operations_1h": total_ops,
                    "error_rate_1h": error_rate,
                    "avg_response_time_1h": avg_response_time,
                    "avg_tokens_per_second_1h": avg_tokens_per_second,
                    "failed_operations_1h": failed_ops
                }
                
                logger.info(f"Health check completed", extra=health_metrics)
                
                # Check for health alerts
                if error_rate > self.thresholds["very_high_error_rate"]:
                    self._trigger_alert({
                        "level": "critical",
                        "type": "high_error_rate",
                        "message": f"Very high error rate: {error_rate:.1%}",
                        "model": "system",
                        "value": error_rate,
                        "threshold": self.thresholds["very_high_error_rate"]
                    })
                elif error_rate > self.thresholds["high_error_rate"]:
                    self._trigger_alert({
                        "level": "warning",
                        "type": "high_error_rate",
                        "message": f"High error rate: {error_rate:.1%}",
                        "model": "system",
                        "value": error_rate,
                        "threshold": self.thresholds["high_error_rate"]
                    })
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def get_metrics_summary(self, time_range_hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        current_time = time.time()
        cutoff_time = current_time - (time_range_hours * 3600)
        
        with self._metrics_lock:
            recent_metrics = [m for m in self._metrics_buffer if m.start_time > cutoff_time]
            
            if not recent_metrics:
                return {"message": "No metrics available for the specified time range"}
            
            total_ops = len(recent_metrics)
            failed_ops = len([m for m in recent_metrics if not m.success])
            successful_ops = total_ops - failed_ops
            
            # Calculate averages for successful operations
            successful_metrics = [m for m in recent_metrics if m.success and m.duration]
            
            summary = {
                "time_range_hours": time_range_hours,
                "total_operations": total_ops,
                "successful_operations": successful_ops,
                "failed_operations": failed_ops,
                "success_rate": successful_ops / total_ops if total_ops > 0 else 0,
                "error_rate": failed_ops / total_ops if total_ops > 0 else 0
            }
            
            if successful_metrics:
                summary.update({
                    "avg_response_time": sum(m.duration for m in successful_metrics) / len(successful_metrics),
                    "min_response_time": min(m.duration for m in successful_metrics),
                    "max_response_time": max(m.duration for m in successful_metrics),
                    "median_response_time": sorted([m.duration for m in successful_metrics])[len(successful_metrics) // 2]
                })
                
                # Token metrics
                token_metrics = [m for m in successful_metrics if m.tokens_per_second]
                if token_metrics:
                    summary.update({
                        "avg_tokens_per_second": sum(m.tokens_per_second for m in token_metrics) / len(token_metrics),
                        "min_tokens_per_second": min(m.tokens_per_second for m in token_metrics),
                        "max_tokens_per_second": max(m.tokens_per_second for m in token_metrics),
                        "total_tokens_generated": sum(m.token_count for m in token_metrics if m.token_count)
                    })
                
                # Model breakdown
                model_stats = defaultdict(list)
                for m in successful_metrics:
                    model_stats[m.model_name].append(m)
                
                summary["model_performance"] = {
                    model: {
                        "operations": len(metrics),
                        "avg_response_time": sum(m.duration for m in metrics) / len(metrics),
                        "avg_tokens_per_second": sum(m.tokens_per_second for m in metrics 
                                                    if m.tokens_per_second) / len([m for m in metrics 
                                                                                   if m.tokens_per_second]) if any(m.tokens_per_second for m in metrics) else 0
                    }
                    for model, metrics in model_stats.items()
                }
            
            return summary
    
    def export_metrics(self, filepath: str, time_range_hours: int = 24, format_type: str = "json"):
        """Export metrics to file"""
        current_time = time.time()
        cutoff_time = current_time - (time_range_hours * 3600)
        
        with self._metrics_lock:
            recent_metrics = [m.to_dict() for m in self._metrics_buffer if m.start_time > cutoff_time]
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_range_hours": time_range_hours,
            "total_metrics": len(recent_metrics),
            "metrics": recent_metrics,
            "summary": self.get_metrics_summary(time_range_hours)
        }
        
        try:
            if format_type.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Metrics exported to {filepath}", extra={
                "filepath": filepath,
                "format": format_type,
                "metrics_count": len(recent_metrics)
            })
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise


# Global instance
opik_integration = OpikIntegration()


# Convenience functions for easy usage
def trace_llm_call(name: str = "llm_call"):
    """Decorator for tracing LLM calls"""
    return opik_integration.create_trace_decorator(
        name=name,
        capture_input=True,
        capture_output=True,
        tags=["llm", "generation"]
    )


def trace_rag_operation(name: str = "rag_operation"):
    """Decorator for tracing RAG operations"""
    return opik_integration.create_trace_decorator(
        name=name,
        capture_input=True,
        capture_output=True,
        tags=["rag", "retrieval"]
    )


def trace_agent_operation(name: str = "agent_operation"):
    """Decorator for tracing agent operations"""
    return opik_integration.create_trace_decorator(
        name=name,
        capture_input=True,
        capture_output=True,
        tags=["agent", "workflow"]
    )


def add_trace_metadata(metadata: Dict[str, Any]):
    """Add metadata to current trace"""
    opik_integration.add_metadata(metadata)


def add_trace_tags(tags: List[str]):
    """Add tags to current trace"""
    opik_integration.add_tags(tags)


def log_trace_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log error to current trace"""
    opik_integration.log_error(error, context)


def start_trace_span(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Start a new trace span"""
    return opik_integration.start_span(name, metadata)


def log_performance_metrics(response_time: float, 
                           token_count: Optional[int] = None,
                           model_name: Optional[str] = None,
                           operation_type: str = "unknown",
                           **kwargs):
    """Log comprehensive performance metrics"""
    opik_integration.log_performance_metrics(
        response_time, token_count, model_name, operation_type, **kwargs
    )

def get_metrics_summary(time_range_hours: int = 1) -> Dict[str, Any]:
    """Get comprehensive metrics summary"""
    return opik_integration.get_metrics_summary(time_range_hours)

def export_metrics(filepath: str, time_range_hours: int = 24, format_type: str = "json"):
    """Export metrics to file"""
    opik_integration.export_metrics(filepath, time_range_hours, format_type)

def register_alert_callback(callback: Callable[[Dict[str, Any]], None]):
    """Register callback for performance alerts"""
    opik_integration.register_alert_callback(callback)

def set_performance_thresholds(thresholds: Dict[str, float]):
    """Update performance monitoring thresholds"""
    opik_integration.thresholds.update(thresholds)
    logger.info(f"Performance thresholds updated", extra=thresholds)

@contextmanager
def trace_operation(name: str, operation_type: str = "custom", **metadata):
    """Context manager for tracing operations with automatic cleanup"""
    operation_id = f"{operation_type}_{int(time.time() * 1000)}"
    start_time = time.time()
    
    # Add to active operations
    opik_integration._active_operations[operation_id] = {
        "name": name,
        "operation_type": operation_type,
        "start_time": start_time,
        "metadata": metadata
    }
    
    span = start_trace_span(name, {"operation_id": operation_id, **metadata})
    
    try:
        yield operation_id
    except Exception as e:
        # Log error
        log_trace_error(e, {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "name": name
        })
        
        # Store error metrics
        with opik_integration._metrics_lock:
            error_metrics = OperationMetrics(
                operation_id=operation_id,
                operation_type=operation_type,
                model_name=metadata.get("model_name", "unknown"),
                start_time=start_time,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                metadata=metadata
            )
            error_metrics.finalize()
            opik_integration._metrics_buffer.append(error_metrics)
            opik_integration._error_buffer.append(error_metrics)
        
        raise
    finally:
        # Clean up active operations
        opik_integration._active_operations.pop(operation_id, None)
        
        # Finalize span
        if hasattr(span, '__exit__'):
            span.__exit__(None, None, None)


def configure_opik(project_name: str, api_key: Optional[str] = None):
    """Configure Opik integration"""
    opik_integration.configure_project(project_name, api_key)


def is_opik_enabled() -> bool:
    """Check if Opik is enabled"""
    return opik_integration.is_enabled()


# Export key functions and classes
__all__ = [
    'OpikIntegration',
    'opik_integration',
    'trace_llm_call',
    'trace_rag_operation', 
    'trace_agent_operation',
    'add_trace_metadata',
    'add_trace_tags',
    'log_trace_error',
    'start_trace_span',
    'log_performance_metrics',
    'configure_opik',
    'is_opik_enabled',
    'get_metrics_summary',
    'export_metrics',
    'register_alert_callback',
    'set_performance_thresholds',
    'trace_operation',
    'OperationMetrics',
    'DEFAULT_THRESHOLDS',
    'OPIK_AVAILABLE'
]
