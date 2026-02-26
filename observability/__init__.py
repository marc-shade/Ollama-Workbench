# observability/__init__.py

"""
Ollama Workbench Observability Package

This package provides comprehensive observability and monitoring capabilities
for the Ollama Workbench, including LLM tracing, performance monitoring,
and analytics integration.
"""

from .opik_integration import (
    opik_integration,
    trace_llm_call,
    trace_rag_operation,
    trace_agent_operation,
    add_trace_metadata,
    add_trace_tags,
    log_trace_error,
    start_trace_span,
    log_performance_metrics,
    configure_opik,
    is_opik_enabled,
    OPIK_AVAILABLE
)

from .config import (
    observability_config,
    get_opik_project_name,
    get_opik_api_key,
    is_local_mode,
    update_opik_settings,
    enable_observability,
    configure_privacy_settings
)

from .dashboard import enhanced_observability_dashboard

__version__ = "1.0.0"
__author__ = "2 Acre Studios"

__all__ = [
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
    'OPIK_AVAILABLE',
    'observability_config',
    'get_opik_project_name',
    'get_opik_api_key',
    'is_local_mode',
    'update_opik_settings',
    'enable_observability',
    'configure_privacy_settings',
    'enhanced_observability_dashboard'
]
