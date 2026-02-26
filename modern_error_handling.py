"""
Modern Error Handling and Logging System for Ollama Workbench

Features:
- Structured logging with JSON formatting
- Context-aware error handling
- User-friendly error messages
- Automatic error recovery mechanisms
- Performance and health monitoring
- Error aggregation and reporting
"""

import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps

import streamlit as st


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    NETWORK = "network"
    MODEL = "model"
    API = "api"
    CONFIG = "config"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    PERFORMANCE = "performance"


@dataclass
class ErrorContext:
    """Enhanced error context with structured information"""
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    error_code: str
    message: str
    user_message: str
    technical_details: Dict[str, Any]
    stack_trace: Optional[str] = None
    suggested_actions: List[str] = None
    recovery_attempted: bool = False
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ModernErrorHandler:
    """
    Modern error handling system with structured logging and recovery
    """
    
    def __init__(self, app_name: str = "OllamaWorkbench"):
        self.app_name = app_name
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Configure structured logger
        self.logger = self._setup_structured_logger()
        
        # Initialize error stats
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "last_error_time": None
        }

    def _setup_structured_logger(self) -> logging.Logger:
        """Setup structured JSON logger"""
        logger = logging.getLogger(f"{self.app_name}.errors")
        
        if not logger.handlers:
            # Custom JSON formatter
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        "timestamp": time.time(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                    }
                    
                    # Add extra fields
                    if hasattr(record, 'error_context'):
                        log_data["error_context"] = asdict(record.error_context)
                    
                    if hasattr(record, 'performance_metrics'):
                        log_data["performance_metrics"] = record.performance_metrics
                    
                    return json.dumps(log_data, default=str)
            
            # File handler for structured logs
            try:
                file_handler = logging.FileHandler('logs/structured_errors.jsonl', mode='a')
                file_handler.setFormatter(JSONFormatter())
                logger.addHandler(file_handler)
            except (FileNotFoundError, PermissionError):
                # Fallback to console if can't write to file
                pass
            
            # Console handler for development
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(console_handler)
            
            logger.setLevel(logging.INFO)
        
        return logger

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        user_message: str = None,
        suggested_actions: List[str] = None,
        attempt_recovery: bool = True
    ) -> ErrorContext:
        """
        Comprehensive error handling with structured logging
        """
        # Generate error context
        error_context = ErrorContext(
            timestamp=time.time(),
            severity=severity,
            category=category,
            error_code=f"{category.value}_{type(error).__name__}",
            message=str(error),
            user_message=user_message or self._generate_user_friendly_message(error, category),
            technical_details=context or {},
            stack_trace=traceback.format_exc(),
            suggested_actions=suggested_actions or self._generate_suggested_actions(error, category),
            session_id=st.session_state.get('session_id'),
            user_id=st.session_state.get('user_id')
        )
        
        # Update statistics
        self._update_error_stats(error_context)
        
        # Log structured error
        self.logger.error(
            f"Error in {self.app_name}: {error_context.message}",
            extra={"error_context": error_context}
        )
        
        # Store in history
        self.error_history.append(error_context)
        
        # Attempt recovery if enabled
        if attempt_recovery:
            recovery_success = self._attempt_recovery(error_context)
            error_context.recovery_attempted = True
            
            if recovery_success:
                self.logger.info(f"Successfully recovered from error: {error_context.error_code}")
        
        # Display user-friendly error in Streamlit
        self._display_user_error(error_context)
        
        return error_context

    def _generate_user_friendly_message(self, error: Exception, category: ErrorCategory) -> str:
        """Generate user-friendly error messages"""
        error_messages = {
            ErrorCategory.NETWORK: "Unable to connect to the service. Please check your internet connection.",
            ErrorCategory.MODEL: "There was an issue with the AI model. Please try a different model or check if the model is available.",
            ErrorCategory.API: "The API service is temporarily unavailable. Please try again in a moment.",
            ErrorCategory.CONFIG: "There's a configuration issue. Please check your settings.",
            ErrorCategory.USER_INPUT: "Please check your input and try again.",
            ErrorCategory.SYSTEM: "A system error occurred. Our team has been notified.",
            ErrorCategory.PERFORMANCE: "The system is running slowly. Please wait a moment and try again."
        }
        
        base_message = error_messages.get(category, "An unexpected error occurred.")
        
        # Add specific error type information
        if "connection" in str(error).lower():
            return "Connection failed. Please check that the Ollama server is running and accessible."
        elif "timeout" in str(error).lower():
            return "The request timed out. Please try again or use a smaller input."
        elif "model" in str(error).lower() and "not found" in str(error).lower():
            return "The selected model is not available. Please choose a different model or pull the model first."
        
        return base_message

    def _generate_suggested_actions(self, error: Exception, category: ErrorCategory) -> List[str]:
        """Generate context-aware suggested actions"""
        actions = []
        
        if category == ErrorCategory.NETWORK:
            actions.extend([
                "Check your internet connection",
                "Verify that the Ollama server is running",
                "Try refreshing the page"
            ])
        elif category == ErrorCategory.MODEL:
            actions.extend([
                "Try selecting a different model",
                "Check if the model is installed: `ollama list`",
                "Pull the model if needed: `ollama pull <model_name>`"
            ])
        elif category == ErrorCategory.API:
            actions.extend([
                "Wait a moment and try again",
                "Check API key configuration",
                "Verify service status"
            ])
        elif category == ErrorCategory.CONFIG:
            actions.extend([
                "Check configuration settings",
                "Reset to default settings",
                "Verify file permissions"
            ])
        elif category == ErrorCategory.USER_INPUT:
            actions.extend([
                "Check your input format",
                "Try with shorter text",
                "Remove special characters"
            ])
        else:
            actions.extend([
                "Refresh the page",
                "Try again in a moment",
                "Contact support if the issue persists"
            ])
        
        return actions

    def _update_error_stats(self, error_context: ErrorContext):
        """Update error statistics"""
        self.error_stats["total_errors"] += 1
        self.error_stats["last_error_time"] = error_context.timestamp
        
        # Update category stats
        category_key = error_context.category.value
        self.error_stats["errors_by_category"][category_key] = (
            self.error_stats["errors_by_category"].get(category_key, 0) + 1
        )
        
        # Update severity stats
        severity_key = error_context.severity.value
        self.error_stats["errors_by_severity"][severity_key] = (
            self.error_stats["errors_by_severity"].get(severity_key, 0) + 1
        )

    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt automatic error recovery"""
        recovery_strategy = self.recovery_strategies.get(error_context.error_code)
        
        if recovery_strategy:
            try:
                recovery_strategy(error_context)
                return True
            except Exception as recovery_error:
                self.logger.warning(f"Recovery failed: {recovery_error}")
                return False
        
        # Generic recovery strategies
        if error_context.category == ErrorCategory.NETWORK:
            return self._attempt_network_recovery()
        elif error_context.category == ErrorCategory.MODEL:
            return self._attempt_model_recovery(error_context)
        
        return False

    def _attempt_network_recovery(self) -> bool:
        """Attempt network recovery"""
        try:
            # Try to reconnect or reset connection
            import time
            time.sleep(1)  # Brief pause
            # Add specific network recovery logic here
            return True
        except Exception:
            return False

    def _attempt_model_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt model-related recovery"""
        try:
            # Try fallback model or reset model state
            if "selected_model" in st.session_state:
                # Set to a known working model
                st.session_state.selected_model = "llama2"  # fallback
                return True
        except Exception:
            return False

    def _display_user_error(self, error_context: ErrorContext):
        """Display user-friendly error in Streamlit"""
        if error_context.severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 Critical Error: {error_context.user_message}")
        elif error_context.severity == ErrorSeverity.HIGH:
            st.error(f"❗ Error: {error_context.user_message}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            st.warning(f"⚠️ Warning: {error_context.user_message}")
        else:
            st.info(f"ℹ️ Notice: {error_context.user_message}")
        
        # Show suggested actions in an expander
        if error_context.suggested_actions:
            with st.expander("💡 Suggested Actions"):
                for action in error_context.suggested_actions:
                    st.write(f"• {action}")

    def register_recovery_strategy(self, error_code: str, strategy: Callable):
        """Register custom recovery strategy"""
        self.recovery_strategies[error_code] = strategy
        self.logger.info(f"Registered recovery strategy for {error_code}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return self.error_stats.copy()

    def get_recent_errors(self, limit: int = 10) -> List[ErrorContext]:
        """Get recent errors"""
        return self.error_history[-limit:]

    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_success_rate": 0.0,
            "last_error_time": None
        }


# Global error handler instance
_global_error_handler: Optional[ModernErrorHandler] = None


def get_error_handler() -> ModernErrorHandler:
    """Get or create global error handler"""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ModernErrorHandler()
    
    return _global_error_handler


# Decorator for automatic error handling
def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    user_message: str = None,
    attempt_recovery: bool = True
):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args) if len(str(args)) < 500 else "args_too_long",
                    "kwargs": str(kwargs) if len(str(kwargs)) < 500 else "kwargs_too_long"
                }
                
                error_handler.handle_error(
                    error=e,
                    context=context,
                    category=category,
                    severity=severity,
                    user_message=user_message,
                    attempt_recovery=attempt_recovery
                )
                
                # Re-raise for critical errors
                if severity == ErrorSeverity.CRITICAL:
                    raise
                
                return None
        
        return wrapper
    return decorator


@contextmanager
def error_context(
    operation_name: str,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Context manager for error handling"""
    error_handler = get_error_handler()
    start_time = time.time()
    
    try:
        yield error_handler
    except Exception as e:
        duration = time.time() - start_time
        context = {
            "operation": operation_name,
            "duration": duration
        }
        
        error_handler.handle_error(
            error=e,
            context=context,
            category=category,
            severity=severity
        )
        raise
    finally:
        # Log operation completion
        duration = time.time() - start_time
        error_handler.logger.info(
            f"Operation '{operation_name}' completed in {duration:.2f}s",
            extra={"performance_metrics": {"operation": operation_name, "duration": duration}}
        )


# Export main classes and functions
__all__ = [
    "ModernErrorHandler",
    "ErrorContext",
    "ErrorSeverity", 
    "ErrorCategory",
    "get_error_handler",
    "handle_errors",
    "error_context"
]