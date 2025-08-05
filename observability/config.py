# observability/config.py

"""
Configuration settings for Ollama Workbench observability features.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_DIR = Path("data")
CONFIG_FILE = CONFIG_DIR / "observability_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "opik": {
        "enabled": True,
        "project_name": "ollama-workbench",
        "api_key": None,
        "workspace": None,
        "capture_input": True,
        "capture_output": True,
        "capture_errors": True,
        "local_mode": True,
        "batch_size": 100,
        "flush_interval": 60
    },
    "privacy": {
        "hash_prompts": False,
        "truncate_responses": False,
        "max_response_length": 1000,
        "exclude_patterns": []
    },
    "performance": {
        "enable_detailed_metrics": True,
        "track_token_usage": True,
        "track_latency": True,
        "track_resource_usage": False
    },
    "alerts": {
        "enabled": True,
        "error_threshold": 0.05,
        "latency_threshold": 10.0,
        "token_usage_threshold": 10000
    },
    "retention": {
        "trace_retention_days": 30,
        "metrics_retention_days": 90,
        "log_retention_days": 7
    }
}


class ObservabilityConfig:
    """Manages observability configuration settings"""
    
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self._load_config()
        self._apply_environment_overrides()
    
    def _load_config(self):
        """Load configuration from file"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    stored_config = json.load(f)
                    self._merge_config(stored_config)
                logger.info(f"Loaded observability config from {CONFIG_FILE}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Recursively merge new config with defaults"""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
            return base
        
        merge_dict(self.config, new_config)
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # Opik settings
        if os.getenv('OPIK_API_KEY'):
            self.config['opik']['api_key'] = os.getenv('OPIK_API_KEY')
            self.config['opik']['local_mode'] = False
        
        if os.getenv('OPIK_PROJECT_NAME'):
            self.config['opik']['project_name'] = os.getenv('OPIK_PROJECT_NAME')
        
        if os.getenv('OPIK_WORKSPACE'):
            self.config['opik']['workspace'] = os.getenv('OPIK_WORKSPACE')
        
        # Privacy settings
        if os.getenv('OBSERVABILITY_HASH_PROMPTS'):
            self.config['privacy']['hash_prompts'] = os.getenv('OBSERVABILITY_HASH_PROMPTS').lower() == 'true'
        
        # Enable/disable observability entirely
        if os.getenv('OBSERVABILITY_ENABLED'):
            self.config['opik']['enabled'] = os.getenv('OBSERVABILITY_ENABLED').lower() == 'true'
    
    def save_config(self):
        """Save current configuration to file"""
        CONFIG_DIR.mkdir(exist_ok=True)
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved observability config to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'opik.project_name')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def is_opik_enabled(self) -> bool:
        """Check if Opik integration is enabled"""
        return self.get('opik.enabled', True)
    
    def should_capture_input(self) -> bool:
        """Check if input capture is enabled"""
        return self.get('opik.capture_input', True)
    
    def should_capture_output(self) -> bool:
        """Check if output capture is enabled"""
        return self.get('opik.capture_output', True)
    
    def should_hash_prompts(self) -> bool:
        """Check if prompts should be hashed for privacy"""
        return self.get('privacy.hash_prompts', False)
    
    def get_opik_config(self) -> Dict[str, Any]:
        """Get Opik-specific configuration"""
        return self.get('opik', {})
    
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy-specific configuration"""
        return self.get('privacy', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration"""
        return self.get('performance', {})
    
    def get_alerts_config(self) -> Dict[str, Any]:
        """Get alerts configuration"""
        return self.get('alerts', {})
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._merge_config(updates)
        self.save_config()


# Global configuration instance
observability_config = ObservabilityConfig()


# Convenience functions
def get_opik_project_name() -> str:
    """Get the Opik project name"""
    return observability_config.get('opik.project_name', 'ollama-workbench')


def get_opik_api_key() -> Optional[str]:
    """Get the Opik API key"""
    return observability_config.get('opik.api_key')


def is_local_mode() -> bool:
    """Check if running in local mode"""
    return observability_config.get('opik.local_mode', True)


def should_capture_detailed_metrics() -> bool:
    """Check if detailed metrics should be captured"""
    return observability_config.get('performance.enable_detailed_metrics', True)


def get_trace_retention_days() -> int:
    """Get trace retention period in days"""
    return observability_config.get('retention.trace_retention_days', 30)


def update_opik_settings(project_name: str = None, api_key: str = None, workspace: str = None):
    """Update Opik settings"""
    updates = {}
    if project_name:
        updates['opik.project_name'] = project_name
    if api_key:
        updates['opik.api_key'] = api_key
        updates['opik.local_mode'] = False
    if workspace:
        updates['opik.workspace'] = workspace
    
    for key, value in updates.items():
        observability_config.set(key, value)
    
    observability_config.save_config()


def enable_observability(enabled: bool = True):
    """Enable or disable observability"""
    observability_config.set('opik.enabled', enabled)
    observability_config.save_config()


def configure_privacy_settings(hash_prompts: bool = None, 
                               truncate_responses: bool = None,
                               max_response_length: int = None):
    """Configure privacy settings"""
    updates = {}
    if hash_prompts is not None:
        updates['privacy.hash_prompts'] = hash_prompts
    if truncate_responses is not None:
        updates['privacy.truncate_responses'] = truncate_responses
    if max_response_length is not None:
        updates['privacy.max_response_length'] = max_response_length
    
    for key, value in updates.items():
        observability_config.set(key, value)
    
    observability_config.save_config()


# Export main objects
__all__ = [
    'ObservabilityConfig',
    'observability_config',
    'get_opik_project_name',
    'get_opik_api_key',
    'is_local_mode',
    'should_capture_detailed_metrics',
    'get_trace_retention_days',
    'update_opik_settings',
    'enable_observability',
    'configure_privacy_settings'
]
