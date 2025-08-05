# security/__init__.py

"""
Enhanced Security Framework for Ollama Workbench
Implements zero-trust security, encryption, authentication, and compliance features.
"""

from .authentication import *
from .encryption import *
from .access_control import *
from .audit_logging import *
from .security_config import *

__all__ = [
    # Authentication
    'AuthenticationManager',
    'authenticate_user',
    'require_authentication',
    'create_jwt_token',
    'verify_jwt_token',
    
    # Encryption
    'EncryptionManager',
    'encrypt_data',
    'decrypt_data',
    'generate_encryption_key',
    'secure_hash',
    
    # Access Control
    'AccessControlManager',
    'Role',
    'Permission',
    'require_permission',
    'check_permission',
    
    # Audit Logging
    'AuditLogger',
    'log_security_event',
    'log_access_attempt',
    'log_data_access',
    
    # Security Configuration
    'SecurityConfig',
    'get_security_config',
    'validate_security_settings'
]