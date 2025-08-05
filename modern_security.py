"""
Modern Security Implementation for Ollama Workbench

Features:
- Secure credential storage with encryption
- Input validation and sanitization
- Rate limiting and abuse prevention
- Session management with security headers
- API key management with rotation
- Audit logging for security events
- Content filtering and PII detection
- Secure configuration management
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import logging

import streamlit as st


class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    timestamp: float
    event_type: str
    severity: SecurityLevel
    user_id: Optional[str]
    session_id: Optional[str]
    description: str
    metadata: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SecureCredentialManager:
    """
    Secure credential storage with encryption and key rotation
    """
    
    def __init__(self, config_dir: str = ".secure"):
        self.config_dir = config_dir
        self.credentials_file = os.path.join(config_dir, "credentials.enc")
        self.salt_file = os.path.join(config_dir, "salt.key")
        
        # Ensure config directory exists
        os.makedirs(config_dir, mode=0o700, exist_ok=True)
        
        # Initialize encryption
        self._init_encryption()
        
        # Load existing credentials
        self.credentials = self._load_credentials()

    def _init_encryption(self):
        """Initialize encryption key from master password or generate new one"""
        if os.path.exists(self.salt_file):
            with open(self.salt_file, 'rb') as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(self.salt_file, 'wb') as f:
                f.write(salt)
            os.chmod(self.salt_file, 0o600)
        
        # Use environment variable for master password or generate
        master_password = os.environ.get('OLLAMA_MASTER_KEY', 'default-key-change-in-production')
        
        # Derive encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        self.cipher = Fernet(key)

    def _load_credentials(self) -> Dict[str, Any]:
        """Load and decrypt credentials"""
        if not os.path.exists(self.credentials_file):
            return {}
        
        try:
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logging.error(f"Failed to load credentials: {e}")
            return {}

    def _save_credentials(self):
        """Encrypt and save credentials"""
        try:
            data = json.dumps(self.credentials).encode()
            encrypted_data = self.cipher.encrypt(data)
            
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            os.chmod(self.credentials_file, 0o600)
        except Exception as e:
            logging.error(f"Failed to save credentials: {e}")

    def store_credential(self, key: str, value: str, metadata: Optional[Dict] = None):
        """Store encrypted credential"""
        self.credentials[key] = {
            "value": value,
            "created_at": time.time(),
            "accessed_at": time.time(),
            "access_count": 0,
            "metadata": metadata or {}
        }
        self._save_credentials()

    def get_credential(self, key: str) -> Optional[str]:
        """Retrieve and decrypt credential"""
        if key in self.credentials:
            credential = self.credentials[key]
            credential["accessed_at"] = time.time()
            credential["access_count"] += 1
            self._save_credentials()
            return credential["value"]
        return None

    def delete_credential(self, key: str) -> bool:
        """Delete credential"""
        if key in self.credentials:
            del self.credentials[key]
            self._save_credentials()
            return True
        return False

    def list_credentials(self) -> List[str]:
        """List credential keys (not values)"""
        return list(self.credentials.keys())

    def rotate_key(self, new_master_password: str):
        """Rotate encryption key"""
        # Decrypt with old key
        old_credentials = self.credentials.copy()
        
        # Update master password and reinitialize
        os.environ['OLLAMA_MASTER_KEY'] = new_master_password
        self._init_encryption()
        
        # Re-encrypt with new key
        self.credentials = old_credentials
        self._save_credentials()


class InputValidator:
    """
    Input validation and sanitization
    """
    
    # Common patterns for validation
    PATTERNS = {
        "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        "api_key": re.compile(r'^[a-zA-Z0-9_-]{20,100}$'),
        "model_name": re.compile(r'^[a-zA-Z0-9._-]+$'),
        "safe_filename": re.compile(r'^[a-zA-Z0-9._-]+$'),
        "alphanumeric": re.compile(r'^[a-zA-Z0-9]+$'),
    }
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        re.compile(r'<script', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'eval\s*\(', re.IGNORECASE),
        re.compile(r'exec\s*\(', re.IGNORECASE),
        re.compile(r'import\s+os', re.IGNORECASE),
        re.compile(r'__import__', re.IGNORECASE),
    ]
    
    @classmethod
    def validate_input(cls, value: str, pattern_name: str) -> bool:
        """Validate input against pattern"""
        if pattern_name in cls.PATTERNS:
            return bool(cls.PATTERNS[pattern_name].match(value))
        return False
    
    @classmethod
    def sanitize_input(cls, value: str, max_length: int = 1000) -> str:
        """Sanitize input for safe usage"""
        if not isinstance(value, str):
            value = str(value)
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remove dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            value = pattern.sub('', value)
        
        # Basic HTML escaping
        value = value.replace('<', '&lt;').replace('>', '&gt;')
        value = value.replace('"', '&quot;').replace("'", '&#x27;')
        
        return value.strip()
    
    @classmethod
    def detect_pii(cls, text: str) -> List[str]:
        """Detect potential PII in text"""
        pii_patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }
        
        detected = []
        for pii_type, pattern in pii_patterns.items():
            if pattern.search(text):
                detected.append(pii_type)
        
        return detected


class RateLimiter:
    """
    Rate limiting to prevent abuse
    """
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.limits = {
            "api_calls": {"count": 100, "window": 3600},  # 100 per hour
            "login_attempts": {"count": 5, "window": 900},  # 5 per 15 minutes
            "model_requests": {"count": 50, "window": 3600},  # 50 per hour
        }

    def is_allowed(self, key: str, limit_type: str = "api_calls") -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        limit_config = self.limits.get(limit_type, self.limits["api_calls"])
        
        # Initialize key if not exists
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests outside window
        window_start = current_time - limit_config["window"]
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
        # Check if under limit
        if len(self.requests[key]) < limit_config["count"]:
            self.requests[key].append(current_time)
            return True
        
        return False

    def get_remaining(self, key: str, limit_type: str = "api_calls") -> int:
        """Get remaining requests for key"""
        current_time = time.time()
        limit_config = self.limits.get(limit_type, self.limits["api_calls"])
        
        if key not in self.requests:
            return limit_config["count"]
        
        # Clean old requests
        window_start = current_time - limit_config["window"]
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
        return max(0, limit_config["count"] - len(self.requests[key]))


class SecurityAuditor:
    """
    Security event logging and monitoring
    """
    
    def __init__(self, log_file: str = "logs/security.jsonl"):
        self.log_file = log_file
        self.events: List[SecurityEvent] = []
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        description: str,
        metadata: Optional[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log security event"""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            user_id=user_id or st.session_state.get('user_id'),
            session_id=session_id or st.session_state.get('session_id'),
            description=description,
            metadata=metadata or {},
        )
        
        self.events.append(event)
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(event), default=str) + '\n')
        except Exception as e:
            logging.error(f"Failed to write security log: {e}")

    def get_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[SecurityLevel] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get filtered security events"""
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        return filtered_events[-limit:]

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics"""
        if not self.events:
            return {"total_events": 0}
        
        recent_events = [e for e in self.events if time.time() - e.timestamp < 86400]  # Last 24h
        
        event_types = {}
        severity_counts = {}
        
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        return {
            "total_events": len(self.events),
            "recent_events_24h": len(recent_events),
            "event_types": event_types,
            "severity_distribution": severity_counts,
            "last_event_time": max(e.timestamp for e in self.events) if self.events else None
        }


class SessionSecurityManager:
    """
    Secure session management
    """
    
    def __init__(self):
        self.session_timeout = 3600  # 1 hour
        self.max_sessions_per_user = 5

    def initialize_session(self) -> str:
        """Initialize secure session"""
        session_id = secrets.token_urlsafe(32)
        
        # Store session info
        st.session_state.update({
            'session_id': session_id,
            'session_start': time.time(),
            'last_activity': time.time(),
            'security_level': SecurityLevel.MEDIUM.value,
            'csrf_token': secrets.token_urlsafe(16)
        })
        
        return session_id

    def validate_session(self) -> bool:
        """Validate current session"""
        if 'session_id' not in st.session_state:
            return False
        
        # Check timeout
        last_activity = st.session_state.get('last_activity', 0)
        if time.time() - last_activity > self.session_timeout:
            self.terminate_session()
            return False
        
        # Update activity
        st.session_state['last_activity'] = time.time()
        return True

    def terminate_session(self):
        """Terminate current session"""
        session_keys = [
            'session_id', 'session_start', 'last_activity',
            'security_level', 'csrf_token', 'user_id'
        ]
        
        for key in session_keys:
            st.session_state.pop(key, None)

    def get_csrf_token(self) -> Optional[str]:
        """Get CSRF token for forms"""
        return st.session_state.get('csrf_token')

    def validate_csrf_token(self, token: str) -> bool:
        """Validate CSRF token"""
        expected_token = st.session_state.get('csrf_token')
        return expected_token and hmac.compare_digest(expected_token, token)


# Global instances
_credential_manager = SecureCredentialManager()
_rate_limiter = RateLimiter()
_security_auditor = SecurityAuditor()
_session_manager = SessionSecurityManager()


def secure_api_key_input(label: str, key: str) -> Optional[str]:
    """Secure API key input with validation"""
    with st.form(key=f"api_key_form_{key}"):
        api_key = st.text_input(
            label,
            type="password",
            help="Your API key will be encrypted and stored securely"
        )
        
        csrf_token = st.text_input("", value=_session_manager.get_csrf_token(), type="password", label_visibility="hidden")
        submit = st.form_submit_button("Save API Key")
        
        if submit:
            # Validate CSRF token
            if not _session_manager.validate_csrf_token(csrf_token):
                st.error("Invalid security token. Please refresh the page.")
                return None
            
            # Validate API key format
            if not InputValidator.validate_input(api_key, "api_key"):
                st.error("Invalid API key format")
                return None
            
            # Store encrypted
            _credential_manager.store_credential(key, api_key)
            _security_auditor.log_event("api_key_stored", SecurityLevel.MEDIUM, f"API key stored for {key}")
            
            st.success("API key saved securely")
            return api_key
    
    # Try to load existing key
    return _credential_manager.get_credential(key)


def validate_user_input(text: str, max_length: int = 10000) -> Tuple[bool, str]:
    """Validate and sanitize user input"""
    # Check for PII
    pii_detected = InputValidator.detect_pii(text)
    if pii_detected:
        _security_auditor.log_event(
            "pii_detected",
            SecurityLevel.HIGH,
            f"PII detected in user input: {', '.join(pii_detected)}"
        )
        return False, f"Potential PII detected: {', '.join(pii_detected)}. Please remove sensitive information."
    
    # Sanitize input
    sanitized = InputValidator.sanitize_input(text, max_length)
    
    if len(sanitized) != len(text):
        _security_auditor.log_event(
            "input_sanitized",
            SecurityLevel.LOW,
            "User input was sanitized"
        )
    
    return True, sanitized


def check_rate_limit(key: str, limit_type: str = "api_calls") -> bool:
    """Check rate limit for user/session"""
    if not _rate_limiter.is_allowed(key, limit_type):
        _security_auditor.log_event(
            "rate_limit_exceeded",
            SecurityLevel.MEDIUM,
            f"Rate limit exceeded for {key} ({limit_type})"
        )
        return False
    return True


def get_security_dashboard() -> Dict[str, Any]:
    """Get security dashboard data"""
    return {
        "session_valid": _session_manager.validate_session(),
        "rate_limits": {
            "api_calls": _rate_limiter.get_remaining(st.session_state.get('session_id', 'anonymous'), "api_calls"),
            "model_requests": _rate_limiter.get_remaining(st.session_state.get('session_id', 'anonymous'), "model_requests")
        },
        "security_events": _security_auditor.get_security_summary(),
        "stored_credentials": len(_credential_manager.list_credentials())
    }


# Export main functions and classes
__all__ = [
    "SecureCredentialManager",
    "InputValidator",
    "RateLimiter", 
    "SecurityAuditor",
    "SessionSecurityManager",
    "SecurityLevel",
    "secure_api_key_input",
    "validate_user_input",
    "check_rate_limit",
    "get_security_dashboard"
]