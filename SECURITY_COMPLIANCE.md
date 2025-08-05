# Security and Compliance Documentation

## Document Information
- **Version**: 1.0
- **Date**: May 22, 2025
- **Project**: Ollama Workbench Security Framework
- **Classification**: Internal Use
- **Last Review**: Initial Creation

## Table of Contents
1. [Security Overview](#security-overview)
2. [Security Architecture](#security-architecture)
3. [Authentication & Authorization](#authentication--authorization)
4. [Data Protection](#data-protection)
5. [Infrastructure Security](#infrastructure-security)
6. [Application Security](#application-security)
7. [Compliance Framework](#compliance-framework)
8. [Security Monitoring](#security-monitoring)
9. [Incident Response](#incident-response)
10. [Security Policies](#security-policies)

---

## Security Overview

### Security Principles
1. **Zero Trust Architecture**: Never trust, always verify
2. **Defense in Depth**: Multiple layers of security controls
3. **Least Privilege**: Minimum necessary access rights
4. **Privacy by Design**: Data protection built into system design
5. **Continuous Monitoring**: Real-time security assessment
6. **Incident Response**: Rapid detection and response capabilities

### Threat Model
- **External Attackers**: Unauthorized access attempts, data breaches
- **Insider Threats**: Malicious or negligent internal users
- **Supply Chain Attacks**: Compromised dependencies or infrastructure
- **Data Exposure**: Accidental or intentional data leaks
- **Service Disruption**: DDoS attacks, system overload
- **Model Poisoning**: Malicious training data or model manipulation

### Security Objectives
- **Confidentiality**: Protect sensitive data and models
- **Integrity**: Ensure data and system reliability
- **Availability**: Maintain system uptime and performance
- **Accountability**: Complete audit trails and traceability
- **Privacy**: Protect user personal information
- **Compliance**: Meet regulatory and industry standards

---

## Security Architecture

### High-Level Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    External Layer                           │
│  ┌───────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   WAF/CDN     │  │ Rate Limiter│  │  DDoS Protection│   │
│  └───────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Network Security Layer                      │
│  ┌───────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  Load Balancer│  │   Firewall  │  │   TLS Gateway   │   │
│  └───────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│              Application Security Layer                     │
│  ┌───────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  API Gateway  │  │    Auth     │  │   Input Valid   │   │
│  │  (AuthN/AuthZ)│  │   Service   │  │   & Sanitization│   │
│  └───────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Data Security Layer                         │
│  ┌───────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  Encryption   │  │   Access    │  │   Data Loss     │   │
│  │  at Rest      │  │   Controls  │  │   Prevention    │   │
│  └───────────────┘  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Security Zones

#### DMZ (Demilitarized Zone)
- Web Application Firewall (WAF)
- Load balancers and reverse proxies
- Public-facing API endpoints
- Static content delivery

#### Application Zone
- Web application servers
- API services
- Pipeline execution engines
- Authentication services

#### Data Zone
- Database servers
- File storage systems
- Vector databases
- Backup systems

#### Management Zone
- Monitoring and logging systems
- Security tools and scanners
- Administrative interfaces
- Development tools

---

## Authentication & Authorization

### Multi-Factor Authentication (MFA)

#### Implementation Strategy
```python
# auth/mfa.py
import pyotp
import qrcode
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class MFAManager:
    """Multi-Factor Authentication management."""
    
    def __init__(self):
        self.issuer_name = "Ollama Workbench"
        self.totp_window = 1  # Allow 30-second window
    
    def generate_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user."""
        secret = pyotp.random_base32()
        
        # Store secret securely in database
        self._store_mfa_secret(user_id, secret)
        
        return secret
    
    def generate_qr_code(self, user_email: str, secret: str) -> bytes:
        """Generate QR code for TOTP setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.issuer_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        import io
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token."""
        secret = self._get_mfa_secret(user_id)
        if not secret:
            return False
        
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=self.totp_window)
    
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup recovery code."""
        backup_codes = self._get_backup_codes(user_id)
        
        if code in backup_codes:
            # Remove used code
            self._remove_backup_code(user_id, code)
            return True
        
        return False
    
    def generate_backup_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate backup recovery codes."""
        import secrets
        codes = [secrets.token_hex(4).upper() for _ in range(count)]
        
        # Store codes securely
        self._store_backup_codes(user_id, codes)
        
        return codes

# Secure session management
class SecureSession:
    """Secure session management with enhanced security."""
    
    def __init__(self):
        self.session_timeout = timedelta(hours=8)
        self.max_idle_time = timedelta(minutes=30)
        self.max_concurrent_sessions = 5
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        mfa_verified: bool = False
    ) -> str:
        """Create secure session with metadata."""
        
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "mfa_verified": mfa_verified,
            "permissions": self._get_user_permissions(user_id),
            "session_id": self._generate_secure_session_id()
        }
        
        # Enforce concurrent session limits
        self._enforce_session_limits(user_id)
        
        # Store session securely
        self._store_session(session_data)
        
        return session_data["session_id"]
    
    def validate_session(self, session_id: str, ip_address: str) -> Optional[Dict]:
        """Validate session with security checks."""
        session = self._get_session(session_id)
        
        if not session:
            return None
        
        # Check session expiry
        if self._is_session_expired(session):
            self._invalidate_session(session_id)
            return None
        
        # Check IP address consistency (optional)
        if session["ip_address"] != ip_address:
            self._log_security_event(
                "session_ip_mismatch",
                session_id=session_id,
                original_ip=session["ip_address"],
                current_ip=ip_address
            )
            # Could invalidate session or require re-authentication
        
        # Update last activity
        self._update_session_activity(session_id)
        
        return session
```

### Role-Based Access Control (RBAC)

#### Permission Matrix
```python
# auth/rbac.py
from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass

class Permission(Enum):
    # Conversation permissions
    CONVERSATIONS_READ = "conversations.read"
    CONVERSATIONS_WRITE = "conversations.write"
    CONVERSATIONS_DELETE = "conversations.delete"
    CONVERSATIONS_SHARE = "conversations.share"
    
    # Model permissions
    MODELS_READ = "models.read"
    MODELS_MANAGE = "models.manage"
    MODELS_CONFIGURE = "models.configure"
    
    # Pipeline permissions
    PIPELINES_READ = "pipelines.read"
    PIPELINES_WRITE = "pipelines.write"
    PIPELINES_EXECUTE = "pipelines.execute"
    PIPELINES_DELETE = "pipelines.delete"
    PIPELINES_PUBLISH = "pipelines.publish"
    
    # Collection permissions
    COLLECTIONS_READ = "collections.read"
    COLLECTIONS_WRITE = "collections.write"
    COLLECTIONS_DELETE = "collections.delete"
    COLLECTIONS_ADMIN = "collections.admin"
    
    # Admin permissions
    USERS_READ = "users.read"
    USERS_MANAGE = "users.manage"
    SYSTEM_CONFIG = "system.config"
    SYSTEM_MONITOR = "system.monitor"
    AUDIT_READ = "audit.read"

@dataclass
class Role:
    name: str
    description: str
    permissions: Set[Permission]
    inherits_from: List[str] = None

# Define roles
ROLES = {
    "viewer": Role(
        name="viewer",
        description="Read-only access to conversations and models",
        permissions={
            Permission.CONVERSATIONS_READ,
            Permission.MODELS_READ,
            Permission.PIPELINES_READ,
            Permission.COLLECTIONS_READ
        }
    ),
    
    "user": Role(
        name="user",
        description="Standard user with conversation and basic pipeline access",
        permissions={
            Permission.CONVERSATIONS_READ,
            Permission.CONVERSATIONS_WRITE,
            Permission.CONVERSATIONS_SHARE,
            Permission.MODELS_READ,
            Permission.PIPELINES_READ,
            Permission.PIPELINES_EXECUTE,
            Permission.COLLECTIONS_READ,
            Permission.COLLECTIONS_WRITE
        }
    ),
    
    "developer": Role(
        name="developer",
        description="Developer access with pipeline creation and model management",
        permissions={
            Permission.CONVERSATIONS_READ,
            Permission.CONVERSATIONS_WRITE,
            Permission.CONVERSATIONS_DELETE,
            Permission.CONVERSATIONS_SHARE,
            Permission.MODELS_READ,
            Permission.MODELS_MANAGE,
            Permission.PIPELINES_READ,
            Permission.PIPELINES_WRITE,
            Permission.PIPELINES_EXECUTE,
            Permission.PIPELINES_DELETE,
            Permission.PIPELINES_PUBLISH,
            Permission.COLLECTIONS_READ,
            Permission.COLLECTIONS_WRITE,
            Permission.COLLECTIONS_DELETE
        }
    ),
    
    "admin": Role(
        name="admin",
        description="Full system administration access",
        permissions=set(Permission)  # All permissions
    )
}

class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self.roles = ROLES
        self.user_roles = {}  # Cache for user roles
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign role to user."""
        if role_name not in self.roles:
            return False
        
        # Store in database
        self._store_user_role(user_id, role_name)
        
        # Update cache
        self.user_roles[user_id] = role_name
        
        return True
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user_role = self._get_user_role(user_id)
        
        if not user_role:
            return False
        
        role = self.roles.get(user_role)
        if not role:
            return False
        
        return permission in role.permissions
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user."""
        user_role = self._get_user_role(user_id)
        
        if not user_role:
            return set()
        
        role = self.roles.get(user_role)
        if not role:
            return set()
        
        return role.permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract user_id from request context
                user_id = self._get_current_user_id()
                
                if not self.check_permission(user_id, permission):
                    raise PermissionError(f"Permission {permission.value} required")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator

# Usage example
rbac = RBACManager()

@rbac.require_permission(Permission.PIPELINES_WRITE)
def create_pipeline(pipeline_config: Dict) -> str:
    """Create new pipeline (requires PIPELINES_WRITE permission)."""
    # Implementation here
    pass
```

### OAuth Integration

#### Multi-Provider OAuth Setup
```python
# auth/oauth.py
from authlib.integrations.flask_client import OAuth
from typing import Dict, Any, Optional

class OAuthManager:
    """Multi-provider OAuth authentication."""
    
    def __init__(self, app):
        self.oauth = OAuth(app)
        self._register_providers()
    
    def _register_providers(self):
        """Register OAuth providers."""
        
        # GitHub OAuth
        self.github = self.oauth.register(
            name='github',
            client_id=os.getenv('GITHUB_CLIENT_ID'),
            client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
            server_metadata_url='https://api.github.com/.well-known/openid_configuration',
            client_kwargs={
                'scope': 'user:email'
            }
        )
        
        # Google OAuth
        self.google = self.oauth.register(
            name='google',
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            server_metadata_url='https://accounts.google.com/.well-known/openid_configuration',
            client_kwargs={
                'scope': 'openid email profile'
            }
        )
        
        # Microsoft OAuth
        self.microsoft = self.oauth.register(
            name='microsoft',
            client_id=os.getenv('MICROSOFT_CLIENT_ID'),
            client_secret=os.getenv('MICROSOFT_CLIENT_SECRET'),
            server_metadata_url='https://login.microsoftonline.com/common/v2.0/.well-known/openid_configuration',
            client_kwargs={
                'scope': 'openid email profile'
            }
        )
    
    def authenticate(self, provider: str, redirect_uri: str) -> str:
        """Initiate OAuth authentication."""
        provider_client = getattr(self, provider, None)
        
        if not provider_client:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return provider_client.authorize_redirect(redirect_uri)
    
    def handle_callback(self, provider: str) -> Dict[str, Any]:
        """Handle OAuth callback and extract user info."""
        provider_client = getattr(self, provider, None)
        
        if not provider_client:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Get access token
        token = provider_client.authorize_access_token()
        
        # Get user info
        user_info = provider_client.parse_id_token(token)
        
        # Normalize user data across providers
        return self._normalize_user_data(provider, user_info)
    
    def _normalize_user_data(self, provider: str, user_info: Dict) -> Dict[str, Any]:
        """Normalize user data from different providers."""
        
        normalized = {
            'provider': provider,
            'provider_id': str(user_info.get('sub', user_info.get('id'))),
            'email': user_info.get('email'),
            'name': user_info.get('name'),
            'avatar_url': user_info.get('picture', user_info.get('avatar_url')),
            'verified': user_info.get('email_verified', False)
        }
        
        # Provider-specific handling
        if provider == 'github':
            normalized['username'] = user_info.get('login')
        elif provider == 'google':
            normalized['given_name'] = user_info.get('given_name')
            normalized['family_name'] = user_info.get('family_name')
        
        return normalized
```

---

## Data Protection

### Encryption Implementation

#### Encryption at Rest
```python
# security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, bytes
import base64
import os

class EncryptionManager:
    """Manage encryption for sensitive data."""
    
    def __init__(self):
        self.master_key = self._get_master_key()
        self.fernet = Fernet(self.master_key)
    
    def _get_master_key(self) -> bytes:
        """Get or generate master encryption key."""
        key_file = "encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Store securely (in production, use a key management service)
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            return key
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded string."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.fernet.encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded data."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def encrypt_file(self, file_path: str, output_path: str = None):
        """Encrypt file contents."""
        if output_path is None:
            output_path = file_path + '.encrypted'
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted = self.fernet.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str):
        """Decrypt file contents."""
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted = self.fernet.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)

# Database field encryption
class EncryptedField:
    """Encrypted database field."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = '_' + name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        encrypted_value = getattr(obj, self.private_name, None)
        if encrypted_value is None:
            return None
        
        return self.encryption_manager.decrypt(encrypted_value)
    
    def __set__(self, obj, value):
        if value is None:
            setattr(obj, self.private_name, None)
        else:
            encrypted_value = self.encryption_manager.encrypt(value)
            setattr(obj, self.private_name, encrypted_value)

# Usage example
encryption_manager = EncryptionManager()

class User:
    """User model with encrypted fields."""
    
    # Encrypt sensitive fields
    api_key = EncryptedField(encryption_manager)
    oauth_token = EncryptedField(encryption_manager)
    
    def __init__(self, username: str):
        self.username = username
        self.api_key = None
        self.oauth_token = None
```

#### Encryption in Transit
```python
# security/tls_config.py
import ssl
from typing import Dict, Any

class TLSManager:
    """TLS/SSL configuration management."""
    
    def __init__(self):
        self.min_tls_version = ssl.TLSVersion.TLSv1_2
        self.cipher_suites = [
            'ECDHE-RSA-AES256-GCM-SHA384',
            'ECDHE-RSA-AES128-GCM-SHA256',
            'ECDHE-RSA-AES256-SHA384',
            'ECDHE-RSA-AES128-SHA256'
        ]
    
    def create_ssl_context(self, cert_file: str, key_file: str) -> ssl.SSLContext:
        """Create secure SSL context."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        
        # Set minimum TLS version
        context.minimum_version = self.min_tls_version
        
        # Load certificates
        context.load_cert_chain(cert_file, key_file)
        
        # Set cipher suites
        context.set_ciphers(':'.join(self.cipher_suites))
        
        # Security options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE
        
        return context
    
    def get_nginx_tls_config(self) -> str:
        """Generate Nginx TLS configuration."""
        return f"""
        # TLS Configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers {':'.join(self.cipher_suites)};
        ssl_prefer_server_ciphers off;
        
        # SSL Session
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        ssl_session_tickets off;
        
        # OCSP Stapling
        ssl_stapling on;
        ssl_stapling_verify on;
        
        # Security Headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
        """
```

### Data Loss Prevention (DLP)

#### Content Scanning and Classification
```python
# security/dlp.py
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class SensitivityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class ScanResult:
    sensitivity: SensitivityLevel
    detected_patterns: List[str]
    redacted_content: str
    risk_score: float

class DLPScanner:
    """Data Loss Prevention scanner."""
    
    def __init__(self):
        self.patterns = {
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'password': r'(?i)password["\s]*[:=]["\s]*[A-Za-z0-9!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]{8,}',
            'private_key': r'-----BEGIN.*PRIVATE KEY-----',
            'aws_access_key': r'\bAKIA[0-9A-Z]{16}\b',
            'database_url': r'(?i)(postgres|mysql|mongodb)://[^\s]+',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }
        
        self.sensitivity_rules = {
            'credit_card': SensitivityLevel.RESTRICTED,
            'ssn': SensitivityLevel.RESTRICTED,
            'password': SensitivityLevel.CONFIDENTIAL,
            'private_key': SensitivityLevel.RESTRICTED,
            'api_key': SensitivityLevel.CONFIDENTIAL,
            'aws_access_key': SensitivityLevel.CONFIDENTIAL,
            'database_url': SensitivityLevel.CONFIDENTIAL
        }
    
    def scan_content(self, content: str) -> ScanResult:
        """Scan content for sensitive information."""
        detected_patterns = []
        redacted_content = content
        max_sensitivity = SensitivityLevel.PUBLIC
        risk_score = 0.0
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, content)
            
            if matches:
                detected_patterns.append(pattern_name)
                
                # Update sensitivity level
                pattern_sensitivity = self.sensitivity_rules.get(
                    pattern_name, SensitivityLevel.INTERNAL
                )
                
                if self._is_higher_sensitivity(pattern_sensitivity, max_sensitivity):
                    max_sensitivity = pattern_sensitivity
                
                # Calculate risk score
                risk_score += len(matches) * self._get_pattern_weight(pattern_name)
                
                # Redact content
                redacted_content = re.sub(
                    pattern, 
                    self._get_redaction_placeholder(pattern_name),
                    redacted_content
                )
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        return ScanResult(
            sensitivity=max_sensitivity,
            detected_patterns=detected_patterns,
            redacted_content=redacted_content,
            risk_score=risk_score
        )
    
    def _is_higher_sensitivity(self, level1: SensitivityLevel, level2: SensitivityLevel) -> bool:
        """Check if level1 is higher sensitivity than level2."""
        sensitivity_order = {
            SensitivityLevel.PUBLIC: 0,
            SensitivityLevel.INTERNAL: 1,
            SensitivityLevel.CONFIDENTIAL: 2,
            SensitivityLevel.RESTRICTED: 3
        }
        
        return sensitivity_order[level1] > sensitivity_order[level2]
    
    def _get_pattern_weight(self, pattern_name: str) -> float:
        """Get risk weight for pattern."""
        weights = {
            'credit_card': 0.9,
            'ssn': 0.9,
            'password': 0.8,
            'private_key': 0.9,
            'api_key': 0.7,
            'aws_access_key': 0.8,
            'database_url': 0.6,
            'email': 0.2,
            'phone': 0.3,
            'ip_address': 0.1
        }
        
        return weights.get(pattern_name, 0.1)
    
    def _get_redaction_placeholder(self, pattern_name: str) -> str:
        """Get redaction placeholder for pattern."""
        placeholders = {
            'credit_card': '[CREDIT_CARD_REDACTED]',
            'ssn': '[SSN_REDACTED]',
            'password': '[PASSWORD_REDACTED]',
            'private_key': '[PRIVATE_KEY_REDACTED]',
            'api_key': '[API_KEY_REDACTED]',
            'aws_access_key': '[AWS_KEY_REDACTED]',
            'database_url': '[DATABASE_URL_REDACTED]',
            'email': '[EMAIL_REDACTED]',
            'phone': '[PHONE_REDACTED]',
            'ip_address': '[IP_ADDRESS_REDACTED]'
        }
        
        return placeholders.get(pattern_name, '[SENSITIVE_DATA_REDACTED]')

# DLP Policy Enforcement
class DLPPolicy:
    """Data Loss Prevention policy enforcement."""
    
    def __init__(self):
        self.scanner = DLPScanner()
        self.policies = {
            'block_restricted': True,
            'redact_confidential': True,
            'log_all_detections': True,
            'alert_on_high_risk': True,
            'quarantine_threshold': 0.8
        }
    
    def enforce_policy(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str, Dict]:
        """Enforce DLP policy on content."""
        scan_result = self.scanner.scan_content(content)
        
        # Policy decisions
        allow_content = True
        final_content = content
        actions_taken = []
        
        # Block restricted content
        if (scan_result.sensitivity == SensitivityLevel.RESTRICTED and 
            self.policies['block_restricted']):
            allow_content = False
            actions_taken.append('blocked_restricted_content')
        
        # Redact confidential content
        elif (scan_result.sensitivity in [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.RESTRICTED] and
              self.policies['redact_confidential']):
            final_content = scan_result.redacted_content
            actions_taken.append('redacted_sensitive_content')
        
        # High risk quarantine
        if (scan_result.risk_score >= self.policies['quarantine_threshold']):
            allow_content = False
            actions_taken.append('quarantined_high_risk_content')
        
        # Logging and alerting
        if self.policies['log_all_detections'] and scan_result.detected_patterns:
            self._log_detection(scan_result, context)
            actions_taken.append('logged_detection')
        
        if (self.policies['alert_on_high_risk'] and 
            scan_result.risk_score >= 0.6):
            self._send_security_alert(scan_result, context)
            actions_taken.append('sent_security_alert')
        
        return allow_content, final_content, {
            'scan_result': scan_result,
            'actions_taken': actions_taken
        }
    
    def _log_detection(self, scan_result: ScanResult, context: Dict[str, Any]):
        """Log DLP detection."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'dlp_detection',
            'sensitivity': scan_result.sensitivity.value,
            'detected_patterns': scan_result.detected_patterns,
            'risk_score': scan_result.risk_score,
            'user_id': context.get('user_id'),
            'session_id': context.get('session_id'),
            'ip_address': context.get('ip_address')
        }
        
        # Log to security log
        self._write_security_log(log_entry)
    
    def _send_security_alert(self, scan_result: ScanResult, context: Dict[str, Any]):
        """Send security alert for high-risk detection."""
        alert = {
            'alert_type': 'dlp_high_risk_detection',
            'severity': 'high' if scan_result.risk_score >= 0.8 else 'medium',
            'detected_patterns': scan_result.detected_patterns,
            'risk_score': scan_result.risk_score,
            'user_context': {
                'user_id': context.get('user_id'),
                'ip_address': context.get('ip_address'),
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        # Send to security team
        self._send_alert(alert)
```

---

## Infrastructure Security

### Container Security

#### Docker Security Configuration
```dockerfile
# Secure Dockerfile template
FROM python:3.11-slim AS base

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install security tools
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Remove unnecessary files
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -delete

# Set file permissions
RUN chmod -R 755 /app && \
    chmod -R 644 /app/*.py

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

#### Kubernetes Security Policies
```yaml
# k8s/security-policies.yaml
apiVersion: v1
kind: SecurityContext
metadata:
  name: ollama-workbench-security-context
spec:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
      - ALL
    add:
      - NET_BIND_SERVICE
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ollama-workbench-network-policy
  namespace: ollama-workbench
spec:
  podSelector:
    matchLabels:
      app: ollama-workbench
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ollama-workbench
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: ollama-workbench
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
  - to: []  # Allow external HTTPS
    ports:
    - protocol: TCP
      port: 443

---
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: ollama-workbench-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### Network Security

#### Firewall Rules
```bash
#!/bin/bash
# scripts/configure_firewall.sh

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict to specific IPs in production)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow application ports (restrict to internal network)
iptables -A INPUT -p tcp --dport 8501 -s 10.0.0.0/8 -j ACCEPT  # Streamlit
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT  # API
iptables -A INPUT -p tcp --dport 8001 -s 10.0.0.0/8 -j ACCEPT  # Pipeline

# Allow database ports (internal only)
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/8 -j ACCEPT  # PostgreSQL
iptables -A INPUT -p tcp --dport 6379 -s 10.0.0.0/8 -j ACCEPT  # Redis

# Rate limiting for SSH
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 -j DROP

# Rate limiting for HTTP
iptables -A INPUT -p tcp --dport 80 -m conntrack --ctstate NEW -m recent --set
iptables -A INPUT -p tcp --dport 80 -m conntrack --ctstate NEW -m recent --update --seconds 1 --hitcount 20 -j DROP

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED: "

# Save rules
iptables-save > /etc/iptables/rules.v4
```

---

## Application Security

### Input Validation and Sanitization

#### Comprehensive Input Validation
```python
# security/validation.py
import re
import html
import bleach
from typing import Any, Dict, List, Optional, Union
from marshmallow import Schema, fields, ValidationError, validates_schema

class SecureInputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        # Allowed HTML tags for rich text
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote',
            'code', 'pre', 'a'
        ]
        
        self.allowed_attributes = {
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title']
        }
        
        # Dangerous patterns
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'data:.*base64',  # Data URLs
            r'vbscript:',  # VBScript
            r'expression\(',  # CSS expressions
        ]
    
    def validate_string(
        self, 
        value: str, 
        max_length: int = 1000,
        allow_html: bool = False,
        required: bool = True
    ) -> str:
        """Validate and sanitize string input."""
        
        if value is None:
            if required:
                raise ValidationError("Field is required")
            return ""
        
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
        
        # Length validation
        if len(value) > max_length:
            raise ValidationError(f"Value exceeds maximum length of {max_length}")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValidationError("Input contains potentially dangerous content")
        
        # Sanitization
        if allow_html:
            # Allow safe HTML tags
            sanitized = bleach.clean(
                value,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
        else:
            # Escape all HTML
            sanitized = html.escape(value)
        
        return sanitized
    
    def validate_email(self, email: str) -> str:
        """Validate email address."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            raise ValidationError("Invalid email format")
        
        return email.lower().strip()
    
    def validate_model_name(self, model_name: str) -> str:
        """Validate model name format."""
        # Allow alphanumeric, hyphens, underscores, colons for tags
        pattern = r'^[a-zA-Z0-9\-_:\.]+$'
        
        if not re.match(pattern, model_name):
            raise ValidationError("Model name contains invalid characters")
        
        if len(model_name) > 100:
            raise ValidationError("Model name too long")
        
        return model_name
    
    def validate_json_data(self, data: str, max_size: int = 10000) -> Dict:
        """Validate and parse JSON data."""
        if len(data) > max_size:
            raise ValidationError(f"JSON data exceeds maximum size of {max_size} bytes")
        
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {str(e)}")
        
        # Recursive validation for nested objects
        self._validate_json_content(parsed)
        
        return parsed
    
    def _validate_json_content(self, obj: Any, depth: int = 0):
        """Recursively validate JSON content."""
        max_depth = 10
        
        if depth > max_depth:
            raise ValidationError("JSON nesting too deep")
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise ValidationError("JSON keys must be strings")
                
                if len(key) > 100:
                    raise ValidationError("JSON key too long")
                
                self._validate_json_content(value, depth + 1)
        
        elif isinstance(obj, list):
            if len(obj) > 1000:
                raise ValidationError("JSON array too large")
            
            for item in obj:
                self._validate_json_content(item, depth + 1)
        
        elif isinstance(obj, str):
            if len(obj) > 10000:
                raise ValidationError("JSON string value too long")

# API request schemas
class ConversationSchema(Schema):
    """Schema for conversation requests."""
    
    model = fields.Str(required=True, validate=lambda x: len(x) <= 100)
    prompt = fields.Str(required=True, validate=lambda x: len(x) <= 10000)
    temperature = fields.Float(missing=0.7, validate=lambda x: 0 <= x <= 2)
    max_tokens = fields.Int(missing=150, validate=lambda x: 1 <= x <= 4096)
    conversation_id = fields.Str(missing=None, validate=lambda x: len(x or "") <= 50)
    
    @validates_schema
    def validate_model_name(self, data, **kwargs):
        validator = SecureInputValidator()
        data['model'] = validator.validate_model_name(data['model'])
        data['prompt'] = validator.validate_string(data['prompt'], max_length=10000)

class PipelineSchema(Schema):
    """Schema for pipeline creation requests."""
    
    name = fields.Str(required=True, validate=lambda x: 1 <= len(x) <= 100)
    description = fields.Str(missing="", validate=lambda x: len(x) <= 1000)
    config = fields.Raw(required=True)
    is_public = fields.Bool(missing=False)
    tags = fields.List(fields.Str(validate=lambda x: len(x) <= 50), missing=[])
    
    @validates_schema
    def validate_pipeline_data(self, data, **kwargs):
        validator = SecureInputValidator()
        
        # Validate name (alphanumeric and underscores only)
        if not re.match(r'^[a-zA-Z0-9_\-]+$', data['name']):
            raise ValidationError("Pipeline name must contain only alphanumeric characters, underscores, and hyphens")
        
        # Validate description
        data['description'] = validator.validate_string(
            data['description'], 
            max_length=1000,
            allow_html=False
        )
        
        # Validate config JSON
        if isinstance(data['config'], str):
            data['config'] = validator.validate_json_data(data['config'])
        
        # Validate tags
        for i, tag in enumerate(data['tags']):
            data['tags'][i] = validator.validate_string(tag, max_length=50)
```

### SQL Injection Prevention

#### Parameterized Queries and ORM Usage
```python
# security/database.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Dict, Any, Optional
import logging

class SecureDatabase:
    """Secure database operations with SQL injection prevention."""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            echo=False,  # Don't log SQL in production
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def execute_query(
        self, 
        query: str, 
        params: Dict[str, Any] = None,
        fetch_results: bool = True
    ) -> Optional[List[Dict]]:
        """Execute parameterized query safely."""
        
        # Validate query doesn't contain dangerous operations
        self._validate_query_safety(query)
        
        with self.SessionLocal() as session:
            try:
                # Use parameterized query
                result = session.execute(text(query), params or {})
                
                if fetch_results:
                    # Convert result to list of dictionaries
                    columns = result.keys()
                    rows = result.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                else:
                    session.commit()
                    return None
                    
            except Exception as e:
                session.rollback()
                logging.error(f"Database query failed: {str(e)}")
                raise
    
    def _validate_query_safety(self, query: str):
        """Validate query for potential SQL injection."""
        
        # Convert to lowercase for pattern matching
        query_lower = query.lower().strip()
        
        # Dangerous patterns in queries
        dangerous_patterns = [
            r';\s*(drop|delete|truncate|alter|create|insert|update)',
            r'union\s+select',
            r'(exec|execute|sp_)',
            r'(xp_|sp_oa)',
            r'into\s+(outfile|dumpfile)',
            r'load_file\s*\(',
            r'benchmark\s*\(',
            r'sleep\s*\(',
            r'information_schema',
            r'mysql\.user',
            r'pg_shadow',
            r'systables',
            r'sysobjects'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                raise ValueError(f"Query contains potentially dangerous pattern: {pattern}")
        
        # Check for comment-based injection attempts
        if '--' in query or '/*' in query:
            raise ValueError("Query contains SQL comments which are not allowed")

# Safe database models using SQLAlchemy ORM
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model with secure field definitions."""
    
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # For local auth
    oauth_provider = Column(String(50), nullable=True)
    oauth_id = Column(String(255), nullable=True)
    role = Column(String(20), default='user', nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(255), nullable=True)  # Encrypted
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    pipelines = relationship("Pipeline", back_populates="owner")

class Conversation(Base):
    """Conversation model with security controls."""
    
    __tablename__ = 'conversations'
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)
    title = Column(String(255), nullable=True)
    model_name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=True)
    settings = Column(Text, nullable=True)  # JSON string
    is_public = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

# Repository pattern for secure data access
class UserRepository:
    """Secure user data access layer."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create new user with validation."""
        
        # Validate input data
        validator = SecureInputValidator()
        validated_data = {
            'username': validator.validate_string(user_data['username'], max_length=50),
            'email': validator.validate_email(user_data['email']),
            'role': user_data.get('role', 'user')
        }
        
        # Check for existing user
        existing_user = self.db.query(User).filter(
            (User.username == validated_data['username']) |
            (User.email == validated_data['email'])
        ).first()
        
        if existing_user:
            raise ValueError("User with this username or email already exists")
        
        # Create user
        user = User(**validated_data)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID with validation."""
        
        # Validate UUID format
        if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', user_id):
            return None
        
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_conversations(
        self, 
        user_id: str, 
        limit: int = 20, 
        offset: int = 0
    ) -> List[Conversation]:
        """Get user conversations with pagination."""
        
        # Validate parameters
        limit = max(1, min(limit, 100))  # Limit between 1 and 100
        offset = max(0, offset)
        
        return self.db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(
            Conversation.updated_at.desc()
        ).limit(limit).offset(offset).all()
```

This security and compliance documentation provides comprehensive coverage of security controls, data protection measures, and compliance frameworks necessary for enterprise deployment of Ollama Workbench. The implementation focuses on defense in depth, zero trust principles, and privacy by design.