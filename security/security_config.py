# security/security_config.py

"""
Security Configuration Management
Handles security settings, validation, and compliance requirements.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import secrets
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    
    # Authentication settings
    enable_authentication: bool = True
    authentication_method: str = "jwt"  # jwt, basic, oauth2, saml
    session_timeout_minutes: int = 480  # 8 hours
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_uppercase: bool = True
    password_require_numbers: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # Encryption settings
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    
    # Access control settings
    enable_rbac: bool = True
    default_role: str = "user"
    admin_role: str = "admin"
    enable_api_rate_limiting: bool = True
    api_rate_limit_per_minute: int = 100
    
    # Audit and compliance
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 365
    log_all_api_calls: bool = True
    log_data_access: bool = True
    log_sensitive_operations: bool = True
    
    # Data protection
    enable_data_classification: bool = True
    enable_dlp: bool = True  # Data Loss Prevention
    mask_sensitive_data: bool = True
    pii_detection_enabled: bool = True
    
    # Network security
    enable_https_only: bool = True
    enable_cors_protection: bool = True
    allowed_origins: List[str] = None
    enable_csrf_protection: bool = True
    
    # Compliance settings
    compliance_mode: str = "standard"  # standard, gdpr, hipaa, sox, iso27001
    data_retention_days: int = 2555  # 7 years default
    enable_right_to_deletion: bool = True
    enable_data_portability: bool = True
    
    # Security monitoring
    enable_intrusion_detection: bool = True
    enable_anomaly_detection: bool = True
    alert_on_suspicious_activity: bool = True
    security_scan_interval_hours: int = 24
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:8501", "https://localhost:8501"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate security configuration and return any issues"""
        issues = []
        
        # Password policy validation
        if self.password_min_length < 8:
            issues.append("Password minimum length should be at least 8 characters")
        
        # Session timeout validation
        if self.session_timeout_minutes < 15:
            issues.append("Session timeout should be at least 15 minutes")
        if self.session_timeout_minutes > 1440:  # 24 hours
            issues.append("Session timeout should not exceed 24 hours")
        
        # Rate limiting validation
        if self.api_rate_limit_per_minute < 1:
            issues.append("API rate limit should be at least 1 request per minute")
        
        # Audit retention validation
        if self.audit_log_retention_days < 30:
            issues.append("Audit log retention should be at least 30 days")
        
        # HTTPS validation in production
        if not self.enable_https_only:
            issues.append("HTTPS should be enabled in production environments")
        
        return issues

class SecurityConfigManager:
    """Manages security configuration loading, saving, and validation"""
    
    def __init__(self, config_dir: str = "security"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "security_config.json"
        self.secrets_file = self.config_dir / "secrets.json"
        self._config: Optional[SecurityConfig] = None
        self._secrets: Dict[str, str] = {}
        
        # Load configuration
        self.load_config()
        self.load_secrets()
    
    def load_config(self) -> SecurityConfig:
        """Load security configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._config = SecurityConfig.from_dict(config_data)
                logger.info("Security configuration loaded successfully")
            else:
                self._config = SecurityConfig()
                self.save_config()
                logger.info("Default security configuration created")
        except Exception as e:
            logger.error(f"Failed to load security configuration: {e}")
            self._config = SecurityConfig()
        
        return self._config
    
    def save_config(self) -> None:
        """Save security configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config.to_dict(), f, indent=2)
            
            # Set restrictive permissions on config file
            os.chmod(self.config_file, 0o600)
            logger.info("Security configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save security configuration: {e}")
    
    def load_secrets(self) -> Dict[str, str]:
        """Load security secrets (keys, tokens, etc.)"""
        try:
            if self.secrets_file.exists():
                with open(self.secrets_file, 'r') as f:
                    self._secrets = json.load(f)
                logger.info("Security secrets loaded successfully")
            else:
                self._secrets = self._generate_default_secrets()
                self.save_secrets()
                logger.info("Default security secrets generated")
        except Exception as e:
            logger.error(f"Failed to load security secrets: {e}")
            self._secrets = self._generate_default_secrets()
        
        return self._secrets
    
    def save_secrets(self) -> None:
        """Save security secrets to file"""
        try:
            with open(self.secrets_file, 'w') as f:
                json.dump(self._secrets, f, indent=2)
            
            # Set very restrictive permissions on secrets file
            os.chmod(self.secrets_file, 0o600)
            logger.info("Security secrets saved successfully")
        except Exception as e:
            logger.error(f"Failed to save security secrets: {e}")
    
    def _generate_default_secrets(self) -> Dict[str, str]:
        """Generate default security secrets"""
        return {
            "jwt_secret_key": secrets.token_urlsafe(64),
            "encryption_key": secrets.token_urlsafe(32),
            "api_key_salt": secrets.token_urlsafe(16),
            "session_secret": secrets.token_urlsafe(32),
            "csrf_secret": secrets.token_urlsafe(16),
            "audit_key": secrets.token_urlsafe(32)
        }
    
    def get_config(self) -> SecurityConfig:
        """Get current security configuration"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> SecurityConfig:
        """Update security configuration"""
        if self._config is None:
            self.load_config()
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Validate configuration
        issues = self._config.validate()
        if issues:
            logger.warning(f"Security configuration issues: {issues}")
        
        self.save_config()
        return self._config
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get a security secret"""
        return self._secrets.get(key)
    
    def set_secret(self, key: str, value: str) -> None:
        """Set a security secret"""
        self._secrets[key] = value
        self.save_secrets()
    
    def rotate_secrets(self) -> None:
        """Rotate all security secrets"""
        old_secrets = self._secrets.copy()
        self._secrets = self._generate_default_secrets()
        self.save_secrets()
        
        logger.info("Security secrets rotated", extra={
            "rotated_keys": list(old_secrets.keys()),
            "timestamp": self._get_timestamp()
        })
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current security configuration"""
        if self._config is None:
            self.load_config()
        
        issues = self._config.validate()
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": self._config.to_dict(),
            "timestamp": self._get_timestamp()
        }
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        config = self.get_config()
        
        compliance_checks = {
            "GDPR": {
                "encryption_enabled": config.enable_encryption,
                "audit_logging": config.enable_audit_logging,
                "right_to_deletion": config.enable_right_to_deletion,
                "data_portability": config.enable_data_portability,
                "data_classification": config.enable_data_classification
            },
            "HIPAA": {
                "encryption_at_rest": config.encrypt_at_rest,
                "encryption_in_transit": config.encrypt_in_transit,
                "access_controls": config.enable_rbac,
                "audit_controls": config.enable_audit_logging,
                "authentication": config.enable_authentication
            },
            "SOX": {
                "audit_logging": config.enable_audit_logging,
                "access_controls": config.enable_rbac,
                "data_retention": config.audit_log_retention_days >= 2555,  # 7 years
                "change_controls": True  # Implemented via access controls
            },
            "ISO27001": {
                "information_security_policy": True,
                "asset_management": config.enable_data_classification,
                "access_control": config.enable_rbac,
                "cryptography": config.enable_encryption,
                "security_monitoring": config.enable_intrusion_detection,
                "incident_management": config.alert_on_suspicious_activity
            }
        }
        
        # Calculate compliance scores
        compliance_scores = {}
        for standard, checks in compliance_checks.items():
            passed = sum(1 for check in checks.values() if check)
            total = len(checks)
            compliance_scores[standard] = {
                "score": passed / total,
                "passed": passed,
                "total": total,
                "details": checks
            }
        
        return {
            "timestamp": self._get_timestamp(),
            "compliance_mode": config.compliance_mode,
            "scores": compliance_scores,
            "overall_score": sum(score["score"] for score in compliance_scores.values()) / len(compliance_scores),
            "recommendations": self._get_compliance_recommendations(compliance_checks)
        }
    
    def _get_compliance_recommendations(self, checks: Dict[str, Dict[str, bool]]) -> List[str]:
        """Get compliance improvement recommendations"""
        recommendations = []
        
        for standard, standard_checks in checks.items():
            for check_name, passed in standard_checks.items():
                if not passed:
                    recommendations.append(f"{standard}: Enable {check_name.replace('_', ' ')}")
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def export_config(self, filepath: str) -> None:
        """Export security configuration to file"""
        config_data = {
            "config": self.get_config().to_dict(),
            "compliance_report": self.get_compliance_report(),
            "validation": self.validate_config(),
            "export_timestamp": self._get_timestamp()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Security configuration exported to {filepath}")

# Global security configuration manager
_security_config_manager = None

def get_security_config() -> SecurityConfig:
    """Get the global security configuration"""
    global _security_config_manager
    if _security_config_manager is None:
        _security_config_manager = SecurityConfigManager()
    return _security_config_manager.get_config()

def get_security_config_manager() -> SecurityConfigManager:
    """Get the global security configuration manager"""
    global _security_config_manager
    if _security_config_manager is None:
        _security_config_manager = SecurityConfigManager()
    return _security_config_manager

def validate_security_settings() -> Dict[str, Any]:
    """Validate current security settings"""
    return get_security_config_manager().validate_config()

def generate_security_report() -> Dict[str, Any]:
    """Generate comprehensive security report"""
    manager = get_security_config_manager()
    
    return {
        "timestamp": manager._get_timestamp(),
        "configuration": manager.get_config().to_dict(),
        "validation": manager.validate_config(),
        "compliance": manager.get_compliance_report(),
        "secrets_status": {
            "total_secrets": len(manager._secrets),
            "secrets_exist": all(secret for secret in manager._secrets.values())
        }
    }