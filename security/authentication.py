# security/authentication.py

"""
Enhanced Authentication System
Implements JWT-based authentication, session management, and multi-factor authentication.
"""

import os
import jwt
import bcrypt
import secrets
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from functools import wraps
import streamlit as st
from .security_config import get_security_config, get_security_config_manager

logger = logging.getLogger(__name__)

@dataclass
class User:
    """User data model"""
    username: str
    email: str
    role: str = "user"
    permissions: List[str] = None
    created_at: datetime = None
    last_login: datetime = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_hash: Optional[str] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    session_token: Optional[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.last_login:
            data['last_login'] = self.last_login.isoformat()
        if self.locked_until:
            data['locked_until'] = self.locked_until.isoformat()
        
        # Remove sensitive data unless explicitly requested
        if not include_sensitive:
            data.pop('password_hash', None)
            data.pop('mfa_secret', None)
            data.pop('session_token', None)
        
        return data
    
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def can_login(self) -> bool:
        """Check if user can login"""
        return not self.is_locked()

class AuthenticationManager:
    """Enhanced authentication manager with JWT, session management, and security features"""
    
    def __init__(self):
        self.config = get_security_config()
        self.config_manager = get_security_config_manager()
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.login_attempts: Dict[str, List[datetime]] = {}
        
        # Initialize default admin user if none exists
        self._init_default_admin()
    
    def _init_default_admin(self) -> None:
        """Initialize default admin user"""
        if not self.users:
            admin_password = os.getenv('ADMIN_PASSWORD', 'admin123!@#')
            admin_user = User(
                username="admin",
                email="admin@localhost",
                role="admin",
                permissions=["*"]  # All permissions
            )
            
            admin_user.password_hash = self._hash_password(admin_password)
            self.users["admin"] = admin_user
            
            logger.warning("Default admin user created. Please change the password immediately!")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user"""
        secret_key = self.config_manager.get_secret('jwt_secret_key')
        
        payload = {
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'permissions': user.permissions,
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(minutes=self.config.session_timeout_minutes),
            'jti': secrets.token_urlsafe(16)  # JWT ID for token invalidation
        }
        
        return jwt.encode(payload, secret_key, algorithm='HS256')
    
    def _verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            secret_key = self.config_manager.get_secret('jwt_secret_key')
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            
            # Check if token is in active sessions
            jti = payload.get('jti')
            if jti not in self.active_sessions:
                logger.warning(f"JWT token not in active sessions: {jti}")
                return None
            
            return payload
        except jwt.ExpiredSignatureError:
            logger.info("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def _check_password_policy(self, password: str) -> List[str]:
        """Check password against security policy"""
        issues = []
        
        if len(password) < self.config.password_min_length:
            issues.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one number")
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")
        
        return issues
    
    def _is_rate_limited(self, username: str) -> bool:
        """Check if user is rate limited for login attempts"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self.config.lockout_duration_minutes)
        
        # Clean old attempts
        if username in self.login_attempts:
            self.login_attempts[username] = [
                attempt for attempt in self.login_attempts[username] 
                if attempt > cutoff
            ]
        
        # Check if rate limited
        recent_attempts = len(self.login_attempts.get(username, []))
        return recent_attempts >= self.config.max_login_attempts
    
    def _record_login_attempt(self, username: str, success: bool) -> None:
        """Record login attempt"""
        now = datetime.now(timezone.utc)
        
        if not success:
            if username not in self.login_attempts:
                self.login_attempts[username] = []
            self.login_attempts[username].append(now)
            
            # Lock user if too many attempts
            if len(self.login_attempts[username]) >= self.config.max_login_attempts:
                if username in self.users:
                    self.users[username].locked_until = now + timedelta(minutes=self.config.lockout_duration_minutes)
                    logger.warning(f"User {username} locked due to too many failed login attempts")
        else:
            # Clear login attempts on successful login
            if username in self.login_attempts:
                del self.login_attempts[username]
            
            # Update user login time
            if username in self.users:
                self.users[username].last_login = now
                self.users[username].login_attempts = 0
                self.users[username].locked_until = None
    
    def create_user(self, username: str, email: str, password: str, role: str = "user", 
                   permissions: List[str] = None) -> Tuple[bool, Union[User, List[str]]]:
        """Create a new user"""
        
        # Check if user already exists
        if username in self.users:
            return False, ["User already exists"]
        
        # Validate password
        password_issues = self._check_password_policy(password)
        if password_issues:
            return False, password_issues
        
        # Create user
        user = User(
            username=username,
            email=email,
            role=role,
            permissions=permissions or []
        )
        
        user.password_hash = self._hash_password(password)
        self.users[username] = user
        
        logger.info(f"User created: {username}")
        return True, user
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Authenticate user and return success, token, and message"""
        
        # Check rate limiting
        if self._is_rate_limited(username):
            logger.warning(f"Rate limited login attempt for user: {username}")
            return False, None, "Too many login attempts. Please try again later."
        
        # Check if user exists
        if username not in self.users:
            self._record_login_attempt(username, False)
            return False, None, "Invalid username or password"
        
        user = self.users[username]
        
        # Check if user is locked
        if user.is_locked():
            return False, None, f"Account is locked until {user.locked_until}"
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            self._record_login_attempt(username, False)
            return False, None, "Invalid username or password"
        
        # Generate token
        token = self._generate_jwt_token(user)
        
        # Store session
        payload = self._verify_jwt_token(token)
        if payload:
            jti = payload['jti']
            self.active_sessions[jti] = {
                'username': username,
                'token': token,
                'created_at': datetime.now(timezone.utc),
                'last_activity': datetime.now(timezone.utc),
                'ip_address': self._get_client_ip()
            }
        
        # Record successful login
        self._record_login_attempt(username, True)
        
        logger.info(f"User authenticated successfully: {username}")
        return True, token, "Authentication successful"
    
    def logout_user(self, token: str) -> bool:
        """Logout user by invalidating token"""
        payload = self._verify_jwt_token(token)
        if payload:
            jti = payload['jti']
            if jti in self.active_sessions:
                username = self.active_sessions[jti]['username']
                del self.active_sessions[jti]
                logger.info(f"User logged out: {username}")
                return True
        return False
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token"""
        payload = self._verify_jwt_token(token)
        if payload:
            # Update session activity
            jti = payload['jti']
            if jti in self.active_sessions:
                self.active_sessions[jti]['last_activity'] = datetime.now(timezone.utc)
            return payload
        return None
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.users.get(username)
    
    def update_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, List[str]]:
        """Update user password"""
        if username not in self.users:
            return False, ["User not found"]
        
        user = self.users[username]
        
        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            return False, ["Current password is incorrect"]
        
        # Validate new password
        password_issues = self._check_password_policy(new_password)
        if password_issues:
            return False, password_issues
        
        # Update password
        user.password_hash = self._hash_password(new_password)
        logger.info(f"Password updated for user: {username}")
        
        return True, []
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        now = datetime.now(timezone.utc)
        expired_sessions = []
        
        for jti, session in self.active_sessions.items():
            last_activity = session['last_activity']
            if now - last_activity > timedelta(minutes=self.config.session_timeout_minutes):
                expired_sessions.append(jti)
        
        for jti in expired_sessions:
            del self.active_sessions[jti]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_active_sessions(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active sessions, optionally filtered by username"""
        sessions = []
        
        for jti, session in self.active_sessions.items():
            if username is None or session['username'] == username:
                session_info = session.copy()
                session_info['jti'] = jti
                # Don't include the actual token in the response
                session_info.pop('token', None)
                sessions.append(session_info)
        
        return sessions
    
    def revoke_session(self, jti: str) -> bool:
        """Revoke a specific session"""
        if jti in self.active_sessions:
            username = self.active_sessions[jti]['username']
            del self.active_sessions[jti]
            logger.info(f"Session revoked for user: {username}, JTI: {jti}")
            return True
        return False
    
    def _get_client_ip(self) -> str:
        """Get client IP address (placeholder for actual implementation)"""
        # In a real implementation, this would extract the client IP from request headers
        return "127.0.0.1"

# Global authentication manager
_auth_manager = None

def get_auth_manager() -> AuthenticationManager:
    """Get the global authentication manager"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Authenticate user - convenience function"""
    return get_auth_manager().authenticate_user(username, password)

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token - convenience function"""
    return get_auth_manager().verify_token(token)

def create_jwt_token(user: User) -> str:
    """Create JWT token for user - convenience function"""
    return get_auth_manager()._generate_jwt_token(user)

def require_authentication(func):
    """Decorator to require authentication for functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if authentication is enabled
        config = get_security_config()
        if not config.enable_authentication:
            return func(*args, **kwargs)
        
        # Check for token in Streamlit session state
        token = st.session_state.get('auth_token')
        if not token:
            st.error("Authentication required. Please log in.")
            st.stop()
        
        # Verify token
        payload = verify_jwt_token(token)
        if not payload:
            st.error("Invalid or expired authentication token. Please log in again.")
            st.session_state.pop('auth_token', None)
            st.stop()
        
        # Add user info to session state
        st.session_state['user'] = payload
        
        return func(*args, **kwargs)
    
    return wrapper

def streamlit_auth_ui():
    """Streamlit authentication UI"""
    config = get_security_config()
    
    if not config.enable_authentication:
        st.info("Authentication is disabled in the security configuration.")
        return True
    
    # Check if user is already authenticated
    if 'auth_token' in st.session_state:
        token = st.session_state['auth_token']
        payload = verify_jwt_token(token)
        if payload:
            return True
        else:
            # Token is invalid, remove it
            st.session_state.pop('auth_token', None)
            st.session_state.pop('user', None)
    
    # Show login form
    st.title("🔐 Login Required")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit and username and password:
            success, token, message = authenticate_user(username, password)
            
            if success:
                st.session_state['auth_token'] = token
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(message)
    
    return False