# security/access_control.py

"""
Role-Based Access Control (RBAC) System
Implements comprehensive access control with roles, permissions, and resource-based authorization.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import streamlit as st
from .security_config import get_security_config
from .authentication import get_auth_manager

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Predefined action types"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"

class ResourceType(Enum):
    """Predefined resource types"""
    MODEL = "model"
    CHAT = "chat"
    FILE = "file"
    PROJECT = "project"
    USER = "user"
    SYSTEM = "system"
    CONFIG = "config"
    AUDIT = "audit"
    API = "api"

@dataclass
class Permission:
    """Represents a permission with resource and action"""
    resource: str
    action: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.resource}:{self.action}"
    
    def matches(self, resource: str, action: str, context: Dict[str, Any] = None) -> bool:
        """Check if this permission matches the requested resource and action"""
        # Check wildcard permissions
        if self.resource == "*" or self.action == "*":
            return True
        
        # Check exact match
        if self.resource == resource and self.action == action:
            return self._check_conditions(context or {})
        
        # Check resource hierarchy (e.g., "model.*" matches "model.list", "model.create")
        if self.resource.endswith(".*") and resource.startswith(self.resource[:-1]):
            if self.action == action or self.action == "*":
                return self._check_conditions(context or {})
        
        return False
    
    def _check_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if conditions are met"""
        if not self.conditions:
            return True
        
        for condition_key, condition_value in self.conditions.items():
            if context.get(condition_key) != condition_value:
                return False
        
        return True

@dataclass
class Role:
    """Represents a role with permissions"""
    name: str
    permissions: List[Permission] = field(default_factory=list)
    inherits_from: List[str] = field(default_factory=list)
    description: str = ""
    
    def add_permission(self, permission: Permission) -> None:
        """Add permission to role"""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: Permission) -> None:
        """Remove permission from role"""
        if permission in self.permissions:
            self.permissions.remove(permission)
    
    def has_permission(self, resource: str, action: str, context: Dict[str, Any] = None) -> bool:
        """Check if role has specific permission"""
        for permission in self.permissions:
            if permission.matches(resource, action, context):
                return True
        return False

class AccessControlManager:
    """Manages roles, permissions, and access control"""
    
    def __init__(self):
        self.config = get_security_config()
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[str]] = {}
        
        # Initialize default roles
        self._init_default_roles()
    
    def _init_default_roles(self) -> None:
        """Initialize default roles and permissions"""
        
        # Guest role - very limited access
        guest_role = Role(
            name="guest",
            description="Guest user with minimal read-only access"
        )
        guest_role.add_permission(Permission("chat", "read"))
        guest_role.add_permission(Permission("model", "list"))
        
        # User role - standard user permissions
        user_role = Role(
            name="user",
            description="Standard user with chat and file access"
        )
        user_role.add_permission(Permission("chat", "*"))
        user_role.add_permission(Permission("file", "create"))
        user_role.add_permission(Permission("file", "read"))
        user_role.add_permission(Permission("file", "update"))
        user_role.add_permission(Permission("file", "delete", {"owner": True}))
        user_role.add_permission(Permission("model", "list"))
        user_role.add_permission(Permission("model", "read"))
        user_role.add_permission(Permission("project", "*", {"owner": True}))
        user_role.add_permission(Permission("api", "read"))
        
        # Power User role - advanced features
        power_user_role = Role(
            name="power_user",
            description="Power user with advanced features",
            inherits_from=["user"]
        )
        power_user_role.add_permission(Permission("model", "execute"))
        power_user_role.add_permission(Permission("file", "*"))
        power_user_role.add_permission(Permission("project", "*"))
        power_user_role.add_permission(Permission("api", "*"))
        
        # Moderator role - content moderation
        moderator_role = Role(
            name="moderator",
            description="Moderator with content management permissions",
            inherits_from=["power_user"]
        )
        moderator_role.add_permission(Permission("chat", "admin"))
        moderator_role.add_permission(Permission("file", "admin"))
        moderator_role.add_permission(Permission("user", "read"))
        moderator_role.add_permission(Permission("audit", "read"))
        
        # Admin role - full system access
        admin_role = Role(
            name="admin",
            description="Administrator with full system access"
        )
        admin_role.add_permission(Permission("*", "*"))
        
        # Add roles to manager
        self.roles = {
            "guest": guest_role,
            "user": user_role,
            "power_user": power_user_role,
            "moderator": moderator_role,
            "admin": admin_role
        }
    
    def create_role(self, name: str, permissions: List[Permission] = None, 
                   inherits_from: List[str] = None, description: str = "") -> Role:
        """Create a new role"""
        role = Role(
            name=name,
            permissions=permissions or [],
            inherits_from=inherits_from or [],
            description=description
        )
        
        self.roles[name] = role
        logger.info(f"Role created: {name}")
        return role
    
    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name"""
        return self.roles.get(name)
    
    def delete_role(self, name: str) -> bool:
        """Delete a role"""
        if name in ["admin", "user", "guest"]:  # Protect default roles
            logger.warning(f"Cannot delete default role: {name}")
            return False
        
        if name in self.roles:
            del self.roles[name]
            
            # Remove role from users
            for username in self.user_roles:
                if name in self.user_roles[username]:
                    self.user_roles[username].remove(name)
            
            logger.info(f"Role deleted: {name}")
            return True
        
        return False
    
    def assign_role(self, username: str, role_name: str) -> bool:
        """Assign role to user"""
        if role_name not in self.roles:
            logger.warning(f"Role not found: {role_name}")
            return False
        
        if username not in self.user_roles:
            self.user_roles[username] = []
        
        if role_name not in self.user_roles[username]:
            self.user_roles[username].append(role_name)
            logger.info(f"Role {role_name} assigned to user {username}")
            return True
        
        return False
    
    def revoke_role(self, username: str, role_name: str) -> bool:
        """Revoke role from user"""
        if username in self.user_roles and role_name in self.user_roles[username]:
            self.user_roles[username].remove(role_name)
            logger.info(f"Role {role_name} revoked from user {username}")
            return True
        
        return False
    
    def get_user_roles(self, username: str) -> List[str]:
        """Get roles assigned to user"""
        return self.user_roles.get(username, [self.config.default_role])
    
    def get_user_permissions(self, username: str) -> Set[Permission]:
        """Get all permissions for user (including inherited)"""
        permissions = set()
        user_roles = self.get_user_roles(username)
        
        for role_name in user_roles:
            permissions.update(self._get_role_permissions(role_name))
        
        return permissions
    
    def _get_role_permissions(self, role_name: str, visited: Set[str] = None) -> Set[Permission]:
        """Get all permissions for a role including inherited permissions"""
        if visited is None:
            visited = set()
        
        if role_name in visited:  # Prevent circular inheritance
            return set()
        
        visited.add(role_name)
        permissions = set()
        
        role = self.roles.get(role_name)
        if not role:
            return permissions
        
        # Add direct permissions
        permissions.update(role.permissions)
        
        # Add inherited permissions
        for parent_role in role.inherits_from:
            permissions.update(self._get_role_permissions(parent_role, visited))
        
        return permissions
    
    def check_permission(self, username: str, resource: str, action: str, 
                        context: Dict[str, Any] = None) -> bool:
        """Check if user has permission for specific resource and action"""
        if not self.config.enable_rbac:
            return True  # RBAC disabled, allow all
        
        user_permissions = self.get_user_permissions(username)
        
        for permission in user_permissions:
            if permission.matches(resource, action, context):
                return True
        
        return False
    
    def get_accessible_resources(self, username: str, action: str) -> List[str]:
        """Get list of resources user can access for specific action"""
        accessible = []
        user_permissions = self.get_user_permissions(username)
        
        for permission in user_permissions:
            if permission.action == "*" or permission.action == action:
                accessible.append(permission.resource)
        
        return accessible
    
    def audit_user_access(self, username: str) -> Dict[str, Any]:
        """Generate access audit report for user"""
        user_roles = self.get_user_roles(username)
        user_permissions = self.get_user_permissions(username)
        
        return {
            "username": username,
            "roles": user_roles,
            "permissions_count": len(user_permissions),
            "permissions": [str(p) for p in user_permissions],
            "effective_permissions": self._analyze_effective_permissions(user_permissions),
            "risk_level": self._calculate_risk_level(user_permissions)
        }
    
    def _analyze_effective_permissions(self, permissions: Set[Permission]) -> Dict[str, List[str]]:
        """Analyze effective permissions by resource"""
        effective = {}
        
        for permission in permissions:
            if permission.resource not in effective:
                effective[permission.resource] = []
            effective[permission.resource].append(permission.action)
        
        return effective
    
    def _calculate_risk_level(self, permissions: Set[Permission]) -> str:
        """Calculate risk level based on permissions"""
        high_risk_permissions = ["*:*", "*:delete", "*:admin", "system:*", "user:*"]
        
        for permission in permissions:
            if str(permission) in high_risk_permissions:
                return "HIGH"
        
        if len(permissions) > 20:
            return "MEDIUM"
        
        return "LOW"

# Global access control manager
_access_control_manager = None

def get_access_control_manager() -> AccessControlManager:
    """Get the global access control manager"""
    global _access_control_manager
    if _access_control_manager is None:
        _access_control_manager = AccessControlManager()
    return _access_control_manager

def check_permission(username: str, resource: str, action: str, 
                    context: Dict[str, Any] = None) -> bool:
    """Check permission - convenience function"""
    return get_access_control_manager().check_permission(username, resource, action, context)

def require_permission(resource: str, action: str, context: Dict[str, Any] = None):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get current user
            user_info = st.session_state.get('user')
            if not user_info:
                st.error("Authentication required")
                st.stop()
            
            username = user_info.get('username')
            if not username:
                st.error("Invalid user session")
                st.stop()
            
            # Check permission
            if not check_permission(username, resource, action, context):
                st.error(f"Access denied. Required permission: {resource}:{action}")
                st.stop()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def streamlit_rbac_ui():
    """Streamlit UI for role and permission management"""
    st.title("🔐 Access Control Management")
    
    # Check if user has admin permissions
    user_info = st.session_state.get('user', {})
    username = user_info.get('username', '')
    
    if not check_permission(username, "system", "admin"):
        st.error("Administrator access required")
        return
    
    manager = get_access_control_manager()
    
    # Tabs for different management areas
    tab1, tab2, tab3, tab4 = st.tabs(["Roles", "User Assignments", "Permissions", "Audit"])
    
    with tab1:
        st.header("Role Management")
        
        # List existing roles
        st.subheader("Existing Roles")
        for role_name, role in manager.roles.items():
            with st.expander(f"Role: {role_name}"):
                st.write(f"**Description:** {role.description}")
                st.write(f"**Permissions:** {len(role.permissions)}")
                st.write(f"**Inherits from:** {', '.join(role.inherits_from) if role.inherits_from else 'None'}")
                
                # Show permissions
                if role.permissions:
                    st.write("**Permissions:**")
                    for perm in role.permissions:
                        st.write(f"- {perm}")
        
        # Create new role
        st.subheader("Create New Role")
        with st.form("create_role"):
            new_role_name = st.text_input("Role Name")
            new_role_desc = st.text_area("Description")
            
            # Permission inputs
            st.write("Permissions:")
            perm_resource = st.text_input("Resource (e.g., 'model', 'chat', '*')")
            perm_action = st.text_input("Action (e.g., 'read', 'write', '*')")
            
            submit = st.form_submit_button("Create Role")
            
            if submit and new_role_name and perm_resource and perm_action:
                permissions = [Permission(perm_resource, perm_action)]
                manager.create_role(new_role_name, permissions, description=new_role_desc)
                st.success(f"Role '{new_role_name}' created successfully")
                st.rerun()
    
    with tab2:
        st.header("User Role Assignments")
        
        # Get all users from auth manager
        auth_manager = get_auth_manager()
        users = list(auth_manager.users.keys())
        
        if users:
            selected_user = st.selectbox("Select User", users)
            
            if selected_user:
                current_roles = manager.get_user_roles(selected_user)
                st.write(f"**Current roles for {selected_user}:** {', '.join(current_roles)}")
                
                # Add role
                available_roles = [role for role in manager.roles.keys() if role not in current_roles]
                if available_roles:
                    role_to_add = st.selectbox("Add Role", [""] + available_roles)
                    if st.button("Add Role") and role_to_add:
                        if manager.assign_role(selected_user, role_to_add):
                            st.success(f"Role '{role_to_add}' added to {selected_user}")
                            st.rerun()
                
                # Remove role
                if current_roles:
                    role_to_remove = st.selectbox("Remove Role", [""] + current_roles)
                    if st.button("Remove Role") and role_to_remove:
                        if manager.revoke_role(selected_user, role_to_remove):
                            st.success(f"Role '{role_to_remove}' removed from {selected_user}")
                            st.rerun()
        else:
            st.info("No users found")
    
    with tab3:
        st.header("Permission Details")
        
        # Show all permissions by role
        for role_name, role in manager.roles.items():
            with st.expander(f"Permissions for {role_name}"):
                permissions = manager._get_role_permissions(role_name)
                if permissions:
                    for perm in permissions:
                        st.write(f"- {perm}")
                else:
                    st.write("No permissions")
    
    with tab4:
        st.header("Access Audit")
        
        # User access audit
        if users:
            audit_user = st.selectbox("Audit User", users, key="audit_user")
            
            if audit_user:
                audit_report = manager.audit_user_access(audit_user)
                
                st.subheader(f"Audit Report for {audit_user}")
                st.json(audit_report)
                
                # Risk assessment
                risk_level = audit_report["risk_level"]
                if risk_level == "HIGH":
                    st.error(f"⚠️ HIGH RISK USER: {audit_user}")
                elif risk_level == "MEDIUM":
                    st.warning(f"⚠️ Medium risk user: {audit_user}")
                else:
                    st.success(f"✅ Low risk user: {audit_user}")
        
        # System-wide statistics
        st.subheader("System Statistics")
        total_users = len(users)
        total_roles = len(manager.roles)
        total_assignments = sum(len(roles) for roles in manager.user_roles.values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("Total Roles", total_roles)
        with col3:
            st.metric("Role Assignments", total_assignments)