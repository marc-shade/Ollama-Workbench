import os
import json
import yaml
import dotenv
import logging
from typing import Dict, Any, Optional
# from pathlib import Path  # Not currently used

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    # Ollama configuration
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_API_KEY": None,  # Ollama currently doesn't require an API key
    
    # External providers configuration
    "OPENAI_API_KEY": None,
    "ANTHROPIC_API_KEY": None,
    "GROQ_API_KEY": None,
    "GOOGLE_API_KEY": None,
    "MISTRAL_API_KEY": None,
    "CLAUDE_API_KEY": None,
    "AZURE_OPENAI_API_KEY": None,
    "AZURE_OPENAI_ENDPOINT": None,
    
    # Application configuration
    "WORKBENCH_HOST": "localhost",
    "WORKBENCH_PORT": 8501,
    "WORKBENCH_DEBUG": False,
    "WORKBENCH_LOG_LEVEL": "INFO",
    
    # Enhanced Security configuration
    "ENABLE_ENHANCED_SECURITY": True,
    "ENABLE_AUTH": True,  # Enable authentication by default
    "AUTH_USERNAME": None,
    "AUTH_PASSWORD": None,
    "ENABLE_RBAC": True,
    "ENABLE_AUDIT_LOGGING": True,
    "ENABLE_ENCRYPTION": True,
    
    # Features configuration
    "ENABLE_TOOL_CALLING": True,
    "ENABLE_MULTIMODAL": True,
    "ENABLE_STRUCTURED_OUTPUT": True,
    "ENABLE_MCP_TOOLS": True,
    "ENABLE_RAG": True,
    
    # Resource limits
    "MAX_UPLOAD_SIZE_MB": 50,
    "MAX_TOKENS": 32000,
    "DEFAULT_TEMPERATURE": 0.7,
    
    # Paths configuration
    "DATA_DIR": "data",
    "UPLOAD_DIR": "uploads",
    "MODELS_DIR": "models",
    "PROJECTS_DIR": "projects",
    "CACHE_DIR": "cache",
    "SECURITY_DIR": "security",
    "AUDIT_DIR": "security/audit",
    "LOGS_DIR": "logs",
    
    # Observability configuration
    "ENABLE_OBSERVABILITY": True,
    "OPIK_PROJECT_NAME": "ollama-workbench",
    "OPIK_API_KEY": None,
    "OPIK_URL": None,
    "OPIK_WORKSPACE": None,
    
    # Extension configuration
    "EXTENSION_ID": "gddghhhklfnhijhhagfgnfiehidcdnba"
}

# Config file paths to check (in order of priority)
CONFIG_PATHS = [
    os.path.join(os.getcwd(), "workbench_config.json"),
    os.path.join(os.getcwd(), "workbench_config.yaml"),
    os.path.join(os.getcwd(), "workbench_config.yml"),
    os.path.expanduser("~/.config/ollama-workbench/config.json"),
    os.path.expanduser("~/.config/ollama-workbench/config.yaml"),
    os.path.expanduser("~/.config/ollama-workbench/config.yml"),
]

# Current config (singleton) - declared here so get_config() can use it
_config = None

def load_config_file() -> Dict[str, Any]:
    """
    Load configuration from a config file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    for config_path in CONFIG_PATHS:
        if os.path.exists(config_path):
            try:
                # Determine file type by extension
                ext = os.path.splitext(config_path)[1].lower()
                
                with open(config_path, "r") as f:
                    if ext == ".json":
                        config = json.load(f)
                    elif ext in [".yaml", ".yml"]:
                        config = yaml.safe_load(f)
                    else:
                        continue
                    
                    logger.info(f"Loaded configuration from {config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
    
    return {}

def get_config() -> Dict[str, Any]:
    """
    Get the application configuration, combining defaults, config files,
    and environment variables. Uses the module-level _config singleton
    after first load; call refresh_config() to force a re-read.

    Returns:
        Dict[str, Any]: Complete configuration dictionary
    """
    global _config
    if _config is not None:
        return _config

    # Start with default config
    config = DEFAULT_CONFIG.copy()

    # Update with config file if exists
    file_config = load_config_file()
    config.update(file_config)
    
    # Update with environment variables (highest priority)
    for key in config:
        env_val = os.environ.get(key)
        if env_val is not None:
            # Convert string environment variables to appropriate types
            if isinstance(config[key], bool):
                # Handle boolean values
                config[key] = env_val.lower() in ["true", "1", "yes", "y", "t"]
            elif isinstance(config[key], int):
                try:
                    config[key] = int(env_val)
                except ValueError:
                    logger.warning(f"Could not convert {key}={env_val} to int, using default")
            elif isinstance(config[key], float):
                try:
                    config[key] = float(env_val)
                except ValueError:
                    logger.warning(f"Could not convert {key}={env_val} to float, using default")
            else:
                config[key] = env_val
    
    # Ensure required directories exist
    for dir_key in ["DATA_DIR", "UPLOAD_DIR", "MODELS_DIR", "PROJECTS_DIR", "CACHE_DIR", "SECURITY_DIR", "AUDIT_DIR", "LOGS_DIR"]:
        if config[dir_key]:
            os.makedirs(config[dir_key], exist_ok=True)
    
    # Initialize security framework if enabled
    if config.get("ENABLE_ENHANCED_SECURITY"):
        try:
            from security import get_security_config_manager, get_audit_logger
            # Initialize security components
            get_security_config_manager()  # Initialize security manager
            get_audit_logger()  # Initialize audit logger
            logger.info("Enhanced security framework initialized")
        except ImportError as e:
            logger.warning(f"Enhanced security framework not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize security framework: {e}")
    
    _config = config
    return config

def save_config(config: Dict[str, Any], path: Optional[str] = None) -> bool:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary to save
        path: Path to save the config to (defaults to workbench_config.json)
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    if path is None:
        path = CONFIG_PATHS[0]  # Use the first path as default
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Determine file type by extension
        ext = os.path.splitext(path)[1].lower()
        
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f, indent=2)
            elif ext in [".yaml", ".yml"]:
                yaml.dump(config, f)
            else:
                # Default to JSON if extension is not recognized
                json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config to {path}: {e}")
        return False

def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of configuration updates
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    config = get_config()
    config.update(updates)
    
    # Save the updated config
    save_config(config)
    
    return config

def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider with enhanced security.
    
    Args:
        provider: Name of the provider (e.g., "openai", "anthropic")
        
    Returns:
        Optional[str]: API key or None if not found
    """
    config = get_config()
    key_name = f"{provider.upper()}_API_KEY"
    encrypted_key_name = f"{key_name}_ENCRYPTED"
    
    # Try to get encrypted key first
    if config.get("ENABLE_ENCRYPTION") and config.get(encrypted_key_name):
        try:
            from security.encryption import decrypt_data
            return decrypt_data(config[encrypted_key_name])
        except ImportError:
            logger.warning(f"Cannot decrypt API key for {provider} (encryption not available)")
        except Exception as e:
            logger.error(f"Failed to decrypt API key for {provider}: {e}")
    
    # Fallback to plain key
    return config.get(key_name)

def set_api_key(provider: str, api_key: str) -> None:
    """
    Set API key for a specific provider with enhanced security.
    
    Args:
        provider: Name of the provider (e.g., "openai", "anthropic")
        api_key: API key to set
    """
    key_name = f"{provider.upper()}_API_KEY"
    
    # Encrypt API key if encryption is enabled
    config = get_config()
    if config.get("ENABLE_ENCRYPTION"):
        try:
            from security.encryption import encrypt_data
            encrypted_key = encrypt_data(api_key)
            update_config({f"{key_name}_ENCRYPTED": encrypted_key})
            logger.info(f"API key for {provider} encrypted and stored")
        except ImportError:
            # Fallback to plain storage if encryption not available
            update_config({key_name: api_key})
            logger.warning(f"API key for {provider} stored without encryption (encryption not available)")
    else:
        update_config({key_name: api_key})
    
    # Also set the environment variable for immediate use
    os.environ[key_name] = api_key
    
    # Log security event if audit logging is enabled
    if config.get("ENABLE_AUDIT_LOGGING"):
        try:
            from security.audit_logging import log_security_event, EventType, EventSeverity
            log_security_event(
                EventType.CONFIGURATION_CHANGE,
                EventSeverity.MEDIUM,
                "system",
                "api_key",
                "update",
                "success",
                details={"provider": provider, "encrypted": config.get("ENABLE_ENCRYPTION", False)}
            )
        except ImportError:
            pass

def server_config_ui():
    """
    Streamlit UI for server configuration.
    """
    import streamlit as st
    
    st.title("⚙️ Server Configuration")
    st.write("Configure the Ollama Workbench server settings")
    
    # Get current configuration
    config = get_config()
    
    # Create tabs for different configuration sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ollama Settings", 
        "API Keys", 
        "Application Settings", 
        "Security", 
        "Advanced"
    ])
    
    with tab1:
        st.header("Ollama Settings")
        
        ollama_host = st.text_input(
            "Ollama Host",
            value=config["OLLAMA_HOST"],
            help="The URL of your Ollama server"
        )
        
        # Update config if changed
        if ollama_host != config["OLLAMA_HOST"]:
            update_config({"OLLAMA_HOST": ollama_host})
            st.success(f"Ollama host updated to {ollama_host}")
    
    with tab2:
        st.header("API Keys")
        st.write("Configure API keys for external providers")
        
        # Collect all API key configurations
        api_providers = [
            ("OpenAI", "OPENAI_API_KEY"),
            ("Anthropic", "ANTHROPIC_API_KEY"),
            ("Groq", "GROQ_API_KEY"),
            ("Google", "GOOGLE_API_KEY"),
            ("Mistral", "MISTRAL_API_KEY"),
            ("Claude", "CLAUDE_API_KEY"),
            ("Azure OpenAI", "AZURE_OPENAI_API_KEY")
        ]
        
        # Create a form for API keys
        with st.form("api_keys_form"):
            api_keys_updates = {}
            
            for display_name, key_name in api_providers:
                current_value = config.get(key_name) or ""
                # Use password input for API keys
                api_key = st.text_input(
                    f"{display_name} API Key",
                    value=current_value,
                    type="password",
                    help=f"API key for {display_name}",
                    key=f"input_{key_name}"
                )
                
                # Only update if the value has changed and is not empty
                if api_key != current_value and api_key:
                    api_keys_updates[key_name] = api_key
            
            # Submit button
            submit = st.form_submit_button("Save API Keys")
            
            if submit and api_keys_updates:
                update_config(api_keys_updates)
                st.success("API keys updated successfully")
    
    with tab3:
        st.header("Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            workbench_host = st.text_input(
                "Workbench Host",
                value=config["WORKBENCH_HOST"],
                help="The host address to bind the Workbench server to"
            )
            
            workbench_debug = st.checkbox(
                "Debug Mode",
                value=config["WORKBENCH_DEBUG"],
                help="Enable debug mode for more verbose logging"
            )
        
        with col2:
            workbench_port = st.number_input(
                "Workbench Port",
                value=config["WORKBENCH_PORT"],
                min_value=1,
                max_value=65535,
                help="The port to run the Workbench server on"
            )
            
            log_level = st.selectbox(
                "Log Level",
                options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(config["WORKBENCH_LOG_LEVEL"]),
                help="Set the logging level"
            )
        
        # Feature toggles
        st.subheader("Feature Toggles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_tool_calling = st.checkbox(
                "Enable Tool Calling",
                value=config["ENABLE_TOOL_CALLING"],
                help="Enable function calling and tool integration"
            )
            
            enable_structured_output = st.checkbox(
                "Enable Structured Output",
                value=config["ENABLE_STRUCTURED_OUTPUT"],
                help="Enable JSON schema structured output generation"
            )
            
            enable_rag = st.checkbox(
                "Enable RAG",
                value=config["ENABLE_RAG"],
                help="Enable Retrieval Augmented Generation features"
            )
        
        with col2:
            enable_multimodal = st.checkbox(
                "Enable Multimodal",
                value=config["ENABLE_MULTIMODAL"],
                help="Enable multimodal (image + text) features"
            )
            
            enable_mcp_tools = st.checkbox(
                "Enable MCP Tools",
                value=config["ENABLE_MCP_TOOLS"],
                help="Enable Model Control Protocol tools integration"
            )
        
        # Collect updates
        app_settings_updates = {
            "WORKBENCH_HOST": workbench_host,
            "WORKBENCH_PORT": int(workbench_port),
            "WORKBENCH_DEBUG": workbench_debug,
            "WORKBENCH_LOG_LEVEL": log_level,
            "ENABLE_TOOL_CALLING": enable_tool_calling,
            "ENABLE_MULTIMODAL": enable_multimodal,
            "ENABLE_STRUCTURED_OUTPUT": enable_structured_output,
            "ENABLE_MCP_TOOLS": enable_mcp_tools,
            "ENABLE_RAG": enable_rag
        }
        
        if st.button("Save Application Settings"):
            update_config(app_settings_updates)
            st.success("Application settings updated successfully")
    
    with tab4:
        st.header("Security Settings")
        
        # Enhanced Security Framework
        enable_enhanced_security = st.checkbox(
            "Enable Enhanced Security Framework",
            value=config["ENABLE_ENHANCED_SECURITY"],
            help="Enable comprehensive security features including RBAC, audit logging, and encryption"
        )
        
        enable_auth = st.checkbox(
            "Enable Authentication",
            value=config["ENABLE_AUTH"],
            help="Enable authentication for the Workbench"
        )
        
        if enable_enhanced_security:
            st.subheader("Advanced Security Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                enable_rbac = st.checkbox(
                    "Enable Role-Based Access Control",
                    value=config.get("ENABLE_RBAC", True),
                    help="Enable role-based access control system"
                )
                
                enable_audit = st.checkbox(
                    "Enable Audit Logging",
                    value=config.get("ENABLE_AUDIT_LOGGING", True),
                    help="Enable comprehensive audit logging"
                )
            
            with col2:
                enable_encryption = st.checkbox(
                    "Enable Data Encryption",
                    value=config.get("ENABLE_ENCRYPTION", True),
                    help="Enable data encryption at rest and in transit"
                )
                
                enable_observability = st.checkbox(
                    "Enable Observability",
                    value=config.get("ENABLE_OBSERVABILITY", True),
                    help="Enable comprehensive monitoring and tracing"
                )
            
            if st.button("Configure Advanced Security"):
                try:
                    # Import and display security management UI
                    # Security management UIs available
                    # from security.access_control import streamlit_rbac_ui
                    # from security.authentication import streamlit_auth_ui
                    
                    st.write("**Advanced Security Configuration**")
                    st.info("Advanced security features are available. Use the dedicated security management pages.")
                except ImportError:
                    st.error("Enhanced security framework not available")
        
        if enable_auth:
            auth_username = st.text_input(
                "Username",
                value=config.get("AUTH_USERNAME", ""),
                help="Authentication username"
            )
            
            auth_password = st.text_input(
                "Password",
                value=config.get("AUTH_PASSWORD", ""),
                type="password",
                help="Authentication password"
            )
            
            security_updates = {
                "ENABLE_ENHANCED_SECURITY": enable_enhanced_security,
                "ENABLE_AUTH": enable_auth,
                "AUTH_USERNAME": auth_username,
                "AUTH_PASSWORD": auth_password,
                "ENABLE_RBAC": enable_rbac if enable_enhanced_security else False,
                "ENABLE_AUDIT_LOGGING": enable_audit if enable_enhanced_security else False,
                "ENABLE_ENCRYPTION": enable_encryption if enable_enhanced_security else False,
                "ENABLE_OBSERVABILITY": enable_observability if enable_enhanced_security else False
            }
            
            if st.button("Save Security Settings"):
                update_config(security_updates)
                st.success("Security settings updated successfully")
                
                # Initialize security framework if enabled
                if enable_enhanced_security:
                    try:
                        from security import get_security_config_manager
                        get_security_config_manager()  # Initialize security
                        st.success("Enhanced security framework initialized")
                    except Exception as e:
                        st.error(f"Failed to initialize security framework: {e}")
        else:
            if st.button("Save Security Settings"):
                update_config({
                    "ENABLE_ENHANCED_SECURITY": enable_enhanced_security,
                    "ENABLE_AUTH": False
                })
                st.success("Authentication disabled")
    
    with tab5:
        st.header("Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_upload_size = st.number_input(
                "Max Upload Size (MB)",
                value=config["MAX_UPLOAD_SIZE_MB"],
                min_value=1,
                help="Maximum file upload size in megabytes"
            )
            
            max_tokens = st.number_input(
                "Max Tokens",
                value=config["MAX_TOKENS"],
                min_value=100,
                help="Maximum number of tokens for model outputs"
            )
        
        with col2:
            default_temperature = st.slider(
                "Default Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(config["DEFAULT_TEMPERATURE"]),
                step=0.1,
                help="Default temperature for model outputs"
            )
            
            extension_id = st.text_input(
                "Extension ID",
                value=config["EXTENSION_ID"],
                help="ID of the browser extension for Ollama Workbench"
            )
        
        # Directory settings
        st.subheader("Directory Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_dir = st.text_input(
                "Data Directory",
                value=config["DATA_DIR"],
                help="Directory to store application data"
            )
            
            uploads_dir = st.text_input(
                "Uploads Directory",
                value=config["UPLOAD_DIR"],
                help="Directory to store uploaded files"
            )
            
            models_dir = st.text_input(
                "Models Directory",
                value=config["MODELS_DIR"],
                help="Directory to store model files"
            )
        
        with col2:
            projects_dir = st.text_input(
                "Projects Directory",
                value=config["PROJECTS_DIR"],
                help="Directory to store project files"
            )
            
            cache_dir = st.text_input(
                "Cache Directory",
                value=config["CACHE_DIR"],
                help="Directory to store cache files"
            )
        
        # Collect advanced updates
        advanced_updates = {
            "MAX_UPLOAD_SIZE_MB": int(max_upload_size),
            "MAX_TOKENS": int(max_tokens),
            "DEFAULT_TEMPERATURE": float(default_temperature),
            "EXTENSION_ID": extension_id,
            "DATA_DIR": data_dir,
            "UPLOAD_DIR": uploads_dir,
            "MODELS_DIR": models_dir,
            "PROJECTS_DIR": projects_dir,
            "CACHE_DIR": cache_dir
        }
        
        if st.button("Save Advanced Settings"):
            update_config(advanced_updates)
            st.success("Advanced settings updated successfully")
    
    # Export/Import configuration
    st.header("Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Configuration"):
            config_json = json.dumps(config, indent=2)
            st.download_button(
                "Download Configuration",
                data=config_json,
                file_name="workbench_config.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import Configuration", type=["json", "yaml", "yml"])
        
        if uploaded_file is not None:
            try:
                # Determine file type by extension
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                
                if ext == ".json":
                    imported_config = json.load(uploaded_file)
                elif ext in [".yaml", ".yml"]:
                    imported_config = yaml.safe_load(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    imported_config = None
                
                if imported_config:
                    if st.button("Apply Imported Configuration"):
                        update_config(imported_config)
                        st.success("Configuration imported and applied successfully")
            except Exception as e:
                st.error(f"Error importing configuration: {e}")

def get_current_config() -> Dict[str, Any]:
    """
    Get the current configuration (singleton pattern).
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    global _config
    if _config is None:
        _config = get_config()
    return _config

def refresh_config() -> Dict[str, Any]:
    """
    Refresh the configuration from sources.
    
    Returns:
        Dict[str, Any]: Updated configuration
    """
    global _config
    _config = get_config()
    
    # Reinitialize security framework if enabled
    if _config.get("ENABLE_ENHANCED_SECURITY"):
        try:
            from security import get_security_config_manager
            get_security_config_manager()  # Reinitialize security
            logger.info("Security framework reinitialized after config refresh")
        except Exception as e:
            logger.warning(f"Failed to reinitialize security framework: {e}")
    
    return _config

# Expose the current config at the module level for easy imports
CONFIG = get_current_config()

if __name__ == "__main__":
    # Simple test function for command line usage
    print("Current configuration:")
    for key, value in get_current_config().items():
        print(f"{key}: {value}")