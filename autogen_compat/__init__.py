"""
Compatibility layer for autogen imports.
This module redirects imports from 'autogen' to 'ag2'.
"""

import sys
import importlib
import warnings

# Show a warning about the compatibility layer
warnings.warn(
    "Using autogen_compat layer: 'autogen' package is not available, redirecting to 'ag2'",
    ImportWarning
)

# Try to import ag2
try:
    import ag2
except ImportError:
    raise ImportError(
        "Neither 'autogen' nor 'ag2' package is installed. "
        "Please install ag2 with: pip install ag2>=0.2.0"
    )

# Add the module to sys.modules
sys.modules['autogen'] = ag2

# Also make submodules available
for submodule_name in [
    'agentchat', 'cache', 'coding', 'oai', 'token_count_utils',
    'browser_utils', 'code_utils', 'exception_utils', 'formatting_utils',
    'function_utils', 'graph_utils', 'retrieve_utils', 'runtime_logging', 'types'
]:
    try:
        submodule = importlib.import_module(f'ag2.{submodule_name}')
        sys.modules[f'autogen.{submodule_name}'] = submodule
    except ImportError:
        # If the submodule doesn't exist in ag2, just skip it
        pass
