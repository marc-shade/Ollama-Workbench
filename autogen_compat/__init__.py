"""
Compatibility layer for autogen imports.
This module redirects imports from 'autogen' to 'pyautogen'.
"""

import sys
import importlib
import warnings

# Show a warning about the compatibility layer
warnings.warn(
    "Using autogen_compat layer: 'autogen' package is not available, redirecting to 'pyautogen'",
    ImportWarning
)

# Try to import pyautogen
try:
    import pyautogen
except ImportError:
    raise ImportError(
        "Neither 'autogen' nor 'pyautogen' package is installed. "
        "Please install pyautogen with: pip install pyautogen>=0.2.0"
    )

# Add the module to sys.modules
sys.modules['autogen'] = pyautogen

# Also make submodules available
for submodule_name in [
    'agentchat', 'cache', 'coding', 'oai', 'token_count_utils',
    'browser_utils', 'code_utils', 'exception_utils', 'formatting_utils',
    'function_utils', 'graph_utils', 'retrieve_utils', 'runtime_logging', 'types'
]:
    try:
        submodule = importlib.import_module(f'pyautogen.{submodule_name}')
        sys.modules[f'autogen.{submodule_name}'] = submodule
    except ImportError:
        # If the submodule doesn't exist in pyautogen, just skip it
        pass
