import os
import json
import glob
import subprocess
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import inspect
import importlib.util
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def find_mcp_directories() -> List[str]:
    """
    Find all potential MCP tool directories on the local system.
    Checks common locations and also searches for directories named 'MCP'.
    
    Returns:
        List[str]: List of paths to MCP directories
    """
    mcp_dirs = []
    
    try:
        # Check known location - fast check first
        known_path = os.path.expanduser("~/Documents/Cline/MCP")
        if os.path.exists(known_path) and os.path.isdir(known_path):
            mcp_dirs.append(known_path)
            logger.info(f"Found MCP directory at known location: {known_path}")
            # Return early with just the known path for performance
            return mcp_dirs
        
        # Get potential search paths - limit scope for performance
        search_paths = []
        # Only check Documents folder instead of entire home directory
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        if os.path.exists(documents_path):
            search_paths.append(documents_path)
        
        # Limit recursive search with a timeout/depth guard
        for search_path in search_paths:
            # Avoid full recursive search - just look 2 levels deep
            for mcp_path in glob.glob(os.path.join(search_path, "**/MCP"), recursive=False):
                if os.path.isdir(mcp_path) and mcp_path not in mcp_dirs:
                    mcp_dirs.append(mcp_path)
                    logger.info(f"Found MCP directory: {mcp_path}")
            
            # Also look one level deeper but avoid full recursion
            for mid_path in glob.glob(os.path.join(search_path, "*/")):
                if os.path.isdir(mid_path):
                    for mcp_path in glob.glob(os.path.join(mid_path, "**/MCP"), recursive=False):
                        if os.path.isdir(mcp_path) and mcp_path not in mcp_dirs:
                            mcp_dirs.append(mcp_path)
                            logger.info(f"Found MCP directory: {mcp_path}")
    except Exception as e:
        # Catch any exceptions to prevent UI from hanging
        logger.error(f"Error searching for MCP directories: {str(e)}")
    
    return mcp_dirs

def discover_mcp_tools() -> Dict[str, Dict[str, Any]]:
    """
    Discover all MCP tools available on the local system.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping tool names to their metadata
    """
    tools = {}
    
    try:
        # Find MCP directories with timeout protection
        mcp_dirs = find_mcp_directories()
        
        # Limit the maximum number of tools to scan to prevent freezing
        max_tools_to_scan = 10
        tools_scanned = 0
        
        for mcp_dir in mcp_dirs:
            # Only scan the top-level directories, not the entire tree
            try:
                # Look for subdirectories, each potentially containing an MCP tool
                for tool_dir in glob.glob(os.path.join(mcp_dir, "*")):
                    # Enforce a maximum to prevent freezing on large directories
                    if tools_scanned >= max_tools_to_scan:
                        logger.warning(f"Reached maximum tool scan limit of {max_tools_to_scan}")
                        break
                        
                    if os.path.isdir(tool_dir):
                        tools_scanned += 1
                        tool_name = os.path.basename(tool_dir)
                        
                        # Check for manifest.json first - fastest approach
                        manifest_path = os.path.join(tool_dir, "manifest.json")
                        if os.path.exists(manifest_path):
                            try:
                                with open(manifest_path, 'r') as f:
                                    manifest = json.load(f)
                                    
                                # Store tool information
                                tools[tool_name] = {
                                    "name": tool_name,
                                    "path": tool_dir,
                                    "manifest": manifest,
                                    "type": "json"
                                }
                                logger.info(f"Found MCP tool with manifest: {tool_name}")
                                continue  # Skip other checks for this tool
                            except Exception as e:
                                logger.error(f"Error reading manifest for {tool_name}: {str(e)}")
                        
                        # Skip remaining checks if we already found a tool
                        if tool_name in tools:
                            continue
                            
                        # Check for Python tool (faster than executable)
                        py_path = os.path.join(tool_dir, "mcp_tool.py")
                        if os.path.exists(py_path):
                            try:
                                # Load the Python module with a timeout guard
                                tool_spec = importlib.util.spec_from_file_location(tool_name, py_path)
                                if tool_spec and tool_spec.loader:
                                    # TODO: In a production system, we would use multiprocessing
                                    # to properly timeout module loading, but we'll simplify here
                                    tool_module = importlib.util.module_from_spec(tool_spec)
                                    sys.modules[tool_name] = tool_module
                                    tool_spec.loader.exec_module(tool_module)
                                    
                                    # Check if module has a tool function
                                    if hasattr(tool_module, "mcp_tool_info"):
                                        tool_info = tool_module.mcp_tool_info()
                                        
                                        # Store tool information
                                        tools[tool_name] = {
                                            "name": tool_name,
                                            "path": tool_dir,
                                            "module": tool_module,
                                            "info": tool_info,
                                            "type": "python"
                                        }
                                        logger.info(f"Found MCP Python tool: {tool_name}")
                                        continue  # Skip other checks for this tool
                            except Exception as e:
                                logger.error(f"Error loading Python tool {tool_name}: {str(e)}")
                        
                        # Skip remaining checks if we already found a tool
                        if tool_name in tools:
                            continue
                            
                        # Finally check for executable scripts but with stricter timeout
                        for script_ext in ["py", "sh"]:  # Reduced list for efficiency
                            script_path = os.path.join(tool_dir, f"mcp_tool.{script_ext}")
                            if os.path.exists(script_path) and os.access(script_path, os.X_OK):
                                try:
                                    # Try to get tool info with a stricter timeout
                                    result = subprocess.run(
                                        [script_path, "--info"], 
                                        capture_output=True, 
                                        text=True,
                                        timeout=2  # Shorter timeout for better UI responsiveness
                                    )
                                    
                                    if result.returncode == 0 and result.stdout:
                                        try:
                                            tool_info = json.loads(result.stdout)
                                            
                                            # Store tool information
                                            tools[tool_name] = {
                                                "name": tool_name,
                                                "path": script_path,
                                                "info": tool_info,
                                                "type": "executable"
                                            }
                                            logger.info(f"Found MCP executable tool: {tool_name}")
                                            break  # Found a tool, break the extension loop
                                        except json.JSONDecodeError:
                                            logger.error(f"Invalid JSON info from {script_path}")
                                except subprocess.TimeoutExpired:
                                    logger.warning(f"Timeout getting info from {script_path}")
                                except Exception as e:
                                    logger.error(f"Error with executable tool {tool_name}: {str(e)}")
            except Exception as dir_error:
                logger.error(f"Error processing MCP directory {mcp_dir}: {str(dir_error)}")
                continue  # Continue with next directory
                
    except Exception as e:
        # Catch-all protection for UI
        logger.error(f"Unexpected error in MCP tool discovery: {str(e)}")
    
    return tools

def get_mcp_tool_schema(tools: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate OpenAI-compatible function schemas for MCP tools.
    
    Args:
        tools: Dictionary of discovered MCP tools
        
    Returns:
        List[Dict[str, Any]]: List of tool schemas for use with LLMs
    """
    tool_schemas = []
    
    for tool_name, tool_info in tools.items():
        schema = None
        
        if tool_info["type"] == "json" and "manifest" in tool_info:
            manifest = tool_info["manifest"]
            if "schema" in manifest:
                schema = {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{tool_name}",
                        "description": manifest.get("description", f"MCP tool: {tool_name}"),
                        "parameters": manifest["schema"]
                    }
                }
        
        elif tool_info["type"] == "python" and "info" in tool_info:
            info = tool_info["info"]
            if "schema" in info:
                schema = {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{tool_name}",
                        "description": info.get("description", f"MCP tool: {tool_name}"),
                        "parameters": info["schema"]
                    }
                }
        
        elif tool_info["type"] == "executable" and "info" in tool_info:
            info = tool_info["info"]
            if "schema" in info:
                schema = {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{tool_name}",
                        "description": info.get("description", f"MCP tool: {tool_name}"),
                        "parameters": info["schema"]
                    }
                }
        
        if schema:
            tool_schemas.append(schema)
    
    return tool_schemas

def execute_mcp_tool(tool_name: str, tools: Dict[str, Dict[str, Any]], args: Dict[str, Any]) -> Any:
    """
    Execute an MCP tool with the given arguments.
    
    Args:
        tool_name: Name of the tool to execute (with "mcp__" prefix removed)
        tools: Dictionary of discovered MCP tools
        args: Arguments to pass to the tool
        
    Returns:
        Any: Result of the tool execution
    """
    # Remove "mcp__" prefix if present
    if tool_name.startswith("mcp__"):
        tool_name = tool_name[5:]
    
    if tool_name not in tools:
        return {"error": f"MCP tool '{tool_name}' not found"}
    
    tool_info = tools[tool_name]
    
    try:
        if tool_info["type"] == "python" and "module" in tool_info:
            module = tool_info["module"]
            if hasattr(module, "execute"):
                return module.execute(args)
            else:
                return {"error": f"Python module for '{tool_name}' does not have execute() function"}
        
        elif tool_info["type"] == "executable":
            script_path = tool_info["path"]
            
            # Convert arguments to JSON and pass to the script
            args_json = json.dumps(args)
            
            result = subprocess.run(
                [script_path, "--execute", args_json],
                capture_output=True,
                text=True,
                timeout=30  # Timeout after 30 seconds
            )
            
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {"result": result.stdout.strip()}
            else:
                return {"error": f"Tool execution failed: {result.stderr}"}
        
        elif tool_info["type"] == "json" and "manifest" in tool_info:
            # Check if there's an executable file to run
            manifest = tool_info["manifest"]
            if "executable" in manifest:
                exe_path = os.path.join(tool_info["path"], manifest["executable"])
                if os.path.exists(exe_path) and os.access(exe_path, os.X_OK):
                    args_json = json.dumps(args)
                    
                    result = subprocess.run(
                        [exe_path, args_json],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        try:
                            return json.loads(result.stdout)
                        except json.JSONDecodeError:
                            return {"result": result.stdout.strip()}
                    else:
                        return {"error": f"Tool execution failed: {result.stderr}"}
            
            return {"error": f"No executable found for JSON tool '{tool_name}'"}
        
        else:
            return {"error": f"Unknown tool type for '{tool_name}'"}
    
    except Exception as e:
        logger.error(f"Error executing MCP tool {tool_name}: {str(e)}")
        return {"error": f"Error executing tool: {str(e)}"}

def get_available_mcp_tools() -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Get available MCP tools and their schemas.
    
    Returns:
        Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]: 
            - Dictionary of tools
            - List of tool schemas
    """
    try:
        # Add timeout protection to prevent UI freezing
        import threading
        import time
        
        # Create a result container for the threaded operation
        result = {'tools': {}, 'schemas': [], 'completed': False}
        
        # Function to run in a separate thread
        def discover_with_timeout():
            try:
                result['tools'] = discover_mcp_tools()
                result['schemas'] = get_mcp_tool_schema(result['tools'])
                result['completed'] = True
            except Exception as e:
                logger.error(f"Error in threaded MCP tool discovery: {str(e)}")
                result['completed'] = True
        
        # Run the discovery in a separate thread
        discovery_thread = threading.Thread(target=discover_with_timeout)
        discovery_thread.daemon = True  # Allow the thread to be terminated when the main thread exits
        discovery_thread.start()
        
        # Wait for the thread to complete with a timeout
        max_wait_time = 3.0  # Maximum seconds to wait
        start_time = time.time()
        
        while not result['completed'] and (time.time() - start_time) < max_wait_time:
            time.sleep(0.1)  # Short sleep to prevent CPU spinning
        
        if not result['completed']:
            logger.warning(f"MCP tool discovery timed out after {max_wait_time}s")
            return {}, []  # Return empty results if timed out
            
        return result['tools'], result['schemas']
    
    except Exception as e:
        # Final fallback
        logger.error(f"Error getting MCP tools: {str(e)}")
        return {}, []