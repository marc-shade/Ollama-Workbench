# Autogen Dependency Fix

## Problem Description

The project is encountering dependency issues with the `autogen` package:

```
ERROR: Could not find a version that satisfies the requirement autogen==0.2.35
ERROR: No matching distribution found for autogen==0.2.35
```

### Root Causes

1. **Package Name Confusion**: The project is trying to install `autogen==0.2.35`, but this version doesn't exist in PyPI. The correct package name for version 0.2.35 is `ag2`.

2. **Import Namespace Mismatch**: The code imports from the `autogen` namespace:
   ```python
   from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
   ```
   But when installing `ag2`, the imports need to be from the `ag2` namespace.

3. **Strict Version Requirements**: The requirements.txt file specifies exact versions for many packages, which can cause compatibility issues across different systems.

## Solution

This fix addresses these issues by:

1. **Updating requirements.txt**: Using more flexible version specifications and correcting the package name from `autogen` to `ag2`.

2. **Creating a Compatibility Layer**: A Python module that redirects imports from `autogen` to `ag2`, allowing the code to work without modifications.

3. **Providing a Helper Script**: A shell script that sets up the environment correctly to use the compatibility layer.

## Files Included

- **fix_autogen_dependency.sh**: Main script that applies the fixes
- **requirements_fixed.txt**: Updated requirements file with correct package names and more flexible versioning
- **test_autogen_compat.py**: Test script to verify the compatibility layer works
- **use_autogen_compat.sh**: Helper script to run applications with the compatibility layer

## How to Use

### 1. Apply the Fix

Run the fix script to update your dependencies and create the compatibility layer:

```bash
./fix_autogen_dependency.sh
```

This will:
- Back up your original requirements.txt
- Replace it with the fixed version
- Create the compatibility layer
- Install the correct dependencies

### 2. Test the Fix

Verify that the compatibility layer works correctly:

```bash
./use_autogen_compat.sh python test_autogen_compat.py
```

### 3. Run Your Application

Run your application using the compatibility layer:

```bash
./use_autogen_compat.sh streamlit run main.py
```

## How the Compatibility Layer Works

The compatibility layer creates a Python module named `autogen_compat` that:

1. Imports `ag2` when code tries to import `autogen`
2. Adds the imported `ag2` module to `sys.modules['autogen']`
3. Also redirects submodule imports (e.g., `autogen.oai` → `ag2.oai`)

This allows your code to continue using `import autogen` statements without modification.

## Troubleshooting

If you encounter issues:

1. **Verify ag2 is installed**: 
   ```bash
   pip show ag2
   ```

2. **Check PYTHONPATH**: Make sure the current directory is in your PYTHONPATH:
   ```bash
   echo $PYTHONPATH
   ```

3. **Inspect the compatibility layer**: 
   ```bash
   cat autogen_compat/__init__.py
   ```

4. **Run with verbose Python**: 
   ```bash
   ./use_autogen_compat.sh python -v test_autogen_compat.py
   ```

## Long-term Solution

While this compatibility layer provides an immediate fix, consider these long-term solutions:

1. **Update imports**: Gradually update your codebase to import from `ag2` directly
2. **Use Poetry**: Consider using Poetry for dependency management, which handles these issues more gracefully
3. **Loosen version requirements**: Use version ranges (e.g., `>=0.2.0,<0.3.0`) instead of exact versions

## Additional Notes

- The compatibility layer adds a small performance overhead due to the import redirection
- This approach is a common pattern for handling package renames in Python
- The fix is non-invasive and doesn't modify your original code