import sys
import subprocess
import importlib.util

print("Python version:", sys.version)
print("\n=== Testing PyPDF2 import ===")

# Try to import PyPDF2 (uppercase)
try:
    import PyPDF2
    print("✅ Successfully imported PyPDF2 (uppercase)")
    print("PyPDF2 version:", getattr(PyPDF2, "__version__", "unknown"))
    print("PyPDF2 path:", PyPDF2.__file__)
except ImportError as e:
    print("❌ Failed to import PyPDF2 (uppercase):", e)

# Try to import pypdf2 (lowercase)
try:
    import pypdf2
    print("✅ Successfully imported pypdf2 (lowercase)")
    print("pypdf2 version:", getattr(pypdf2, "__version__", "unknown"))
    print("pypdf2 path:", pypdf2.__file__)
except ImportError as e:
    print("❌ Failed to import pypdf2 (lowercase):", e)

# Check what's installed with pip
print("\n=== Checking pip installation ===")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "PyPDF2"],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        print("✅ PyPDF2 (uppercase) is installed via pip:")
        for line in result.stdout.splitlines():
            if line.startswith("Name:") or line.startswith("Version:") or line.startswith("Location:"):
                print("  ", line)
    else:
        print("❌ PyPDF2 (uppercase) is NOT installed via pip")
except Exception as e:
    print("Error checking PyPDF2 installation:", e)

try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "pypdf2"],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        print("✅ pypdf2 (lowercase) is installed via pip:")
        for line in result.stdout.splitlines():
            if line.startswith("Name:") or line.startswith("Version:") or line.startswith("Location:"):
                print("  ", line)
    else:
        print("❌ pypdf2 (lowercase) is NOT installed via pip")
except Exception as e:
    print("Error checking pypdf2 installation:", e)