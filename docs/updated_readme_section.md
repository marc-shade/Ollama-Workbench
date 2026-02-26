## Installation

### Quick Start (Just 2 Steps!)

1. **Setup your environment:**
   ```bash
   ./test_setup.sh
   ```
   This automated script:
   - Finds or installs Python 3.11 (required for compatibility)
   - Creates a properly configured virtual environment
   - Installs all dependencies in the correct order
   - Creates a launcher script

2. **Run Ollama Workbench:**
   ```bash
   ./run_ollama_workbench.sh
   ```
   The launcher automatically ensures everything is set up correctly before starting the application.

### How It Works

The installation process prioritizes compatibility by:

1. **Strictly enforcing Python 3.11** - This specific version provides the best compatibility with all dependencies (especially tiktoken)
2. **Installing dependencies in the correct order** - Some packages need to be installed before others to avoid conflicts
3. **Platform-specific optimizations** - Special handling for Apple Silicon Macs to ensure maximum performance
4. **Fallback implementations** - Automatic workarounds if any dependencies have issues

### Troubleshooting

If you encounter any issues:

- **"Python 3.11 not found"**: The script will attempt to install it automatically, but you may need to install Python 3.11 manually from [python.org](https://www.python.org/downloads/) or using your system's package manager.

- **Environment issues**: If you have problems with the virtual environment, you can always run the setup script again to create a fresh one.

- **Ollama connection issues**: Make sure Ollama is installed and running. The launcher will offer to start Ollama if it's not running.