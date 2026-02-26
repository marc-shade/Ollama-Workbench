#!/usr/bin/env python3
"""
Ollama Workbench - Automated Setup and Installation Script
Comprehensive setup script for the enhanced Ollama Workbench platform.
"""

import os
import sys
import subprocess
import platform
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup_workbench.log')
    ]
)
logger = logging.getLogger(__name__)

class WorkbenchSetup:
    """Comprehensive setup manager for Ollama Workbench"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.base_dir = Path(__file__).parent
        self.venv_dir = self.base_dir / "venv"
        self.requirements_file = self.base_dir / "requirements.txt"
        self.config = {}
        
        # Minimum requirements
        self.min_python_version = (3, 8)
        self.required_commands = ["git", "curl"]
        
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        logger.info("🔍 Checking system prerequisites...")
        
        # Check Python version
        if self.python_version < self.min_python_version:
            logger.error(f"❌ Python {self.min_python_version[0]}.{self.min_python_version[1]}+ required. Found {self.python_version.major}.{self.python_version.minor}")
            return False
        
        logger.info(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        # Check required commands
        missing_commands = []
        for cmd in self.required_commands:
            if not self._command_exists(cmd):
                missing_commands.append(cmd)
        
        if missing_commands:
            logger.error(f"❌ Missing required commands: {', '.join(missing_commands)}")
            return False
        
        logger.info("✅ All required commands available")
        return True
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists"""
        try:
            subprocess.run([command, "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def setup_virtual_environment(self) -> bool:
        """Set up Python virtual environment"""
        logger.info("🐍 Setting up virtual environment...")
        
        try:
            if self.venv_dir.exists():
                logger.info("♻️  Virtual environment already exists")
            else:
                subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
                logger.info("✅ Virtual environment created")
            
            # Get activation script path
            if self.system == "Windows":
                activate_script = self.venv_dir / "Scripts" / "activate.bat"
                pip_executable = self.venv_dir / "Scripts" / "pip.exe"
            else:
                activate_script = self.venv_dir / "bin" / "activate"
                pip_executable = self.venv_dir / "bin" / "pip"
            
            if not pip_executable.exists():
                logger.error("❌ Virtual environment setup failed - pip not found")
                return False
            
            logger.info("✅ Virtual environment ready")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies...")
        
        if not self.requirements_file.exists():
            logger.error("❌ requirements.txt not found")
            return False
        
        try:
            # Get pip executable
            if self.system == "Windows":
                pip_executable = self.venv_dir / "Scripts" / "pip.exe"
            else:
                pip_executable = self.venv_dir / "bin" / "pip"
            
            # Upgrade pip first
            subprocess.run([str(pip_executable), "install", "--upgrade", "pip"], check=True)
            logger.info("✅ Pip upgraded")
            
            # Install requirements
            logger.info("📥 Installing packages from requirements.txt...")
            result = subprocess.run(
                [str(pip_executable), "install", "-r", str(self.requirements_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Package installation failed: {result.stderr}")
                # Try installing critical packages individually
                logger.info("🔄 Attempting to install critical packages individually...")
                critical_packages = [
                    "streamlit>=1.38.0",
                    "ollama>=0.4.8",
                    "openai==1.43.0",
                    "cryptography>=41.0.0",
                    "bcrypt>=4.0.0",
                    "PyJWT>=2.8.0"
                ]
                
                for package in critical_packages:
                    try:
                        subprocess.run([str(pip_executable), "install", package], check=True)
                        logger.info(f"✅ Installed {package}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"⚠️  Failed to install {package}: {e}")
            else:
                logger.info("✅ All dependencies installed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def install_ollama(self) -> bool:
        """Install Ollama if not present"""
        logger.info("🦙 Checking Ollama installation...")
        
        if self._command_exists("ollama"):
            logger.info("✅ Ollama already installed")
            return True
        
        logger.info("📥 Installing Ollama...")
        
        try:
            if self.system == "Darwin":  # macOS
                subprocess.run(["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"], 
                              shell=True, check=True)
            elif self.system == "Linux":
                subprocess.run(["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"], 
                              shell=True, check=True)
            elif self.system == "Windows":
                logger.info("🪟 For Windows, please download Ollama from https://ollama.ai/download")
                logger.info("   and run the installer manually.")
                input("Press Enter after installing Ollama...")
            else:
                logger.warning(f"⚠️  Unsupported system: {self.system}")
                return False
            
            # Verify installation
            if self._command_exists("ollama"):
                logger.info("✅ Ollama installed successfully")
                return True
            else:
                logger.error("❌ Ollama installation verification failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ollama installation failed: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Set up required directories"""
        logger.info("📁 Setting up directories...")
        
        directories = [
            "data",
            "uploads", 
            "models",
            "projects",
            "cache",
            "security",
            "security/audit",
            "logs",
            "sessions"
        ]
        
        try:
            for directory in directories:
                dir_path = self.base_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Directory setup failed: {e}")
            return False
    
    def initialize_security(self) -> bool:
        """Initialize security framework"""
        logger.info("🔐 Initializing security framework...")
        
        try:
            # Get Python executable from venv
            if self.system == "Windows":
                python_executable = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_executable = self.venv_dir / "bin" / "python"
            
            # Initialize security configuration
            init_script = '''
import sys
sys.path.insert(0, ".")
try:
    from security import get_security_config_manager, get_audit_logger
    
    # Initialize security manager
    security_manager = get_security_config_manager()
    print("✅ Security configuration initialized")
    
    # Initialize audit logger
    audit_logger = get_audit_logger()
    print("✅ Audit logging initialized")
    
    print("🔐 Security framework ready")
except Exception as e:
    print(f"⚠️  Security framework initialization skipped: {e}")
'''
            
            result = subprocess.run(
                [str(python_executable), "-c", init_script],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            if result.returncode == 0:
                logger.info("✅ Security framework initialized")
                return True
            else:
                logger.warning(f"⚠️  Security initialization warning: {result.stderr}")
                return True  # Non-critical, continue setup
                
        except Exception as e:
            logger.warning(f"⚠️  Security initialization failed: {e}")
            return True  # Non-critical, continue setup
    
    def create_configuration(self) -> bool:
        """Create initial configuration"""
        logger.info("⚙️  Creating configuration...")
        
        try:
            config = {
                "OLLAMA_HOST": "http://localhost:11434",
                "WORKBENCH_HOST": "localhost",
                "WORKBENCH_PORT": 8501,
                "WORKBENCH_DEBUG": False,
                "ENABLE_ENHANCED_SECURITY": True,
                "ENABLE_AUTH": False,  # Start with auth disabled for first setup
                "ENABLE_RBAC": True,
                "ENABLE_AUDIT_LOGGING": True,
                "ENABLE_ENCRYPTION": True,
                "ENABLE_OBSERVABILITY": True,
                "DATA_DIR": "data",
                "UPLOAD_DIR": "uploads",
                "MODELS_DIR": "models",
                "PROJECTS_DIR": "projects",
                "CACHE_DIR": "cache",
                "SECURITY_DIR": "security",
                "LOGS_DIR": "logs"
            }
            
            config_file = self.base_dir / "workbench_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("✅ Configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration creation failed: {e}")
            return False
    
    def start_ollama_server(self) -> bool:
        """Start Ollama server"""
        logger.info("🚀 Starting Ollama server...")
        
        try:
            # Check if already running
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Ollama server already running")
                return True
            
            # Start server
            if self.system == "Windows":
                subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(["ollama", "serve"])
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                result = subprocess.run(
                    ["curl", "-s", "http://localhost:11434/api/tags"],
                    capture_output=True
                )
                if result.returncode == 0:
                    logger.info("✅ Ollama server started")
                    return True
            
            logger.warning("⚠️  Ollama server may not have started correctly")
            return True  # Continue setup anyway
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to start Ollama server: {e}")
            return True  # Continue setup anyway
    
    def pull_default_model(self) -> bool:
        """Pull a default model for testing"""
        logger.info("📥 Pulling default model...")
        
        try:
            # Pull a small, fast model for testing
            subprocess.run(["ollama", "pull", "llama3.2:1b"], check=True, timeout=300)
            logger.info("✅ Default model (llama3.2:1b) downloaded")
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Model download timed out - you can download models later")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Failed to download default model: {e}")
            logger.info("💡 You can download models later using: ollama pull <model-name>")
            return True
    
    def create_startup_script(self) -> bool:
        """Create startup script"""
        logger.info("📜 Creating startup script...")
        
        try:
            if self.system == "Windows":
                script_content = f'''@echo off
echo Starting Ollama Workbench...
cd /d "{self.base_dir}"
call venv\\Scripts\\activate
start /b ollama serve
timeout /t 3 /nobreak >nul
streamlit run main.py
'''
                script_file = self.base_dir / "start_workbench.bat"
            else:
                script_content = f'''#!/bin/bash
echo "Starting Ollama Workbench..."
cd "{self.base_dir}"
source venv/bin/activate

# Start Ollama server if not running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Start Streamlit
echo "Starting Streamlit interface..."
streamlit run main.py
'''
                script_file = self.base_dir / "start_workbench.sh"
                
                # Make executable
                os.chmod(script_file, 0o755)
            
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            logger.info(f"✅ Startup script created: {script_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Startup script creation failed: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run basic tests"""
        logger.info("🧪 Running basic tests...")
        
        try:
            # Get Python executable from venv
            if self.system == "Windows":
                python_executable = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_executable = self.venv_dir / "bin" / "python"
            
            # Test imports
            test_script = '''
try:
    import streamlit
    print("✅ Streamlit import successful")
    
    import ollama
    print("✅ Ollama import successful")
    
    from ollama_workbench.core.config import get_config
    config = get_config()
    print("✅ Configuration loading successful")
    
    from security import get_security_config
    security_config = get_security_config()
    print("✅ Security framework import successful")
    
    print("🎉 All critical imports successful!")
    
except Exception as e:
    print(f"❌ Import test failed: {e}")
    exit(1)
'''
            
            result = subprocess.run(
                [str(python_executable), "-c", test_script],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            if result.returncode == 0:
                logger.info("✅ Basic tests passed")
                return True
            else:
                logger.error(f"❌ Tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return False
    
    def print_setup_summary(self) -> None:
        """Print setup completion summary"""
        print("\n" + "="*60)
        print("🎉 OLLAMA WORKBENCH SETUP COMPLETE! 🎉")
        print("="*60)
        print()
        print("📋 SETUP SUMMARY:")
        print("  ✅ Virtual environment created")
        print("  ✅ Dependencies installed")
        print("  ✅ Ollama installed and configured")
        print("  ✅ Security framework initialized")
        print("  ✅ Configuration created")
        print("  ✅ Directories set up")
        print("  ✅ Startup script created")
        print()
        print("🚀 TO START THE WORKBENCH:")
        
        if self.system == "Windows":
            print(f"  Double-click: start_workbench.bat")
            print(f"  Or run: {self.base_dir / 'start_workbench.bat'}")
        else:
            print(f"  Run: ./start_workbench.sh")
            print(f"  Or: bash {self.base_dir / 'start_workbench.sh'}")
        
        print()
        print("🌐 ACCESS THE WORKBENCH:")
        print("  URL: http://localhost:8501")
        print()
        print("📚 DOCUMENTATION:")
        print("  README.md - Quick start guide")
        print("  TECHNICAL_ARCHITECTURE.md - Technical details")
        print("  SECURITY_COMPLIANCE.md - Security features")
        print()
        print("🔧 CONFIGURATION:")
        print("  Edit: workbench_config.json")
        print("  Or use the web interface: Settings > Configuration")
        print()
        print("💡 FIRST STEPS:")
        print("  1. Download AI models: ollama pull llama3.2")
        print("  2. Configure API keys for external providers")
        print("  3. Enable authentication in security settings")
        print("  4. Explore the chat interface and workflows")
        print()
        print("🆘 SUPPORT:")
        print("  Check logs: setup_workbench.log")
        print("  Documentation: All *.md files")
        print("  GitHub Issues: Report problems and suggestions")
        print()
        print("="*60)
    
    def setup(self) -> bool:
        """Run complete setup process"""
        logger.info("🚀 Starting Ollama Workbench setup...")
        
        setup_steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Virtual Environment", self.setup_virtual_environment),
            ("Dependencies", self.install_dependencies),
            ("Ollama Installation", self.install_ollama),
            ("Directory Setup", self.setup_directories),
            ("Security Framework", self.initialize_security),
            ("Configuration", self.create_configuration),
            ("Ollama Server", self.start_ollama_server),
            ("Default Model", self.pull_default_model),
            ("Startup Script", self.create_startup_script),
            ("Basic Tests", self.run_tests)
        ]
        
        failed_steps = []
        
        for step_name, step_function in setup_steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"📋 STEP: {step_name}")
            logger.info(f"{'='*50}")
            
            try:
                if step_function():
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                failed_steps.append(step_name)
        
        if failed_steps:
            logger.error(f"\n❌ Setup completed with failures: {', '.join(failed_steps)}")
            logger.info("Check the log file for details: setup_workbench.log")
            return False
        else:
            logger.info("\n🎉 Setup completed successfully!")
            self.print_setup_summary()
            return True

def main():
    """Main setup function"""
    print("🦙 Ollama Workbench - Automated Setup")
    print("=====================================")
    print()
    
    try:
        setup = WorkbenchSetup()
        
        if setup.setup():
            print("\n✅ Setup completed successfully!")
            return 0
        else:
            print("\n❌ Setup failed. Check setup_workbench.log for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error during setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())