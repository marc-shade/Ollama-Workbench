#!/usr/bin/env python3
"""
Ollama Workbench - Deployment Validation Script
Comprehensive validation script to ensure the platform is properly configured and ready for use.
"""

import os
import sys
import subprocess
import platform
import logging
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment_validation.log')
    ]
)
logger = logging.getLogger(__name__)

class DeploymentValidator:
    """Comprehensive deployment validation for Ollama Workbench"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.base_dir = Path(__file__).parent
        self.venv_dir = self.base_dir / "venv"
        self.validation_results = {}
        self.critical_failures = []
        self.warnings = []
        
    def validate_python_environment(self) -> bool:
        """Validate Python environment and virtual environment"""
        logger.info("🐍 Validating Python environment...")
        
        try:
            # Check Python version
            if self.python_version < (3, 8):
                self.critical_failures.append(f"Python 3.8+ required, found {self.python_version.major}.{self.python_version.minor}")
                return False
            
            logger.info(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
            
            # Check virtual environment
            if not self.venv_dir.exists():
                self.critical_failures.append("Virtual environment not found")
                return False
            
            # Get Python executable from venv
            if self.system == "Windows":
                python_exe = self.venv_dir / "Scripts" / "python.exe"
                pip_exe = self.venv_dir / "Scripts" / "pip.exe"
            else:
                python_exe = self.venv_dir / "bin" / "python"
                pip_exe = self.venv_dir / "bin" / "pip"
            
            if not python_exe.exists() or not pip_exe.exists():
                self.critical_failures.append("Virtual environment is incomplete")
                return False
            
            logger.info("✅ Virtual environment is properly configured")
            return True
            
        except Exception as e:
            self.critical_failures.append(f"Python environment validation failed: {e}")
            return False
    
    def validate_dependencies(self) -> bool:
        """Validate that all critical dependencies are installed"""
        logger.info("📦 Validating dependencies...")
        
        critical_packages = [
            "streamlit",
            "ollama", 
            "openai",
            "cryptography",
            "bcrypt",
            "PyJWT",
            "requests",
            "pandas",
            "numpy"
        ]
        
        try:
            # Get Python executable from venv
            if self.system == "Windows":
                python_exe = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_exe = self.venv_dir / "bin" / "python"
            
            missing_packages = []
            
            for package in critical_packages:
                try:
                    result = subprocess.run(
                        [str(python_exe), "-c", f"import {package}"],
                        capture_output=True,
                        text=True,
                        cwd=str(self.base_dir)
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"✅ {package}")
                    else:
                        missing_packages.append(package)
                        logger.error(f"❌ {package} - not found")
                        
                except Exception as e:
                    missing_packages.append(package)
                    logger.error(f"❌ {package} - error: {e}")
            
            if missing_packages:
                self.critical_failures.append(f"Missing critical packages: {', '.join(missing_packages)}")
                return False
            
            logger.info("✅ All critical dependencies are installed")
            return True
            
        except Exception as e:
            self.critical_failures.append(f"Dependency validation failed: {e}")
            return False
    
    def validate_ollama_installation(self) -> bool:
        """Validate Ollama installation and server"""
        logger.info("🦙 Validating Ollama installation...")
        
        try:
            # Check if Ollama command exists
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.critical_failures.append("Ollama command not found")
                return False
            
            logger.info(f"✅ Ollama installed: {result.stdout.strip()}")
            
            # Check if server is running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    logger.info(f"✅ Ollama server running with {len(models)} models")
                    
                    if len(models) == 0:
                        self.warnings.append("No models downloaded - consider running 'ollama pull llama3.2:1b'")
                    
                    return True
                else:
                    self.warnings.append("Ollama server not responding correctly")
                    return False
                    
            except requests.RequestException:
                self.warnings.append("Ollama server not running - start with 'ollama serve'")
                return False
                
        except FileNotFoundError:
            self.critical_failures.append("Ollama not installed")
            return False
        except Exception as e:
            self.critical_failures.append(f"Ollama validation failed: {e}")
            return False
    
    def validate_configuration(self) -> bool:
        """Validate configuration files and settings"""
        logger.info("⚙️ Validating configuration...")
        
        try:
            # Check for configuration file
            config_file = self.base_dir / "workbench_config.json"
            if not config_file.exists():
                # Try to import config module
                if self.system == "Windows":
                    python_exe = self.venv_dir / "Scripts" / "python.exe"
                else:
                    python_exe = self.venv_dir / "bin" / "python"
                
                result = subprocess.run(
                    [str(python_exe), "-c", "from config import get_config; print('OK')"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.base_dir)
                )
                
                if result.returncode != 0:
                    self.critical_failures.append("Configuration system not working")
                    return False
                
                self.warnings.append("No config file found, using defaults")
            else:
                # Validate config file
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    required_keys = [
                        "OLLAMA_HOST",
                        "WORKBENCH_PORT",
                        "ENABLE_ENHANCED_SECURITY"
                    ]
                    
                    missing_keys = [key for key in required_keys if key not in config]
                    if missing_keys:
                        self.warnings.append(f"Configuration missing keys: {', '.join(missing_keys)}")
                    
                    logger.info("✅ Configuration file is valid")
                    
                except json.JSONDecodeError:
                    self.critical_failures.append("Configuration file is not valid JSON")
                    return False
            
            return True
            
        except Exception as e:
            self.critical_failures.append(f"Configuration validation failed: {e}")
            return False
    
    def validate_security_framework(self) -> bool:
        """Validate security framework installation"""
        logger.info("🔐 Validating security framework...")
        
        try:
            # Get Python executable from venv
            if self.system == "Windows":
                python_exe = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_exe = self.venv_dir / "bin" / "python"
            
            # Test security module imports
            security_test = '''
try:
    from security import get_security_config, get_audit_logger
    from security.authentication import get_auth_manager
    from security.encryption import get_encryption_manager
    from security.access_control import get_access_control_manager
    print("Security framework OK")
except ImportError as e:
    print(f"Security import error: {e}")
    exit(1)
except Exception as e:
    print(f"Security error: {e}")
    exit(1)
'''
            
            result = subprocess.run(
                [str(python_exe), "-c", security_test],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            if result.returncode == 0:
                logger.info("✅ Security framework is properly installed")
                
                # Check security directories
                security_dir = self.base_dir / "security"
                if security_dir.exists():
                    logger.info("✅ Security directory exists")
                else:
                    self.warnings.append("Security directory not found")
                
                return True
            else:
                self.warnings.append(f"Security framework issues: {result.stderr}")
                return False
                
        except Exception as e:
            self.warnings.append(f"Security validation failed: {e}")
            return False
    
    def validate_observability(self) -> bool:
        """Validate observability integration"""
        logger.info("📊 Validating observability...")
        
        try:
            # Get Python executable from venv
            if self.system == "Windows":
                python_exe = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_exe = self.venv_dir / "bin" / "python"
            
            # Test observability imports
            observability_test = '''
try:
    from observability.opik_integration import get_opik_integration
    print("Observability framework OK")
except ImportError as e:
    print(f"Observability import error: {e}")
    exit(1)
except Exception as e:
    print(f"Observability error: {e}")
    exit(1)
'''
            
            result = subprocess.run(
                [str(python_exe), "-c", observability_test],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            if result.returncode == 0:
                logger.info("✅ Observability framework is properly installed")
                return True
            else:
                self.warnings.append(f"Observability framework issues: {result.stderr}")
                return False
                
        except Exception as e:
            self.warnings.append(f"Observability validation failed: {e}")
            return False
    
    def validate_directories(self) -> bool:
        """Validate required directories exist"""
        logger.info("📁 Validating directories...")
        
        required_dirs = [
            "data",
            "uploads",
            "models", 
            "projects",
            "cache",
            "security",
            "logs",
            "sessions"
        ]
        
        missing_dirs = []
        
        try:
            for directory in required_dirs:
                dir_path = self.base_dir / directory
                if not dir_path.exists():
                    missing_dirs.append(directory)
                    logger.warning(f"⚠️ Missing directory: {directory}")
                else:
                    logger.info(f"✅ Directory exists: {directory}")
            
            if missing_dirs:
                self.warnings.append(f"Missing directories: {', '.join(missing_dirs)}")
                # Try to create missing directories
                for directory in missing_dirs:
                    try:
                        dir_path = self.base_dir / directory
                        dir_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"✅ Created directory: {directory}")
                    except Exception as e:
                        logger.error(f"❌ Failed to create directory {directory}: {e}")
            
            return True
            
        except Exception as e:
            self.warnings.append(f"Directory validation failed: {e}")
            return False
    
    def validate_startup_scripts(self) -> bool:
        """Validate startup scripts exist and are executable"""
        logger.info("📜 Validating startup scripts...")
        
        try:
            if self.system == "Windows":
                script_file = self.base_dir / "start_workbench.bat"
            else:
                script_file = self.base_dir / "start_workbench.sh"
            
            if not script_file.exists():
                self.warnings.append(f"Startup script not found: {script_file.name}")
                return False
            
            # Check if script is executable (Unix-like systems)
            if self.system != "Windows":
                import stat
                file_stat = script_file.stat()
                if not (file_stat.st_mode & stat.S_IEXEC):
                    self.warnings.append("Startup script is not executable")
                    try:
                        os.chmod(script_file, 0o755)
                        logger.info("✅ Made startup script executable")
                    except Exception as e:
                        logger.error(f"❌ Failed to make script executable: {e}")
            
            logger.info(f"✅ Startup script exists: {script_file.name}")
            return True
            
        except Exception as e:
            self.warnings.append(f"Startup script validation failed: {e}")
            return False
    
    def validate_streamlit_functionality(self) -> bool:
        """Validate Streamlit can start properly"""
        logger.info("🌐 Validating Streamlit functionality...")
        
        try:
            # Get Python executable from venv
            if self.system == "Windows":
                python_exe = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_exe = self.venv_dir / "bin" / "python"
            
            # Test Streamlit import and basic functionality
            streamlit_test = '''
try:
    import streamlit as st
    from main import *
    print("Streamlit functionality OK")
except ImportError as e:
    print(f"Streamlit import error: {e}")
    exit(1)
except Exception as e:
    print(f"Streamlit error: {e}")
    exit(1)
'''
            
            result = subprocess.run(
                [str(python_exe), "-c", streamlit_test],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ Streamlit functionality validated")
                return True
            else:
                self.warnings.append(f"Streamlit issues: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.warnings.append("Streamlit validation timed out")
            return False
        except Exception as e:
            self.warnings.append(f"Streamlit validation failed: {e}")
            return False
    
    def validate_network_connectivity(self) -> bool:
        """Validate network connectivity for external services"""
        logger.info("🌐 Validating network connectivity...")
        
        test_urls = [
            ("Ollama API", "https://ollama.ai"),
            ("OpenAI API", "https://api.openai.com"),
            ("Groq API", "https://api.groq.com"),
            ("GitHub", "https://github.com")
        ]
        
        connectivity_issues = []
        
        for service, url in test_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code < 400:
                    logger.info(f"✅ {service} connectivity OK")
                else:
                    connectivity_issues.append(f"{service} returned {response.status_code}")
                    
            except requests.RequestException as e:
                connectivity_issues.append(f"{service}: {str(e)}")
                logger.warning(f"⚠️ {service} connectivity issue: {e}")
        
        if connectivity_issues:
            self.warnings.extend(connectivity_issues)
            logger.info("⚠️ Some network connectivity issues detected (may not affect local operation)")
        else:
            logger.info("✅ All network connectivity tests passed")
        
        return True  # Network issues are warnings, not critical failures
    
    def run_integration_tests(self) -> bool:
        """Run basic integration tests"""
        logger.info("🧪 Running integration tests...")
        
        try:
            # Get Python executable from venv
            if self.system == "Windows":
                python_exe = self.venv_dir / "Scripts" / "python.exe"
            else:
                python_exe = self.venv_dir / "bin" / "python"
            
            # Test basic application imports and initialization
            integration_test = '''
import sys
sys.path.insert(0, ".")

try:
    # Test core imports
    from config import get_config
    config = get_config()
    print("✓ Configuration system")
    
    # Test security if enabled
    if config.get("ENABLE_ENHANCED_SECURITY"):
        from security import get_security_config
        security_config = get_security_config()
        print("✓ Security framework")
    
    # Test observability if enabled
    if config.get("ENABLE_OBSERVABILITY"):
        from observability.opik_integration import is_opik_enabled
        print("✓ Observability framework")
    
    # Test Ollama utilities
    from ollama_utils import get_available_models
    print("✓ Ollama utilities")
    
    print("All integration tests passed")
    
except Exception as e:
    print(f"Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
'''
            
            result = subprocess.run(
                [str(python_exe), "-c", integration_test],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✅ All integration tests passed")
                return True
            else:
                self.critical_failures.append(f"Integration tests failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.critical_failures.append("Integration tests timed out")
            return False
        except Exception as e:
            self.critical_failures.append(f"Integration test execution failed: {e}")
            return False
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results.values() if result)
        
        report = {
            "timestamp": time.time(),
            "system_info": {
                "platform": platform.platform(),
                "python_version": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
                "architecture": platform.architecture()[0]
            },
            "validation_summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "success_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0
            },
            "validation_results": self.validation_results,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "deployment_ready": len(self.critical_failures) == 0,
            "recommendations": []
        }
        
        # Generate recommendations
        if self.critical_failures:
            report["recommendations"].append("Address critical failures before deployment")
        
        if self.warnings:
            report["recommendations"].append("Review warnings for optimal configuration")
        
        if len(self.critical_failures) == 0 and len(self.warnings) == 0:
            report["recommendations"].append("Deployment is ready - all checks passed!")
        
        return report
    
    def validate_deployment(self) -> bool:
        """Run complete deployment validation"""
        logger.info("🚀 Starting Ollama Workbench deployment validation...")
        
        validation_steps = [
            ("Python Environment", self.validate_python_environment),
            ("Dependencies", self.validate_dependencies),
            ("Ollama Installation", self.validate_ollama_installation),
            ("Configuration", self.validate_configuration),
            ("Security Framework", self.validate_security_framework),
            ("Observability", self.validate_observability),
            ("Directories", self.validate_directories),
            ("Startup Scripts", self.validate_startup_scripts),
            ("Streamlit Functionality", self.validate_streamlit_functionality),
            ("Network Connectivity", self.validate_network_connectivity),
            ("Integration Tests", self.run_integration_tests)
        ]
        
        for step_name, step_function in validation_steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"📋 VALIDATING: {step_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = step_function()
                self.validation_results[step_name] = result
                
                if result:
                    logger.info(f"✅ {step_name} - PASSED")
                else:
                    logger.error(f"❌ {step_name} - FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                self.validation_results[step_name] = False
                self.critical_failures.append(f"{step_name}: {str(e)}")
        
        # Generate final report
        report = self.generate_validation_report()
        
        # Save report to file
        report_file = self.base_dir / "deployment_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self.print_validation_summary(report)
        
        return report["deployment_ready"]
    
    def print_validation_summary(self, report: Dict[str, Any]) -> None:
        """Print validation summary"""
        
        print("\n" + "="*60)
        if report["deployment_ready"]:
            print("🎉 DEPLOYMENT VALIDATION SUCCESSFUL! 🎉")
        else:
            print("❌ DEPLOYMENT VALIDATION FAILED ❌")
        print("="*60)
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"  Total Checks: {report['validation_summary']['total_checks']}")
        print(f"  Passed: {report['validation_summary']['passed_checks']}")
        print(f"  Failed: {report['validation_summary']['failed_checks']}")
        print(f"  Success Rate: {report['validation_summary']['success_rate']:.1f}%")
        
        if self.critical_failures:
            print(f"\n❌ CRITICAL FAILURES ({len(self.critical_failures)}):")
            for i, failure in enumerate(self.critical_failures, 1):
                print(f"  {i}. {failure}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        if report["deployment_ready"]:
            print(f"\n🚀 READY TO START:")
            if self.system == "Windows":
                print(f"  Double-click: start_workbench.bat")
            else:
                print(f"  Run: ./start_workbench.sh")
            
            print(f"\n🌐 ACCESS URL:")
            print(f"  http://localhost:8501")
        else:
            print(f"\n🔧 NEXT STEPS:")
            print(f"  1. Fix critical failures listed above")
            print(f"  2. Re-run validation: python validate_deployment.py")
            print(f"  3. Check logs: deployment_validation.log")
        
        print(f"\n📄 DETAILED REPORT:")
        print(f"  deployment_validation_report.json")
        print("="*60)

def main():
    """Main validation function"""
    print("🦙 Ollama Workbench - Deployment Validation")
    print("===========================================")
    print()
    
    try:
        validator = DeploymentValidator()
        
        if validator.validate_deployment():
            print("\n✅ Validation completed successfully!")
            print("Your Ollama Workbench deployment is ready!")
            return 0
        else:
            print("\n❌ Validation failed.")
            print("Please address the issues and run validation again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error during validation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())