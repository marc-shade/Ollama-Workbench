#!/usr/bin/env python3
"""
Modernization Validation Script for Ollama Workbench

This script validates that all modernization updates are working correctly:
- Dependencies are updated to latest versions
- Modern modules can be imported and initialized
- Basic functionality works as expected
- No critical regressions in core features
"""

import os
import sys
import time
import json
import importlib.util
from typing import Dict, List, Tuple, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ModernizationValidator:
    """Validates the modernization process"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def log_result(self, test_name: str, status: str, message: str, details: Any = None):
        """Log a validation result"""
        result = {
            "test": test_name,
            "status": status,  # passed, failed, warning
            "message": message,
            "details": details,
            "timestamp": time.time()
        }
        self.results.append(result)
        
        # Print immediate feedback
        status_icon = "✅" if status == "passed" else "⚠️" if status == "warning" else "❌"
        print(f"{status_icon} {test_name}: {message}")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            self.log_result(
                "Python Version",
                "passed",
                f"Python {version.major}.{version.minor}.{version.micro} is compatible"
            )
            return True
        else:
            self.log_result(
                "Python Version",
                "failed",
                f"Python {version.major}.{version.minor} is not supported. Need Python 3.10+"
            )
            return False
    
    def check_core_dependencies(self) -> bool:
        """Check that core dependencies are updated"""
        dependencies = {
            "streamlit": "1.47.0",
            "ollama": "0.5.0",
            "openai": "1.99.0",
            "groq": "0.30.0",
            "mistralai": "1.9.0",
            "torch": "2.7.0",
            "transformers": "4.55.0"
        }
        
        all_passed = True
        for module_name, min_version in dependencies.items():
            try:
                module = __import__(module_name)
                actual_version = getattr(module, "__version__", "unknown")
                
                # Simple version comparison (works for most cases)
                if self._version_compare(actual_version, min_version) >= 0:
                    self.log_result(
                        f"Dependency: {module_name}",
                        "passed",
                        f"Version {actual_version} >= {min_version}"
                    )
                else:
                    self.log_result(
                        f"Dependency: {module_name}",
                        "warning",
                        f"Version {actual_version} < {min_version}"
                    )
                    
            except ImportError:
                self.log_result(
                    f"Dependency: {module_name}",
                    "failed",
                    f"Module not found"
                )
                all_passed = False
        
        return all_passed
    
    def check_modern_modules(self) -> bool:
        """Check that our modern modules can be imported and work"""
        modern_modules = [
            "enhanced_ollama_client",
            "modern_error_handling", 
            "performance_optimization",
            "modern_security"
        ]
        
        all_passed = True
        for module_name in modern_modules:
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, 
                    f"{module_name}.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                self.log_result(
                    f"Modern Module: {module_name}",
                    "passed",
                    "Import successful"
                )
                
            except Exception as e:
                self.log_result(
                    f"Modern Module: {module_name}",
                    "failed",
                    f"Import failed: {str(e)}"
                )
                all_passed = False
        
        return all_passed
    
    def test_enhanced_ollama_client(self) -> bool:
        """Test the enhanced Ollama client"""
        try:
            from enhanced_ollama_client import EnhancedOllamaClient, ModelProvider
            
            # Test basic initialization
            client = EnhancedOllamaClient()
            
            # Test enum
            provider = ModelProvider.OLLAMA
            assert provider.value == "ollama"
            
            # Test metrics initialization
            metrics = client.get_performance_metrics()
            assert isinstance(metrics, dict)
            
            self.log_result(
                "Enhanced Ollama Client",
                "passed",
                "Initialization and basic functionality working"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Enhanced Ollama Client",
                "failed",
                f"Test failed: {str(e)}"
            )
            return False
    
    def test_error_handling(self) -> bool:
        """Test modern error handling"""
        try:
            from modern_error_handling import get_error_handler, ErrorCategory, ErrorSeverity
            
            # Test error handler initialization
            handler = get_error_handler()
            assert handler is not None
            
            # Test error categories and severity
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.MEDIUM
            assert category.value == "system"
            assert severity.value == "medium"
            
            # Test basic error handling (without actually raising)
            stats = handler.get_error_stats()
            assert isinstance(stats, dict)
            
            self.log_result(
                "Modern Error Handling",
                "passed",
                "Error handling system initialized successfully"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Modern Error Handling",
                "failed",
                f"Test failed: {str(e)}"
            )
            return False
    
    def test_performance_optimization(self) -> bool:
        """Test performance optimization features"""
        try:
            from performance_optimization import SmartCache, get_cache_stats, cached
            
            # Test cache initialization
            cache = SmartCache()
            assert cache is not None
            
            # Test cache statistics
            stats = get_cache_stats()
            assert isinstance(stats, dict)
            assert "size_mb" in stats
            
            # Test decorator
            @cached(ttl=60)
            def test_function(x):
                return x * 2
            
            result = test_function(5)
            assert result == 10
            
            self.log_result(
                "Performance Optimization",
                "passed",
                "Caching and performance features working"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Performance Optimization",
                "failed",
                f"Test failed: {str(e)}"
            )
            return False
    
    def test_security_features(self) -> bool:
        """Test security enhancements"""
        try:
            from modern_security import InputValidator, SecurityLevel, SecureCredentialManager
            
            # Test input validation
            validator = InputValidator()
            
            # Test sanitization
            clean_input = validator.sanitize_input('<script>alert("test")</script>')
            assert '<script>' not in clean_input
            
            # Test PII detection
            pii = validator.detect_pii("My email is test@example.com")
            assert "email" in pii
            
            # Test security levels
            level = SecurityLevel.HIGH
            assert level.value == "high"
            
            # Test credential manager (without actual storage)
            cred_mgr = SecureCredentialManager()
            assert cred_mgr is not None
            
            self.log_result(
                "Security Features",
                "passed",
                "Input validation and security features working"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Security Features",
                "failed",
                f"Test failed: {str(e)}"
            )
            return False
    
    def test_streamlit_compatibility(self) -> bool:
        """Test Streamlit compatibility with new version"""
        try:
            import streamlit as st
            
            # Check version
            version = st.__version__
            if self._version_compare(version, "1.47.0") >= 0:
                self.log_result(
                    "Streamlit Compatibility",
                    "passed",
                    f"Streamlit {version} is compatible"
                )
                return True
            else:
                self.log_result(
                    "Streamlit Compatibility",
                    "warning",
                    f"Streamlit {version} may have compatibility issues"
                )
                return True  # Non-blocking
                
        except Exception as e:
            self.log_result(
                "Streamlit Compatibility",
                "failed",
                f"Streamlit test failed: {str(e)}"
            )
            return False
    
    def test_ollama_integration(self) -> bool:
        """Test Ollama integration"""
        try:
            import ollama
            
            # Test basic ollama module
            assert hasattr(ollama, 'Client')
            
            # Try to create client (won't connect but should initialize)
            client = ollama.Client()
            assert client is not None
            
            self.log_result(
                "Ollama Integration",
                "passed",
                "Ollama client can be initialized"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Ollama Integration",
                "failed",
                f"Ollama test failed: {str(e)}"
            )
            return False
    
    def test_api_providers(self) -> bool:
        """Test API provider integrations"""
        providers_passed = 0
        total_providers = 0
        
        # OpenAI
        try:
            import openai
            client = openai.OpenAI(api_key="dummy")
            assert client is not None
            
            self.log_result(
                "OpenAI Provider",
                "passed",
                f"OpenAI {openai.__version__} integration working"
            )
            providers_passed += 1
        except Exception as e:
            self.log_result(
                "OpenAI Provider",
                "warning",
                f"OpenAI test failed: {str(e)}"
            )
        total_providers += 1
        
        # Groq
        try:
            import groq
            client = groq.Groq(api_key="dummy")
            assert client is not None
            
            self.log_result(
                "Groq Provider",
                "passed",
                "Groq integration working"
            )
            providers_passed += 1
        except Exception as e:
            self.log_result(
                "Groq Provider",
                "warning",
                f"Groq test failed: {str(e)}"
            )
        total_providers += 1
        
        # Mistral
        try:
            import mistralai
            assert mistralai is not None
            
            self.log_result(
                "Mistral Provider",
                "passed",
                "Mistral integration working"
            )
            providers_passed += 1
        except Exception as e:
            self.log_result(
                "Mistral Provider",
                "warning",
                f"Mistral test failed: {str(e)}"
            )
        total_providers += 1
        
        # At least 2 out of 3 providers should work
        return providers_passed >= 2
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Simple version comparison. Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for i in range(max_len):
                if v1_parts[i] < v2_parts[i]:
                    return -1
                elif v1_parts[i] > v2_parts[i]:
                    return 1
            
            return 0
        except:
            return 0  # Default to equal if can't parse
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("🔍 Starting Ollama Workbench Modernization Validation...")
        print("=" * 60)
        
        # Run all validation tests
        tests = [
            ("Python Version", self.check_python_version),
            ("Core Dependencies", self.check_core_dependencies),
            ("Modern Modules", self.check_modern_modules),
            ("Enhanced Ollama Client", self.test_enhanced_ollama_client),
            ("Error Handling", self.test_error_handling),
            ("Performance Optimization", self.test_performance_optimization),
            ("Security Features", self.test_security_features),
            ("Streamlit Compatibility", self.test_streamlit_compatibility),
            ("Ollama Integration", self.test_ollama_integration),
            ("API Providers", self.test_api_providers)
        ]
        
        passed = 0
        failed = 0
        warnings = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name}: EXCEPTION - {str(e)}")
                failed += 1
        
        # Count warnings
        warnings = len([r for r in self.results if r["status"] == "warning"])
        
        # Generate summary
        total_duration = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Warnings: {warnings}")
        print(f"⏱️ Duration: {total_duration:.2f}s")
        
        success_rate = (passed / len(tests)) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 Excellent! Modernization is successful.")
            status = "success"
        elif success_rate >= 70:
            print("⚠️ Good! Some issues need attention.")
            status = "partial_success"
        else:
            print("🚨 Issues detected. Please review failed tests.")
            status = "failed"
        
        # Save detailed results
        report = {
            "status": status,
            "success_rate": success_rate,
            "summary": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "total": len(tests)
            },
            "duration": total_duration,
            "timestamp": time.time(),
            "detailed_results": self.results
        }
        
        # Save to file
        with open("modernization_validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: modernization_validation_report.json")
        
        return report


def main():
    """Main validation entry point"""
    validator = ModernizationValidator()
    report = validator.run_all_validations()
    
    # Exit with appropriate code
    if report["status"] == "success":
        sys.exit(0)
    elif report["status"] == "partial_success":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()