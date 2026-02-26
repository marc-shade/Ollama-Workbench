#!/usr/bin/env python3
"""
Modern Test Runner for Ollama Workbench

Features:
- Async test execution for better performance
- Modern error handling with detailed reporting
- Performance profiling of tests
- Security testing integration
- Compatibility testing across different environments
- Comprehensive coverage reporting
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modern modules
try:
    from modern_error_handling import get_error_handler, ErrorCategory, ErrorSeverity
    from performance_optimization import get_performance_stats, performance_monitored
    from modern_security import InputValidator, SecurityLevel
    from enhanced_ollama_client import EnhancedOllamaClient
    MODERN_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Modern modules not available: {e}")
    MODERN_MODULES_AVAILABLE = False


@dataclass
class TestResult:
    """Enhanced test result with performance and security metrics"""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str]
    performance_metrics: Dict[str, Any]
    security_events: List[Dict[str, Any]]
    timestamp: float


@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    test_files: List[str]
    dependencies: List[str]
    timeout: float
    parallel: bool


class ModernTestRunner:
    """
    Modern test runner with async support and comprehensive reporting
    """
    
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = results_dir
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        # Configure logging
        self.logger = self._setup_logging()
        
        # Initialize modern components if available
        if MODERN_MODULES_AVAILABLE:
            self.error_handler = get_error_handler()
        
        # Define test suites
        self.test_suites = self._define_test_suites()

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("ModernTestRunner")
        
        if not logger.handlers:
            # File handler for detailed logs
            log_file = os.path.join(self.results_dir, f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            ))
            
            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(levelname)s - %(message)s'
            ))
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
        
        return logger

    def _define_test_suites(self) -> Dict[str, TestSuite]:
        """Define test suites for different components"""
        return {
            "core": TestSuite(
                name="Core Functionality Tests",
                test_files=[
                    "test_main.py",
                    "test_ollama_utils.py", 
                    "test_openai_utils.py",
                    "test_config.py"
                ],
                dependencies=["streamlit", "ollama", "openai"],
                timeout=300,
                parallel=True
            ),
            "chat_interfaces": TestSuite(
                name="Chat Interface Tests",
                test_files=[
                    "tests/test_chat_interfaces.py",
                    "tests/test_model_settings.py",
                    "tests/test_advanced_features.py"
                ],
                dependencies=["streamlit", "ollama"],
                timeout=600,
                parallel=True
            ),
            "modern_modules": TestSuite(
                name="Modern Module Tests",
                test_files=[
                    "test_modern_error_handling.py",
                    "test_performance_optimization.py",
                    "test_modern_security.py",
                    "test_enhanced_ollama_client.py"
                ],
                dependencies=["cryptography", "httpx", "asyncio"],
                timeout=300,
                parallel=True
            ),
            "integration": TestSuite(
                name="Integration Tests",
                test_files=[
                    "test_full_workflow.py",
                    "test_multimodal_integration.py",
                    "test_provider_compatibility.py"
                ],
                dependencies=["streamlit", "ollama", "openai", "groq"],
                timeout=900,
                parallel=False
            ),
            "performance": TestSuite(
                name="Performance Tests",
                test_files=[
                    "test_performance_benchmarks.py",
                    "test_memory_usage.py",
                    "test_concurrent_usage.py"
                ],
                dependencies=["psutil", "memory_profiler"],
                timeout=1200,
                parallel=False
            )
        }

    @performance_monitored("check_dependencies")
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check if all required dependencies are available"""
        self.logger.info("Checking dependencies...")
        
        required_modules = [
            "streamlit", "ollama", "openai", "groq", "mistralai",
            "numpy", "pandas", "torch", "transformers",
            "cryptography", "httpx", "psutil"
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                self.logger.debug(f"✓ {module}")
            except ImportError:
                missing_modules.append(module)
                self.logger.warning(f"✗ {module}")
        
        if missing_modules:
            self.logger.error(f"Missing modules: {', '.join(missing_modules)}")
            print(f"Missing required modules: {', '.join(missing_modules)}")
            print("Run: pip install -r requirements.txt")
            return False, missing_modules
        
        self.logger.info("All dependencies satisfied")
        return True, []

    @performance_monitored("run_single_test")
    async def run_single_test(self, test_file: str, timeout: float = 300) -> TestResult:
        """Run a single test file with timeout and error handling"""
        test_name = os.path.basename(test_file)
        start_time = time.time()
        
        self.logger.info(f"Running test: {test_name}")
        
        try:
            if not os.path.exists(test_file):
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    duration=0.0,
                    error_message=f"Test file not found: {test_file}",
                    performance_metrics={},
                    security_events=[],
                    timestamp=start_time
                )
            
            # Run test with timeout
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                duration = time.time() - start_time
                
                # Parse test results
                if process.returncode == 0:
                    status = "passed"
                    error_message = None
                else:
                    status = "failed"
                    error_message = stderr.decode() if stderr else "Unknown error"
                
                # Get performance metrics if available
                perf_metrics = {}
                if MODERN_MODULES_AVAILABLE:
                    perf_metrics = get_performance_stats(test_name) or {}
                
                return TestResult(
                    test_name=test_name,
                    status=status,
                    duration=duration,
                    error_message=error_message,
                    performance_metrics=perf_metrics,
                    security_events=[],
                    timestamp=start_time
                )
                
            except asyncio.TimeoutError:
                process.kill()
                return TestResult(
                    test_name=test_name,
                    status="timeout",
                    duration=timeout,
                    error_message=f"Test timed out after {timeout}s",
                    performance_metrics={},
                    security_events=[],
                    timestamp=start_time
                )
        
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error running test: {str(e)}\n{traceback.format_exc()}"
            
            if MODERN_MODULES_AVAILABLE:
                self.error_handler.handle_error(
                    e, 
                    context={"test_file": test_file},
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH
                )
            
            return TestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                error_message=error_msg,
                performance_metrics={},
                security_events=[],
                timestamp=start_time
            )

    async def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run a complete test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite = self.test_suites[suite_name]
        self.logger.info(f"Running test suite: {suite.name}")
        
        # Check dependencies for this suite
        missing_deps = []
        for dep in suite.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            self.logger.warning(f"Skipping suite {suite_name} due to missing dependencies: {missing_deps}")
            return []
        
        if suite.parallel:
            # Run tests in parallel
            tasks = [
                self.run_single_test(test_file, suite.timeout)
                for test_file in suite.test_files
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(TestResult(
                        test_name=suite.test_files[i],
                        status="error",
                        duration=0.0,
                        error_message=str(result),
                        performance_metrics={},
                        security_events=[],
                        timestamp=time.time()
                    ))
                else:
                    final_results.append(result)
            
            return final_results
        else:
            # Run tests sequentially
            results = []
            for test_file in suite.test_files:
                result = await self.run_single_test(test_file, suite.timeout)
                results.append(result)
            
            return results

    async def run_all_tests(self, selected_suites: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report"""
        self.logger.info("Starting comprehensive test run")
        print("🚀 Starting modern test runner...")
        
        # Check dependencies first
        deps_ok, missing_deps = self.check_dependencies()
        if not deps_ok:
            return {
                "status": "failed",
                "error": f"Missing dependencies: {missing_deps}",
                "results": [],
                "summary": {}
            }
        
        # Determine which suites to run
        suites_to_run = selected_suites or list(self.test_suites.keys())
        
        # Run test suites
        all_results = []
        suite_results = {}
        
        for suite_name in suites_to_run:
            print(f"📋 Running {self.test_suites[suite_name].name}...")
            suite_start = time.time()
            
            try:
                results = await self.run_test_suite(suite_name)
                suite_results[suite_name] = results
                all_results.extend(results)
                
                # Print suite summary
                passed = len([r for r in results if r.status == "passed"])
                failed = len([r for r in results if r.status == "failed"])
                errors = len([r for r in results if r.status == "error"])
                skipped = len([r for r in results if r.status == "skipped"])
                
                suite_duration = time.time() - suite_start
                print(f"   ✅ {passed} passed, ❌ {failed} failed, 🚫 {errors} errors, ⏭️ {skipped} skipped")
                print(f"   ⏱️ Completed in {suite_duration:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error running suite {suite_name}: {e}")
                print(f"   💥 Suite failed: {e}")
        
        # Store results
        self.results = all_results
        
        # Generate comprehensive report
        return self._generate_report(suite_results)

    def _generate_report(self, suite_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "passed"])
        failed = len([r for r in self.results if r.status == "failed"])
        errors = len([r for r in self.results if r.status == "error"])
        skipped = len([r for r in self.results if r.status == "skipped"])
        timeouts = len([r for r in self.results if r.status == "timeout"])
        
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        # Performance statistics
        avg_test_duration = sum(r.duration for r in self.results) / total_tests if total_tests > 0 else 0
        slowest_test = max(self.results, key=lambda r: r.duration) if self.results else None
        
        # Generate detailed report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "timeouts": timeouts,
                "success_rate": success_rate
            },
            "performance": {
                "average_test_duration": avg_test_duration,
                "slowest_test": {
                    "name": slowest_test.test_name if slowest_test else None,
                    "duration": slowest_test.duration if slowest_test else None
                }
            },
            "suite_results": {
                suite_name: [asdict(result) for result in results]
                for suite_name, results in suite_results.items()
            },
            "failed_tests": [
                {
                    "test_name": r.test_name,
                    "error_message": r.error_message,
                    "duration": r.duration
                }
                for r in self.results if r.status in ["failed", "error"]
            ]
        }
        
        # Save report to file
        report_file = os.path.join(self.results_dir, "test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Test report saved to {report_file}")
        
        return report

    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable test summary"""
        print("\n" + "="*80)
        print("🧪 MODERN TEST RUNNER SUMMARY")
        print("="*80)
        
        summary = report["summary"]
        
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"🚫 Errors: {summary['errors']}")
        print(f"⏭️ Skipped: {summary['skipped']}")
        print(f"⏰ Timeouts: {summary['timeouts']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Total Duration: {report['total_duration']:.2f}s")
        
        if report["failed_tests"]:
            print("\n❌ FAILED TESTS:")
            for test in report["failed_tests"]:
                print(f"   • {test['test_name']} ({test['duration']:.2f}s)")
                if test['error_message']:
                    # Show first line of error
                    error_lines = test['error_message'].split('\n')
                    print(f"     Error: {error_lines[0]}")
        
        print("\n" + "="*80)
        
        if summary['success_rate'] >= 90:
            print("🎉 Excellent! Most tests are passing.")
        elif summary['success_rate'] >= 70:
            print("⚠️ Good, but some tests need attention.")
        else:
            print("🚨 Many tests are failing. Please review and fix issues.")


async def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Modern Test Runner for Ollama Workbench")
    parser.add_argument("--suites", nargs="+", help="Test suites to run", 
                       choices=["core", "chat_interfaces", "modern_modules", "integration", "performance"],
                       default=None)
    parser.add_argument("--results-dir", default="test_results", help="Results directory")
    
    args = parser.parse_args()
    
    # Create and run test runner
    runner = ModernTestRunner(results_dir=args.results_dir)
    
    try:
        report = await runner.run_all_tests(selected_suites=args.suites)
        runner.print_summary(report)
        
        # Exit with appropriate code
        if report["summary"]["success_rate"] == 100:
            sys.exit(0)
        elif report["summary"]["success_rate"] >= 80:
            sys.exit(1)  # Some failures but mostly working
        else:
            sys.exit(2)  # Many failures
            
    except KeyboardInterrupt:
        print("\n⚠️ Test run interrupted by user")
        sys.exit(3)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        runner.logger.error(f"Test runner failed: {e}", exc_info=True)
        sys.exit(4)


if __name__ == "__main__":
    # Ensure event loop compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())