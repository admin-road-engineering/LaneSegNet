#!/usr/bin/env python3
"""
LaneSegNet Test Runner Script
Provides convenient commands for running different types of tests.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False


def run_unit_tests(args):
    """Run unit tests with coverage."""
    cmd = f"pytest tests/ -m unit --cov=app --cov-report=html --cov-report=term-missing"
    
    if args.verbose:
        cmd += " -v"
    
    if args.parallel:
        cmd += f" -n {args.parallel}"
    
    if args.coverage_fail:
        cmd += f" --cov-fail-under={args.coverage_fail}"
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(args):
    """Run integration tests."""
    cmd = "pytest tests/ -m integration"
    
    if args.verbose:
        cmd += " -v"
    
    if args.parallel:
        cmd += f" -n {args.parallel}"
    
    return run_command(cmd, "Integration Tests")


def run_api_tests(args):
    """Run API tests."""
    # Check if server is running
    health_check = "curl -f http://localhost:8010/health"
    print("Checking if API server is running...")
    
    try:
        subprocess.run(health_check, shell=True, check=True, capture_output=True)
        print("✅ API server is running")
    except subprocess.CalledProcessError:
        print("❌ API server not running. Please start it first:")
        print("python -m uvicorn app.main:app --reload --port 8010")
        return False
    
    cmd = "pytest tests/ -m api"
    
    if args.verbose:
        cmd += " -v"
    
    return run_command(cmd, "API Tests")


def run_load_tests(args):
    """Run load tests using Locust."""
    if not args.host:
        args.host = "http://localhost:8010"
    
    # Basic load test parameters
    users = args.users or 10
    spawn_rate = args.spawn_rate or 2
    run_time = args.run_time or "60s"
    
    cmd = f"locust -f tests/test_load_testing.py --host={args.host} " \
          f"--users {users} --spawn-rate {spawn_rate} --run-time {run_time} " \
          f"--headless --csv=load_test_results"
    
    return run_command(cmd, f"Load Tests ({users} users, {run_time})")


def run_security_tests(args):
    """Run security tests."""
    success = True
    
    # Safety check for known vulnerabilities
    if not run_command("safety check --file requirements.txt", "Safety Check"):
        success = False
    
    # Bandit security linter
    if not run_command("bandit -r app/ -f json -o bandit-report.json", "Bandit Security Scan"):
        success = False
    
    return success


def run_code_quality(args):
    """Run code quality checks."""
    success = True
    
    # Black formatting check
    if not run_command("black --check app/ tests/", "Black Format Check"):
        if args.fix:
            run_command("black app/ tests/", "Auto-fixing with Black")
        else:
            success = False
    
    # Import sorting check
    if not run_command("isort --check-only app/ tests/", "Import Sort Check"):
        if args.fix:
            run_command("isort app/ tests/", "Auto-fixing with isort")
        else:
            success = False
    
    # Flake8 linting
    if not run_command("flake8 app/ tests/ --max-line-length=100", "Flake8 Linting"):
        success = False
    
    # MyPy type checking (non-blocking)
    run_command("mypy app/ --ignore-missing-imports", "MyPy Type Check")
    
    return success


def run_all_tests(args):
    """Run all tests in sequence."""
    print("🚀 Running complete test suite...")
    
    results = {
        "Code Quality": run_code_quality(args),
        "Unit Tests": run_unit_tests(args), 
        "Integration Tests": run_integration_tests(args),
        "Security Tests": run_security_tests(args),
    }
    
    # Only run API tests if server is available
    try:
        subprocess.run("curl -f http://localhost:8010/health", 
                      shell=True, check=True, capture_output=True)
        results["API Tests"] = run_api_tests(args)
    except subprocess.CalledProcessError:
        print("⚠️ Skipping API tests - server not running")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for test_type, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_type:20} {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


def debug_bypass_check(args):
    """Check for debug bypass code."""
    print("🔍 Checking for debug bypass patterns...")
    
    success = True
    
    # Check for specific debug bypass in enhanced_post_processing.py
    try:
        with open("app/enhanced_post_processing.py", "r") as f:
            content = f.read()
            
        if "debug_markings[:10]" in content:
            print("❌ CRITICAL: Debug bypass found in enhanced_post_processing.py")
            print("This bypasses physics-informed filtering!")
            success = False
        
        if "return debug_markings" in content:
            print("❌ CRITICAL: Debug return statement found")
            success = False
            
    except FileNotFoundError:
        print("⚠️ enhanced_post_processing.py not found")
    
    if success:
        print("✅ No debug bypass patterns detected")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="LaneSegNet Test Runner")
    subparsers = parser.add_subparsers(dest="command", help="Test commands")
    
    # Common arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    
    # Unit tests
    unit_parser = subparsers.add_parser("unit", help="Run unit tests")
    unit_parser.add_argument("--coverage-fail", type=int, default=95, 
                           help="Fail if coverage below this percentage")
    
    # Integration tests  
    subparsers.add_parser("integration", help="Run integration tests")
    
    # API tests
    subparsers.add_parser("api", help="Run API tests")
    
    # Load tests
    load_parser = subparsers.add_parser("load", help="Run load tests")
    load_parser.add_argument("--host", help="Target host for load testing")
    load_parser.add_argument("--users", type=int, help="Number of concurrent users")
    load_parser.add_argument("--spawn-rate", type=int, help="User spawn rate")
    load_parser.add_argument("--run-time", help="Test duration (e.g., 60s, 5m)")
    
    # Security tests
    subparsers.add_parser("security", help="Run security tests")
    
    # Code quality
    quality_parser = subparsers.add_parser("quality", help="Run code quality checks")
    quality_parser.add_argument("--fix", action="store_true", 
                               help="Auto-fix formatting issues")
    
    # All tests
    subparsers.add_parser("all", help="Run all tests")
    
    # Debug bypass check
    subparsers.add_parser("debug-check", help="Check for debug bypass code")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set environment for testing
    os.environ["TESTING_MODE"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Command dispatch
    commands = {
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "api": run_api_tests,
        "load": run_load_tests,
        "security": run_security_tests,
        "quality": run_code_quality,
        "all": run_all_tests,
        "debug-check": debug_bypass_check,
    }
    
    success = commands[args.command](args)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())