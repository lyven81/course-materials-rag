#!/usr/bin/env python3
"""Script to run all quality checks."""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=Path(__file__).parent.parent)
    return result.returncode


def main() -> int:
    """Run all quality checks."""
    print("Running code quality checks...")
    exit_code = 0

    # Check formatting
    print("\nChecking code formatting...")
    black_check = run_command(
        ["uv", "run", "black", "--check", "backend/", "main.py", "scripts/"]
    )
    isort_check = run_command(
        ["uv", "run", "isort", "--check-only", "backend/", "main.py", "scripts/"]
    )

    if black_check != 0:
        exit_code = black_check
        print("[FAIL] Black formatting check failed")
    else:
        print("[PASS] Black formatting check passed")

    if isort_check != 0:
        exit_code = isort_check
        print("[FAIL] isort import sorting check failed")
    else:
        print("[PASS] isort import sorting check passed")

    # Run linting
    print("\nRunning linting checks...")
    flake8_result = run_command(
        ["uv", "run", "flake8", "backend/", "main.py", "scripts/"]
    )
    if flake8_result != 0:
        exit_code = flake8_result
        print("[FAIL] Flake8 linting failed")
    else:
        print("[PASS] Flake8 linting passed")

    # Run type checking
    print("\nRunning type checking...")
    mypy_result = run_command(["uv", "run", "mypy", "backend/", "main.py", "scripts/"])
    if mypy_result != 0:
        exit_code = mypy_result
        print("[FAIL] MyPy type checking failed")
    else:
        print("[PASS] MyPy type checking passed")

    # Run tests
    print("\nRunning tests...")
    test_result = run_command(["uv", "run", "pytest", "backend/tests/", "-v"])
    if test_result != 0:
        exit_code = test_result
        print("[FAIL] Tests failed")
    else:
        print("[PASS] Tests passed")

    if exit_code == 0:
        print("\nAll quality checks passed!")
    else:
        print("\nSome quality checks failed. Please fix the issues above.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
