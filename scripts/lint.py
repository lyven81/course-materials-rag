#!/usr/bin/env python3
"""Script to run linting checks using flake8 and mypy."""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=Path(__file__).parent.parent)
    return result.returncode


def main() -> int:
    """Run linting checks."""
    exit_code = 0

    # Run flake8
    flake8_result = run_command(
        ["uv", "run", "flake8", "backend/", "main.py", "scripts/"]
    )
    if flake8_result != 0:
        exit_code = flake8_result
        print("❌ Flake8 linting failed")
    else:
        print("✅ Flake8 linting passed")

    # Run mypy
    mypy_result = run_command(["uv", "run", "mypy", "backend/", "main.py", "scripts/"])
    if mypy_result != 0:
        exit_code = mypy_result
        print("❌ MyPy type checking failed")
    else:
        print("✅ MyPy type checking passed")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
