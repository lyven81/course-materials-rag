#!/usr/bin/env python3
"""Script to format code using black and isort."""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> int:
    """Run a command and return its exit code."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=Path(__file__).parent.parent)
    return result.returncode


def main() -> int:
    """Format code using black and isort."""
    exit_code = 0

    # Run black
    black_result = run_command(
        ["uv", "run", "black", "backend/", "main.py", "scripts/"]
    )
    if black_result != 0:
        exit_code = black_result

    # Run isort
    isort_result = run_command(
        ["uv", "run", "isort", "backend/", "main.py", "scripts/"]
    )
    if isort_result != 0:
        exit_code = isort_result

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
