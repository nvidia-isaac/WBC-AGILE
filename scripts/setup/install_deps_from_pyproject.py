#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper script to extract and install dependencies from pyproject.toml
This ensures all install scripts use the same dependency definitions.
"""

import subprocess
import sys
from pathlib import Path

# Try to import tomllib (Python 3.11+) or fall back to toml
try:
    import tomllib

    def load_toml(f):
        return tomllib.load(f)
except ImportError:
    try:
        import toml

        def load_toml(f):
            return toml.load(f)
    except ImportError:
        print("Error: Neither tomllib nor toml package found. Installing toml...")
        subprocess.run([sys.executable, "-m", "pip", "install", "toml"])
        import toml

        def load_toml(f):
            return toml.load(f)


def get_dependencies_from_pyproject(exclude_packages=None):
    """Read dependencies from pyproject.toml."""
    exclude_packages = exclude_packages or []

    # Read pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb" if "tomllib" in sys.modules else "r") as f:
        data = load_toml(f)

    # Get dependencies
    deps = data.get("project", {}).get("dependencies", [])

    # Filter out excluded packages (like torch, which Isaac Sim provides)
    filtered_deps = []
    for dep in deps:
        # Extract package name (before any version specifier)
        pkg_name = dep.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].strip()
        if pkg_name.lower() not in [ex.lower() for ex in exclude_packages]:
            filtered_deps.append(dep)

    return filtered_deps


def install_dependencies(deps):
    """Install dependencies using pip."""
    if not deps:
        print("No dependencies to install")
        return 0

    # Use the current Python interpreter's pip to avoid version mismatches
    cmd = [sys.executable, "-m", "pip", "install"] + deps
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    # Packages that Isaac Sim provides - don't reinstall these
    exclude = ["torch", "numpy"]  # Isaac Sim provides these

    # Get dependencies
    deps = get_dependencies_from_pyproject(exclude_packages=exclude)

    print(f"Installing {len(deps)} dependencies from pyproject.toml:")
    for dep in deps:
        print(f"  - {dep}")

    # Install them
    sys.exit(install_dependencies(deps))
