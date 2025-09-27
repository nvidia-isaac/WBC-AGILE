#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#=============================================================================
# CI INSTALL SCRIPT
#
# This script is used in CI/CD pipelines to install agile with all dependencies.
# It's similar to install_deps_local.sh but optimized for CI environments.
#=============================================================================

# Check if ISAACLAB_PATH is set
if [ -z "${ISAACLAB_PATH}" ]; then
    echo "Error: ISAACLAB_PATH environment variable is not set."
    echo "Please set it to the path of your Isaac Lab installation."
    echo "Example: export ISAACLAB_PATH=/workspace/isaaclab"
    exit 1
fi

echo "=== INSTALLING AGILE FOR CI ==="

# Install dependencies from pyproject.toml (excluding packages Isaac Sim provides)
echo "Installing dependencies from pyproject.toml..."
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/setup/install_deps_from_pyproject.py

# Install agile package itself without dependencies
echo "Installing agile package..."
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install --no-deps -e .

# Install custom rsl_rl if present (force reinstall to override any existing version)
if [ -d "agile/algorithms/rsl_rl" ]; then
    echo "Installing custom rsl_rl package..."
    # First uninstall any existing rsl_rl packages to avoid conflicts
    ${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip uninstall -y rsl_rl rsl-rl-lib 2>/dev/null || true
    # Install our custom version with --no-deps since agile already installed the deps
    ${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install --no-deps -e agile/algorithms/rsl_rl
    echo "Custom rsl_rl installed successfully"
fi

# Verify critical imports work
echo "Verifying installation..."
${ISAACLAB_PATH:?}/isaaclab.sh -p -c "import agile; import tensordict; import wandb; print('✅ All imports successful')"

echo "=== AGILE CI INSTALLATION COMPLETE ==="
