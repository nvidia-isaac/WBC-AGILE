#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#=============================================================================
# LOCAL DEVELOPMENT INSTALL SCRIPT
#
# This script is for LOCAL development on your machine.
# It installs ALL dependencies AND packages needed to run the agile extension.
#
# For Docker/OSMO deployment, use install_deps.sh instead.
#=============================================================================

# Check if ISAACLAB_PATH is set
if [ -z "${ISAACLAB_PATH}" ]; then
    echo "Error: ISAACLAB_PATH environment variable is not set."
    echo "Please set it to the path of your Isaac Lab installation."
    echo "Example: export ISAACLAB_PATH=/path/to/isaac_lab"
    exit 1
fi

echo "=== INSTALLING AGILE AND DEPENDENCIES ==="
# First install dependencies from pyproject.toml (excluding torch/numpy that Isaac provides)
echo "Installing dependencies from pyproject.toml..."
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/setup/install_deps_from_pyproject.py

echo "=== INSTALLING AGILE PACKAGE ==="
# Install agile package itself with --no-deps
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install --no-deps -e .

# Install custom rsl_rl if present (force reinstall to override any existing version)
if [ -d "agile/algorithms/rsl_rl" ]; then
    echo "Installing custom rsl_rl package..."
    # First uninstall any existing rsl_rl packages to avoid conflicts
    # Note: The package might be named rsl_rl or rsl-rl-lib
    ${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip uninstall -y rsl_rl rsl-rl-lib 2>/dev/null || true
    # Install our custom version (this will be installed as rsl-rl-lib but imports as rsl_rl)
    ${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install --no-deps -e agile/algorithms/rsl_rl
    echo "Custom rsl_rl installed successfully"
fi

# Check that IsaacLab is using the correct version.
if [ -d ${ISAACLAB_PATH}/.git ]; then
  expected_isaac_lab_tag="v2.3.0"
  if ! git -C ${ISAACLAB_PATH} tag -l "${expected_isaac_lab_tag}" | grep -q "${expected_isaac_lab_tag}"; then
      echo "Error: IsaacLab does not have the expected tag."
      echo "Expected tag: ${expected_isaac_lab_tag}"
      echo "Please checkout the correct version: git -C ${ISAACLAB_PATH} checkout ${expected_isaac_lab_tag}"
      exit 1
  fi
fi

echo "=== LOCAL DEVELOPMENT INSTALLATION COMPLETE ==="
echo "🎉 All dependencies and packages installed successfully!"
echo "You can now set up the pre-commit hooks by running: ./scripts/setup/setup_hooks.sh"
