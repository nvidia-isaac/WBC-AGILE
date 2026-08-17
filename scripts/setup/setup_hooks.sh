#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

echo "Installing pre-commit hooks..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

# Install pre-commit
uv pip install pre-commit

# Install the git hooks
pre-commit install

echo "Pre-commit hooks installed successfully!"
echo "The hooks will run automatically on each commit."
echo "To run the hooks manually on all files, use: pre-commit run --all-files"
