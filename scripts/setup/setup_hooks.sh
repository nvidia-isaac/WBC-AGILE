# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/bin/bash
set -e

echo "Installing pre-commit hooks..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed or not in PATH"
    exit 1
fi

# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

echo "Pre-commit hooks installed successfully!"
echo "The hooks will run automatically on each commit."
echo "To run the hooks manually on all files, use: pre-commit run --all-files"
