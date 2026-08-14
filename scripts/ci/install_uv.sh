#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download the uv binary into the job-local .uv-bin dir if it is not already on
# PATH. The caller is responsible for prepending .uv-bin to PATH so this and
# subsequent CI steps can invoke uv.
#
# Requires: UV_VERSION and CI_PROJECT_DIR in the environment.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  mkdir -p "${CI_PROJECT_DIR}/.uv-bin"
  curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C "${CI_PROJECT_DIR}/.uv-bin" --strip-components=1 \
      "uv-x86_64-unknown-linux-gnu/uv" "uv-x86_64-unknown-linux-gnu/uvx"
fi
uv --version
