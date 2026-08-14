# syntax=docker.io/docker/dockerfile:1.7-labs

# Build arguments for checkpoint handling (must be before any FROM)
ARG RESUME_CHECKPOINT
ARG RESUME_STAGE=no

###############################
# Base Stage - Isaac Lab v3.0.0-beta2
###############################
# Uses the published Isaac Lab beta2 image and installs AGILE into a locked uv environment.
# x86_64 only - the OSMO GPU pools are x86_64.
FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2 AS base

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

ENV ISAACLAB_PATH=/workspace/isaaclab
ENV UV_CACHE_DIR=/opt/uv-cache
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/workspace/agile/.venv
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV TERM=xterm
USER root

# System dependencies used by AGILE workflows.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git-lfs \
    patch && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix potential Vulkan conflicts.
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

WORKDIR /workspace/agile

###############################
# Install AGILE dependencies from the locked uv environment
###############################
# Only the dependency manifests are copied here so this expensive layer is keyed
# solely on the locked dependency set. Editing AGILE source (or README.md) no
# longer busts it. The BuildKit cache mount keeps uv's wheel cache on the builder
# across builds, so even when the lock *does* change uv reuses already-downloaded
# wheels (torch CUDA, isaacsim, ...) instead of re-downloading them. The cache
# mount lives outside the image layers, so it never bloats the image and no
# `uv cache clean` is needed (removing the mount without restoring a clean would
# bake the cache into a layer, so keep the two together).
COPY --chmod=0644 pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/opt/uv-cache \
    uv sync --frozen --no-dev --no-install-project

###############################
# Install AGILE source
###############################
COPY \
  --exclude=.git/ \
  --exclude=.venv/ \
  --exclude=.vscode/ \
  --exclude=wandb/ \
  --exclude=.mypy_cache/ \
  --exclude=.pytest_cache/ \
  --exclude=build/ \
  --exclude=logs/ \
  --exclude=outputs/ \
  --exclude=workflows/ \
  . /workspace/agile
RUN --mount=type=cache,target=/opt/uv-cache \
    uv sync --frozen --no-dev

# Verify and apply AGILE's public rsl-rl-lib compatibility patch during image build.
RUN uv run --frozen scripts/verify_rsl_rl.py

###############################
# Resume Policy Stage (conditionally named)
###############################
# Use this stage if no resume policy is provided:
FROM base AS resume-no

# Use this stage if a resume policy is provided:
FROM base AS resume-yes
ARG RESUME_PATH=agile/data/policy/
ARG RESUME_CHECKPOINT
COPY ${RESUME_PATH} /workspace/agile/policy/resume
ENV RESUME_PATH=/workspace/agile/policy/resume
ENV RESUME_CHECKPOINT=${RESUME_CHECKPOINT}

###############################
# Final Stage (selects the correct resume stage)
###############################
FROM resume-${RESUME_STAGE}
