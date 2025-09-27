# syntax=docker.io/docker/dockerfile:1.7-labs

# Build arguments for checkpoint handling (must be before any FROM)
ARG RESUME_CHECKPOINT
ARG RESUME_STAGE=no

###############################
# Base Stage
###############################
FROM nvcr.io/nvidia/isaac-lab:2.3.0 AS base

# Fix potential Vulkan conflicts
RUN if [ -e "/usr/share/vulkan" ] && [ -e "/etc/vulkan" ]; then \
      mv /usr/share/vulkan /usr/share/vulkan_hidden; \
    fi

WORKDIR /workspace/agile

###############################
# Install Python Dependencies
###############################
# IMPORTANT: We must install directly into Isaac Lab's Python environment
# using ${ISAACLAB_PATH}/isaaclab.sh -p to ensure compatibility
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    tensordict==0.8.3 \
    datasets==2.19.0 \
    wandb==0.22.2 \
    importlib_metadata \
    orjson \
    pandas \
    plotly \
    jinja2 \
    seaborn \
    matplotlib

###############################
# Install Custom rsl_rl Package
###############################
# Remove any pre-existing rsl_rl packages from Isaac Lab
# These would conflict with our custom version that has TensorDict support
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y rsl_rl rsl-rl-lib 2>/dev/null || true

# Copy and install our custom rsl_rl package
COPY scripts/setup/install_deps.sh scripts/setup/install_deps.sh
COPY pyproject.toml .
COPY agile/algorithms/rsl_rl agile/algorithms/rsl_rl
RUN ./scripts/setup/install_deps.sh

# Verify correct installation
COPY scripts/verify_rsl_rl.py scripts/verify_rsl_rl.py
RUN ${ISAACLAB_PATH}/isaaclab.sh -p scripts/verify_rsl_rl.py

# Install the rest of the code
COPY \
  --exclude=.git/ \
  --exclude=.vscode/ \
  --exclude=wandb/ \
  --exclude=.mypy_cache/ \
  --exclude=.pytest_cache/ \
  --exclude=build/ \
  --exclude=scripts/setup/ \
  --exclude=logs/ \
  --exclude=outputs/ \
  --exclude=pyproject.toml \
  --exclude=workflows/ \
  . /workspace/agile

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
