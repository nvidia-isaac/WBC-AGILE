# WBC-AGILE training image based on NVIDIA Isaac Lab 2.3.1
# Isaac Sim 5.1.0 + Isaac Lab 2.3.1 + Python 3.11
#
# Build:
#   docker build -t wbc-agile:latest -f docker/Dockerfile .
#
# Run (single GPU test):
#   docker run --rm --gpus '"device=0"' --network=host \
#     -v $(pwd)/logs:/workspace/WBC-AGILE/logs \
#     -v $(pwd)/outputs:/workspace/WBC-AGILE/outputs \
#     wbc-agile:latest \
#     --task Velocity-G1-History-v0 --num_envs 2048 --max_iterations 3000
#
# Run (8 GPU):
#   docker run --rm --gpus all --network=host \
#     -v $(pwd)/logs:/workspace/WBC-AGILE/logs \
#     -v $(pwd)/outputs:/workspace/WBC-AGILE/outputs \
#     wbc-agile:latest

FROM nvcr.io/nvidia/isaac-lab:2.3.1

# Accept EULA
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=yes

# Isaac Sim env vars
ENV CARB_APP_PATH=/isaac-sim/kit
ENV EXP_PATH=/isaac-sim/apps
ENV ISAAC_PATH=/isaac-sim
ENV ISAACLAB_PATH=/workspace/isaaclab

# Python path
ENV PYTHON=/isaac-sim/kit/python/bin/python3.11

# ---- Remove bundled old torch/torchvision from ml_archive to avoid CUDA symbol conflicts ----
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch* \
           /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia \
           /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torchvision*

# ---- Install PyTorch (CUDA 12.4, compatible with driver 570.x) ----
RUN ${PYTHON} -m pip install --timeout 600 --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ---- Uninstall system rsl_rl (3.0.1) that conflicts with WBC-AGILE's fork ----
RUN ${PYTHON} -m pip uninstall -y rsl-rl-lib

# ---- Copy WBC-AGILE code ----
COPY . /workspace/WBC-AGILE
WORKDIR /workspace/WBC-AGILE

# ---- Install WBC-AGILE's custom rsl_rl fork (v2.3.3) ----
RUN cd /workspace/WBC-AGILE/agile/algorithms/rsl_rl && \
    ${PYTHON} -m pip install --no-deps --no-build-isolation -e .

# ---- Install agile package (no-deps to avoid pulling torch again) ----
RUN ${PYTHON} -m pip install --no-deps --no-build-isolation -e .

# ---- Install remaining small dependencies ----
RUN ${PYTHON} -m pip install --timeout 300 --no-cache-dir \
    toml \
    wandb==0.22.2 \
    protobuf==3.20.3 \
    tqdm \
    pyyaml \
    "gym>=0.21.0" \
    tensordict==0.8.3 \
    && echo "Dependencies installed"

# ---- Source Isaac Sim python env on every bash invocation ----
RUN echo 'source /isaac-sim/setup_python_env.sh' >> /root/.bashrc

# ---- Entrypoint ----
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default: 8 GPU training
CMD []
