# OSMO Workflows for Isaac Lab

Production-ready OSMO workflow configurations for reinforcement learning training with Isaac Lab.

## Overview

This directory contains complete OSMO workflow definitions for:
- **Training**: Multi-environment RL training with checkpoint management
- **Evaluation**: Deterministic scenario testing with HTML report generation
- **Hyperparameter Sweeps**: Parallel wandb sweep execution

These workflows demonstrate production patterns for running Isaac Lab workloads on OSMO, including credential management, dynamic script injection, checkpoint handling, and resource optimization.

## Files

- **`Dockerfile`**: Builds the training container
  - Based on `nvcr.io/nvidia/isaac-lab:2.3.1`
  - Installs dependencies into Isaac Lab's Python environment
  - Replaces Isaac Lab's rsl_rl with our custom version

- **`train_workflow.yaml`**: OSMO workflow for single training runs
- **`eval_workflow.yaml`**: OSMO workflow for evaluation jobs
- **`sweep_workflow.yaml`**: OSMO workflow for hyperparameter sweeps
- **`run_config.yaml`**: Run configuration for remote workflows (gitignored, created from example)
- **`run_config.example.yaml`**: Template for run configuration

## Features

### Training Workflow
- **Resume Support**: Continue training from saved checkpoints
- **Video Recording**: Automatic video capture every 200 iterations
- **Multi-Seed**: Run multiple seeds in parallel
- **Logging**: Wandb or Tensorboard integration

### Evaluation Workflow
- **Flexible Checkpoints**: Load from Wandb or bundle in Docker
- **Deterministic Testing**: Reproducible evaluation scenarios
- **HTML Reports**: Automatic visualization generation
- **Auto-Upload**: Results automatically uploaded to Wandb

### Sweep Workflow
- **Parallel Search**: Multiple agents per job
- **Wandb Integration**: Native sweep support
- **Scalable**: Easy to spawn many concurrent sweeps

## Prerequisites

### 1. OSMO Setup

Install OSMO and set up your Kubernetes cluster:
```bash
# Install OSMO CLI
# See: https://github.com/NVIDIA/OSMO

# Verify installation
osmo version
```

### 2. Compute Pool

Create an OSMO compute pool with GPU resources:
```bash
osmo pool create your-gpu-pool \
  --platform <your-platform> \
  --gpus <gpu-count>

# Verify pool
osmo pool list
```

### 3. Container Registry

Set up access to a container registry:
```bash
# Examples:
# - Docker Hub: docker.io/myorg
# - GitHub Container Registry: ghcr.io/myorg
# - Azure Container Registry: yourregistry.azurecr.io

docker login docker.io
```

### 4. Kubernetes Secrets

Configure credentials in your Kubernetes cluster:

```bash
# Wandb credentials (required)
kubectl create secret generic wandb-credentials \
  --from-literal=wandb_pass=your_api_key \
  --from-literal=wandb_user=your-team-name

# Omniverse credentials (optional - only if using Isaac Sim features)
kubectl create secret generic omni-auth \
  --from-literal=omni_pass=your_password \
  --from-literal=omni_user=your_username
```

## Quick Start

### Step 1: Configure

Copy and customize the run configuration:

```bash
cd workflows/
cp run_config.example.yaml run_config.yaml
vim run_config.yaml
```

Update these fields:
- `image_name`: Your container registry path (e.g., `docker.io/myorg/agile`)
- `osmo_pools`: Your OSMO compute pool names

### Step 2: Deploy a run

#### Step 2.1: Deploy via `run.py`
The `run.py` script in the project root handles Docker image building and OSMO workflow submission:

```bash
# Deploy with new Docker image
./run.py train --name my_experiment --task_name Standing-G1-v0 --rebuild

# Reuse existing image (faster, use after code hasn't changed)
./run.py train --name my_experiment_v2 --task_name Standing-G1-v0 --use-existing

# Multiple seeds in parallel
./run.py train --name multi_seed_exp --task_name Standing-G1-v0 --seeds 0 42 1337 --rebuild
```

#### Step 2.2 (Alternative): Submit Manually

You can also build and submit workflows directly:

```bash
# Build the image
docker build -f workflows/Dockerfile -t docker.io/myorg/agile:latest .
docker push docker.io/myorg/agile:latest

# Submit workflow
export WANDB_API_KEY=your_key
osmo workflow submit workflows/train_workflow.yaml \
  --pool=your-gpu-pool \
  --set workflow_name=my_first_training \
  --set image=docker.io/myorg/agile:latest \
  --set task_name=Standing-G1-v0 \
  --set project_name=my_project \
  --set run_name=experiment_1 \
  --set wandb_pass=$WANDB_API_KEY \
  --set wandb_username=your-team
```

## Workflow Details

### Training Workflow (`train_workflow.yaml`)

**Resources:**
- CPU: 6 cores
- GPU: 1
- Memory: 60Gi (sufficient for Isaac Sim physics)
- Storage: 60Gi (datasets, checkpoints, videos)
- Timeout: 7 days

**Key Parameters:**
```bash
--set task_name=Standing-G1-v0      # Isaac Lab task
--set num_envs=4096                  # Parallel environments
--set max_iterations=20000           # Training iterations
--set logger=wandb                   # wandb or tensorboard
--set seed=42                        # Random seed (optional)
```

**Resume Training:**
```bash
--set resume=True \
--set RESUME_PATH=/path/to/checkpoints \
--set RESUME_CHECKPOINT=model_10000.pt
```

### Evaluation Workflow (`eval_workflow.yaml`)

**Resources:**
- CPU: 4 cores
- GPU: 1
- Memory: 60Gi
- Storage: 100Gi
- Timeout: 2 hours

**Evaluate from Wandb:**
```bash
./run.py eval --name eval_test --wandb_run your-team/project/run_id --task_name Velocity-Height-G1-Dev-v0

# Or manually:
osmo workflow submit workflows/eval_workflow.yaml \
  --pool=your-gpu-pool \
  --set workflow_name=eval_experiment \
  --set image=docker.io/myorg/agile:latest \
  --set task_name=Velocity-Height-G1-Dev-v0 \
  --set wandb_run_path=your-team/project/run_id \
  --set iteration=10000 \
  --set use_local_checkpoint=false
```

**Evaluate Local Checkpoint:**
```bash
./run.py eval --name eval_local --checkpoint_path /path/to/model.pt --task_name Velocity-Height-G1-Dev-v0

# Or manually: bundle checkpoint in Docker image first
docker build -f workflows/Dockerfile \
  --build-arg RESUME_STAGE=yes \
  --build-arg RESUME_PATH=path/to/checkpoint/ \
  --build-arg RESUME_CHECKPOINT=model.pt \
  -t docker.io/myorg/agile:eval .

osmo workflow submit workflows/eval_workflow.yaml \
  --set use_local_checkpoint=true
```

### Sweep Workflow (`sweep_workflow.yaml`)

**Resources:**
- CPU: 16 cores
- GPU: 1
- Memory: 100Gi

**Usage:**
```bash
# Step 1: Initialize sweep on Wandb
python scripts/wandb_sweep/init_sweep.py --project_name my_sweep

# Step 2: Deploy to OSMO
./run.py sweep --name sweep_experiment --sweep_name my_sweep --rebuild

# Step 3: Deploy more agents with the same image
./run.py sweep --name sweep_experiment --sweep_name my_sweep --use-existing

# Or submit multiple agents at once
for i in {1..10}; do
  ./run.py sweep --name sweep_experiment --sweep_name my_sweep --use-existing
done
```

## Advanced Patterns

These workflows demonstrate production OSMO patterns:

### 1. Multi-Service Credential Management

```yaml
credentials:
  omni-auth:
    OMNI_PASS: omni_pass
    OMNI_USER: omni_user
  wandb:
    WANDB_API_KEY: wandb_pass
    WANDB_USERNAME: wandb_user
```

Maps Kubernetes secrets to environment variables securely.

### 2. Dynamic Script Injection

```yaml
files:
  - path: /tmp/entry.sh
    contents: |
      # Complex bash script with OSMO templating
      CMD="${ISAACLAB_PATH}/isaaclab.sh -p scripts/train.py "
      {% if seed is defined %}
      CMD+="--seed {{seed}} "
      {% endif %}
```

Generates entry scripts dynamically based on workflow parameters.

### 3. Conditional Checkpoint Handling

**Option A:** Bundle in Docker (for local checkpoints)
```dockerfile
ARG RESUME_STAGE=yes
FROM base AS resume-yes
COPY checkpoints/ /workspace/agile/policy/resume
```

**Option B:** Download from Wandb (inside container)
```bash
# Python script downloads checkpoint during workflow execution
api = wandb.Api()
run = api.run(wandb_run_path)
file = run.file(f'model_{iteration}.pt')
file.download(root=checkpoint_dir)
```

### 4. Dataset Outputs

```yaml
outputs:
  - dataset:
      name: agile:{{workflow_id}}
      path: outputs
```

OSMO automatically versions and stores outputs with content-addressable storage.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Image pull failed | Verify `image_name` in config and registry credentials |
| Pool not found | Run `osmo pool create` and update `osmo_pools` in config |
| Credential errors | Check Kubernetes secrets: `kubectl get secrets` |
| Out of memory | Reduce `num_envs` or increase pool memory allocation |
| Timeout | Increase `exec_timeout` in workflow YAML or reduce `max_iterations` |
| Workflow stuck | Check logs: `osmo workflow logs <workflow-name>` |
| `ModuleNotFoundError: tensordict` | Rebuild Docker image with `--rebuild` |
| Wrong rsl_rl version | Run `scripts/verify_rsl_rl.py` to check |
| Docker build fails | Check `agile/algorithms/rsl_rl/` exists |
| Isaac Sim init failures | Wrapper auto-retries (2 attempts with 10s delay) |

## Monitoring

```bash
# View real-time logs
osmo workflow logs <workflow-name> --follow

# Check workflow status
osmo workflow query <workflow-name>

# List all workflows
osmo workflow list

# Cancel workflow
osmo workflow cancel <workflow-name>

# Port-forward for interactive debugging
osmo workflow port-forward <workflow-name> train --port 8080
```

## Debugging

To verify the Docker image locally before deployment:

```bash
# Build the image
docker build -f workflows/Dockerfile -t agile:test .

# Run verification
docker run --rm agile:test ${ISAACLAB_PATH}/isaaclab.sh -p scripts/verify_rsl_rl.py
```

## Resource Planning

### Training Workloads
- **Small tasks** (< 1000 envs): 50Gi memory, 1 GPU
- **Medium tasks** (1000-4096 envs): 100Gi memory, 1 GPU
- **Large tasks** (4096-16384 envs): 200Gi memory, 1-2 GPUs

### Evaluation Workloads
- **Quick eval** (< 100 episodes): 50Gi memory, 1 GPU, 30min
- **Standard eval** (100-1000 episodes): 100Gi memory, 1 GPU, 2hr
- **Comprehensive eval** (> 1000 episodes): 150Gi memory, 1 GPU, 6hr

## Additional Resources

- [OSMO Documentation](https://nvidia.github.io/OSMO)
- [OSMO GitHub Repository](https://github.com/NVIDIA/OSMO)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)
- [Weights & Biases Documentation](https://docs.wandb.ai)
