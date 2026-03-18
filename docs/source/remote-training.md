# Remote Training with OSMO

[OSMO](https://github.com/NVIDIA/OSMO) enables running training, evaluation, and sweep jobs on remote GPU clusters. The `run.py` CLI at the project root handles Docker image building, pushing, and workflow submission.

## Setup

### 1. OSMO CLI

Install OSMO and set up your Kubernetes cluster:

```bash
# See: https://github.com/NVIDIA/OSMO
osmo version
```

### 2. Compute Pool

Create an OSMO compute pool with GPU resources:

```bash
osmo pool create your-gpu-pool --platform <your-platform> --gpus <gpu-count>
osmo pool list
```

### 3. Container Registry

Set up access to a container registry:

```bash
docker login docker.io
```

### 4. Kubernetes Secrets

Configure credentials in your Kubernetes cluster:

```bash
# W&B credentials (required)
kubectl create secret generic wandb-credentials \
  --from-literal=wandb_pass=your_api_key \
  --from-literal=wandb_user=your-team-name

# Omniverse credentials (optional)
kubectl create secret generic omni-auth \
  --from-literal=omni_pass=your_password \
  --from-literal=omni_user=your_username
```

### 5. Run Configuration

Copy and customize the run configuration:

```bash
cd workflows/
cp run_config.example.yaml run_config.yaml
```

Update these fields:
- `image_name`: Your container registry path (e.g., `docker.io/myorg/agile`)
- `osmo_pools`: Your OSMO compute pool names

## Training

```bash
# Train with a fresh Docker image (use after code changes)
./run.py train --name my_experiment --task_name Velocity-T1-v0 --rebuild

# Reuse existing image (faster, when code hasn't changed)
./run.py train --name my_experiment_v2 --task_name Velocity-T1-v0 --use-existing

# Multiple seeds in parallel
./run.py train --name multi_seed --task_name Velocity-T1-v0 --seeds 0 42 1337 --rebuild

# Resume from checkpoint
./run.py train --name resumed_run --task_name Velocity-T1-v0 \
    --resume_checkpoint /path/to/model_5000.pt --rebuild

# Custom max iterations and project name
./run.py train --name long_run --task_name Velocity-T1-v0 \
    --max_iterations 50000 --project_name my-project --rebuild
```

Use `--rebuild` after code changes and `--use-existing` to reuse a previously built image. Run `./run.py train -h` for all options.

## Evaluation

```bash
# Evaluate latest checkpoint from a W&B training run
./run.py eval --name eval_test \
    --wandb_run your-team/project/run_id \
    --task_name Velocity-Height-G1-Dev-v0

# Evaluate specific checkpoints
./run.py eval --name multi_ckpt \
    --wandb_run your-team/project/run_id \
    --task_name Velocity-Height-G1-Dev-v0 \
    --checkpoints 5000,10000,15000

# Evaluate a local checkpoint
./run.py eval --name eval_local \
    --checkpoint_path /path/to/model_5000.pt \
    --task_name Velocity-Height-G1-Dev-v0

# With custom evaluation scenario
./run.py eval --name custom_eval \
    --wandb_run your-team/project/run_id \
    --task_name Velocity-Height-G1-Dev-v0 \
    --eval_config agile/algorithms/evaluation/configs/examples/multi_env_capability_test.yaml
```

## Sweeps

```bash
# Initialize sweep on W&B
python scripts/wandb_sweep/init_sweep.py --project_name my_sweep

# Deploy agents to OSMO
./run.py sweep --name sweep_experiment --sweep_name my_sweep --rebuild

# Deploy more agents with the same image
./run.py sweep --name sweep_experiment --sweep_name my_sweep --use-existing

# Deploy many agents at once
for i in {1..10}; do
  ./run.py sweep --name sweep_experiment --sweep_name my_sweep --use-existing
done
```

## Workflow Resources

| Workflow | CPU | GPU | Memory | Timeout |
|----------|-----|-----|--------|---------|
| Training | 6 | 1 | 60Gi | 7 days |
| Evaluation | 4 | 1 | 60Gi | 2 hours |
| Sweep | 16 | 1 | 100Gi | -- |

### Resource Planning

| Workload | Memory | GPUs | Time |
|----------|--------|------|------|
| Small training (< 1000 envs) | 50Gi | 1 | -- |
| Medium training (1000-4096 envs) | 100Gi | 1 | -- |
| Large training (4096-16384 envs) | 200Gi | 1-2 | -- |
| Quick eval (< 100 episodes) | 50Gi | 1 | 30min |
| Standard eval (100-1000 episodes) | 100Gi | 1 | 2hr |
| Comprehensive eval (> 1000 episodes) | 150Gi | 1 | 6hr |

## Docker Image

The `workflows/Dockerfile` builds on `nvcr.io/nvidia/isaac-lab:2.3.2`:

1. Installs Python dependencies into Isaac Lab's environment
2. Removes conflicting rsl_rl packages
3. Installs custom rsl_rl with TensorDict support
4. Verifies correct installation

```bash
# Build and test locally
docker build -f workflows/Dockerfile -t agile:test .
docker run --rm agile:test ${ISAACLAB_PATH}/isaaclab.sh -p scripts/verify_rsl_rl.py
```

## Manual Workflow Submission

You can also build and submit workflows directly without `run.py`:

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
  --set task_name=Velocity-T1-v0 \
  --set project_name=my_project \
  --set run_name=experiment_1 \
  --set wandb_pass=$WANDB_API_KEY \
  --set wandb_username=your-team
```

## Advanced Workflow Patterns

The OSMO workflow YAML files demonstrate several production patterns:

### Credential Management

Kubernetes secrets are mapped to environment variables in workflow configs:

```yaml
credentials:
  omni-auth:
    OMNI_PASS: omni_pass
    OMNI_USER: omni_user
  wandb:
    WANDB_API_KEY: wandb_pass
    WANDB_USERNAME: wandb_user
```

### Dynamic Script Injection

Entry scripts are generated dynamically using OSMO's Jinja templating:

```yaml
files:
  - path: /tmp/entry.sh
    contents: |
      CMD="${ISAACLAB_PATH}/isaaclab.sh -p scripts/train.py "
      {% if seed is defined %}
      CMD+="--seed {{seed}} "
      {% endif %}
```

### Conditional Checkpoint Handling

**Option A -- Bundle in Docker** (for local checkpoints):
```dockerfile
ARG RESUME_STAGE=yes
FROM base AS resume-yes
COPY checkpoints/ /workspace/agile/policy/resume
```

**Option B -- Download from W&B** (inside container):
```python
api = wandb.Api()
run = api.run(wandb_run_path)
file = run.file(f'model_{iteration}.pt')
file.download(root=checkpoint_dir)
```

### Dataset Outputs

OSMO automatically versions and stores outputs with content-addressable storage:

```yaml
outputs:
  - dataset:
      name: agile:{{workflow_id}}
      path: outputs
```

## Monitoring

```bash
osmo workflow logs <workflow-name> --follow   # Real-time logs
osmo workflow query <workflow-name>           # Status
osmo workflow list                            # List all
osmo workflow cancel <workflow-name>          # Cancel
osmo workflow port-forward <workflow-name> train --port 8080  # Debug
```

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
