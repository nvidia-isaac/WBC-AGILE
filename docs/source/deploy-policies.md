# Deploy Policies

## LEAPP and Isaac ROS Deploy

To deploy policies sim-to-sim and sim-to-real this project uses LEAPP and Isaac ROS Deploy. This allows you to deploy your policy as fast and efficient C++ code without rewriting any code.

LEAPP packages a policy into a self-contained artifact, a LEAPP bundle, which can then be executed by a LEAPP runtime.

[Isaac ROS Deploy](https://isaac.gitlab-master-pages.nvidia.com/isaac/repositories_and_packages/isaac_ros_deploy/index.html)
provides LEAPP runtimes for real robots and simulation-in-the-loop deployment.
The runtimes execute policies through C++, providing deterministic,
low-latency deployment without requiring policy code changes.

For a complete Unitree G1 runtime walkthrough, see the
[Isaac ROS AGILE G1 deployment tutorial](https://isaac.gitlab-master-pages.nvidia.com/isaac/repositories_and_packages/isaac_ros_deploy/tutorials/deploy_agile.html).

## Export a Policy with LEAPP

Use `scripts/export_policy_leapp.py` to export a supported AGILE training
checkpoint. Choose the task and checkpoint that match the policy you want to
deploy:

```bash
uv run scripts/export_policy_leapp.py \
    --task <task-name> \
    --checkpoint /path/to/checkpoint.pt \
    --export_save_path /path/to/leapp-bundles \
    --disable_graph_visualization
```

The command creates a directory named for the task. Keep that entire directory
together: its YAML configuration references the ONNX model and, when present,
the safetensors state artifact through relative paths.

## Example: Export the G1 Velocity Policy

The repository ships a `Velocity-G1-History-v0` checkpoint that can be exported for
the Isaac ROS Controller Manager:

```bash
export AGILE_LEAPP_EXPORT_DIR="$HOME/agile-leapp/velocity-g1"
uv run scripts/export_policy_leapp.py \
    --task Velocity-G1-History-v0 \
    --checkpoint agile/data/policy/velocity_g1/unitree_g1_velocity_history_state_dict.pt \
    --export_save_path "${AGILE_LEAPP_EXPORT_DIR}" \
    --disable_graph_visualization
```

Use `${AGILE_LEAPP_EXPORT_DIR}/Velocity-G1-History-v0/Velocity-G1-History-v0.yaml` as the
LEAPP configuration path. Copy the complete `Velocity-G1-History-v0` directory when
moving the bundle to the runtime machine.
