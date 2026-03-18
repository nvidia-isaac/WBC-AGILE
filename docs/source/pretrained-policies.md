# Pre-trained Policies

Pre-trained policies are available in `agile/data/policy/`.

## Directory Structure

```
policy/
  velocity_g1/              # G1 - Velocity tracking (TorchScript)
  velocity_height_g1/       # G1 - Velocity + height (TorchScript + Checkpoint)
    exported/               # Exported student policy (TorchScript + ONNX)
    *_teacher.pt            # Teacher policy (TorchScript)
    *_student.pt            # Student policy (TorchScript)
    *_student_checkpoint.pt # Student training checkpoint (State dict)
  velocity_t1/              # T1 - Velocity tracking (TorchScript)
```

## Available Policies

| Policy | Task | Commands | Format | Description |
|--------|------|----------|--------|-------------|
| `velocity_g1/unitree_g1_velocity_history.pt` | `Velocity-G1-History-v0` | v_x, v_y, w_z | TorchScript | History-based |
| `velocity_height_g1/unitree_g1_velocity_height_teacher.pt` | `Velocity-Height-G1-v0` | v_x, v_y, w_z | TorchScript | Privileged teacher |
| `velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt` | `Velocity-Height-G1-Distillation-Recurrent-v0` | v_x, v_y, w_z, h | TorchScript | Recurrent LSTM student |
| `velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt` | `Velocity-Height-G1-Distillation-Recurrent-v0` | v_x, v_y, w_z, h | State dict | Training checkpoint |
| `velocity_t1/booster_t1_velocity_v0.pt` | `Velocity-T1-v0` | v_x, v_y, w_z | TorchScript | History-based |

```{note}
Root linear velocity is considered privileged information, as accurate estimation usually requires additional hardware during deployment. Only the velocity-height teacher policy accesses this information; all other policies are suitable for direct deployment on real robots. The velocity-height policies are tuned for improved command tracking performance. The teacher policy is also useful in simulation since it observes privileged linear velocity and performs better at velocity tracking.
```

## Policy Formats

- **TorchScript** (`.pt` + `.yaml`): Exported policies ready for deployment. Self-contained with normalizer included. Load with `torch.jit.load()`.
- **State dict** (`.pt` only): Training checkpoints containing `model_state_dict`, `optimizer_state_dict`, and `iter`. Load with `torch.load()`. Required for resuming training or batched evaluation.
- **ONNX** (`.onnx`): For hardware inference engines.
- **YAML files**: Required for TorchScript policy deployment in MuJoCo and on real hardware, containing task and architecture configs.

## Usage

```bash
# TorchScript policies (auto-detected)
python scripts/eval.py --task Velocity-G1-History-v0 \
    --checkpoint agile/data/policy/velocity_g1/unitree_g1_velocity_history.pt

# State dict checkpoint (for batched evaluation / resuming training)
python scripts/eval.py --task Velocity-Height-G1-Distillation-Recurrent-v0 \
    --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student_checkpoint.pt
```

The evaluation script automatically detects the format, loads accordingly, and exports policies to `exported/` (TorchScript + ONNX).

## I/O Descriptor Export

Export observation and action space descriptors for deployment:

```bash
python scripts/export_IODescriptors.py --task Velocity-T1-v0 --output_dir .
```

Generates a YAML file describing the model's input/output spaces, used by the sim-to-MuJoCo framework and deployment pipelines.
