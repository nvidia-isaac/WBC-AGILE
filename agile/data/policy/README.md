# Pre-trained Policies

This directory contains pre-trained policies for locomotion tasks.

## Directory Structure

```
policy/
├── velocity_g1/              # G1 - Velocity tracking (TorchScript)
├── velocity_height_g1/       # G1 - Velocity + height map (Checkpoints)
└── velocity_t1/              # T1 - Velocity tracking (TorchScript)
```

## Policy Formats

### TorchScript (`.pt` + `.yaml`)
Exported policies ready for deployment. Self-contained with normalizer included.
- **velocity_g1/** - `unitree_g1_velocity_history.pt`
- **velocity_t1/** - `booster_t1_velocity_v0.pt`

### Checkpoint (`.pt` only)
Training checkpoints that support recurrent policies and batched evaluation.
- **velocity_height_g1/** - Recurrent student policy and TorchScript teacher

## Available Policies

| Policy | Task | Command | Format | Description |
|--------|------|--------|--------|-------------|
| `velocity_g1/unitree_g1_velocity_history.pt` | `Velocity-G1-History-v0` | $\mathrm{v}_x$, $\mathrm{v}_y$, $\omega_z$ |TorchScript | History-based |
| `velocity_height_g1/unitree_g1_velocity_height_teacher.pt` | `Velocity-Height-G1-v0` | $\mathrm{v}_x$, $\mathrm{v}_y$, $\omega_z$ | TorchScript | Privilged teacher  |
| `velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt` | `Velocity-Height-G1-Distillation-Recurrent-v0` | $\mathrm{v}_x$, $\mathrm{v}_y$, $\omega_z$, $h_{root}$|Checkpoint | Recurrent LSTM student |
| `velocity_t1/booster_t1_velocity_v0.pt` | `Velocity-T1-v0` | $\mathrm{v}_x$, $\mathrm{v}_y$, $\omega_z$, $h_{root}$ | TorchScript | History-based |

Note: Root linear velocity is considered privileged information, as accurate estimation usually requires additional hardware during deployment. Only the velocity height teacher policy accesses this information during training and deployment; all other policies do not rely on it and are suitable for direct deployment on real robots. Additionally, velocity height policies provided are tuned for improved command tracking performance. The velocity height teacher policy is also a nice consideration in just simulation since it observes privileged linear velocity information and will perform better in velocity tracking.

## Usage

```bash
# TorchScript policies (auto-detected)
python scripts/eval.py --task Velocity-G1-History-v0 \
    --checkpoint agile/data/policy/velocity_g1/unitree_g1_velocity_history.pt

# Checkpoint policies
python scripts/eval.py --task Velocity-Height-G1-Distillation-Recurrent-v0 \
    --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt
```

The evaluation script automatically:
- Loads TorchScript for non-recurrent policies (fast inference)
- Falls back to checkpoint loading for recurrent policies (supports batched evaluation)
- Exports policies to `exported/` folder (TorchScript + ONNX)

## Notes

- **Recurrent policies:** Always use checkpoint format for batched evaluation
- **YAML files:** Required for TorchScript policies deployment in mujoco and real, contain task and architecture configs
