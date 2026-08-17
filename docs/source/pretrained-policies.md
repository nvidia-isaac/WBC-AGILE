# Pre-trained Policies

Pre-trained policies are available in `agile/data/policy/`.

## Directory Structure

```
policy/
  velocity_g1/
    unitree_g1_velocity_history_state_dict.pt
    unitree_g1_velocity_history_torchscript.pt
    leapp/Velocity-G1-v0/
      Velocity-G1-v0.onnx
      Velocity-G1-v0.yaml
      Velocity-G1-v0_initial_values.safetensors
  velocity_height_g1/
    unitree_g1_velocity_height_history_state_dict.pt
    unitree_g1_velocity_height_history_torchscript.pt
    unitree_g1_velocity_height_teacher_state_dict.pt
    unitree_g1_velocity_height_teacher_torchscript.pt
    leapp/Velocity-Height-G1-History-v0/
      Velocity-Height-G1-History-v0.onnx
      Velocity-Height-G1-History-v0.yaml
      Velocity-Height-G1-History-v0_initial_values.safetensors
  velocity_t1/
    booster_t1_velocity_v0.pt
    booster_t1_velocity_v0_state_dict.pt
    leapp/Velocity-T1-v0/
      Velocity-T1-v0.onnx
      Velocity-T1-v0.yaml
      Velocity-T1-v0_initial_values.safetensors
```

## Available Policies

| Policy | Task | Commands | Included formats |
|--------|------|----------|------------------|
| G1 velocity history | `Velocity-G1-History-v0` | v_x, v_y, w_z | TorchScript, rsl_rl checkpoint, LEAPP |
| G1 velocity-height history | `Velocity-Height-G1-History-v0` | v_x, v_y, w_z, height | TorchScript, rsl_rl checkpoint, LEAPP |
| G1 velocity-height teacher | `Velocity-Height-G1-Teacher-v0` | v_x, v_y, w_z, height | TorchScript, rsl_rl checkpoint |
| T1 velocity | `Velocity-T1-v0` | v_x, v_y, w_z | TorchScript, rsl_rl checkpoint, LEAPP |

## Policy Formats

- **TorchScript** (`.pt`): Exported policies ready for deployment. Load with `torch.jit.load()`.
- **rsl_rl checkpoint** (`.pt` only): Training checkpoints containing rsl_rl 5.x keys such as `actor_state_dict` or `student_state_dict`, `optimizer_state_dict`, and `iter`. Required for resuming training or batched evaluation.
- **LEAPP bundle** (`.yaml`, `.onnx`, and `.safetensors`): Self-contained policy graph, model, and feedback initial values for deployment and Sim2MuJoCo evaluation.

## Usage

```bash
# T1 TorchScript policy
uv run scripts/eval.py --task Velocity-T1-v0 \
    --checkpoint agile/data/policy/velocity_t1/booster_t1_velocity_v0.pt

# G1 rsl_rl checkpoint
uv run scripts/eval.py --task Velocity-G1-History-v0 \
    --checkpoint agile/data/policy/velocity_g1/unitree_g1_velocity_history_state_dict.pt

# G1 velocity + height training checkpoint
uv run scripts/eval.py --task Velocity-Height-G1-Teacher-v0 \
    --checkpoint agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher_state_dict.pt
```

The evaluation script automatically detects TorchScript and rsl_rl checkpoint formats.
