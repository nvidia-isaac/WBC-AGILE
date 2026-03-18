# AGILE

**A Generic Isaac-Lab based Engine** for humanoid loco-manipulation learning.

AGILE is an [external Isaac Lab project](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html) that showcases how to build a full RL training pipeline on top of Isaac Lab, from task design and training to evaluation and sim-to-real deployment.

<table align="center">
  <tr>
    <th colspan="2">Booster T1 – Stand-Up</th>
    <th colspan="2">Booster T1 – Velocity Tracking</th>
  </tr>
  <tr>
    <td align="center"><img src="booster_t1_stand_up_sim2sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="booster_t1_stand_up_sim2real.gif" width="240"><br><em>Real</em></td>
    <td align="center"><img src="booster_t1_vel_sim2sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="booster_t1_vel_sim2real.gif" width="240"><br><em>Real</em></td>
  </tr>
  <tr>
    <th colspan="2">Unitree G1 – Velocity-Height Tracking</th>
    <th colspan="2">Unitree G1 – Sit-Down / Stand-Up</th>
  </tr>
  <tr>
    <td align="center"><img src="unitree_g1_vel_height_sim2sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="unitree_g1_vel_height_sim2real.gif" width="240"><br><em>Real</em></td>
    <td align="center"><img src="unitree_g1_updown_sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="unitree_g1_updown.gif" width="240"><br><em>Real</em></td>
  </tr>
  <tr>
    <th colspan="2">Unitree G1 – Teleoperation</th>
    <th colspan="2">Unitree G1 – Dancing</th>
  </tr>
  <tr>
    <td align="center"><img src="locomanipulation-g1-sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="unitree_g1_teleop.gif" width="240"><br><em>Real</em></td>
    <td align="center"><img src="unitree_g1_dancing_sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="unitree_g1_dancing.gif" width="240"><br><em>Real</em></td>
  </tr>
</table>

---

## Key Features

- **Multi-Robot Support**: Validated on Booster T1 and Unitree G1 with sim-to-real transfer
- **Teacher-Student Distillation**: Train with privileged observations, distill to deployable student policies
- **Self-Contained Tasks**: Each task config is a single file; MDP term functions are shared across tasks via a common library (`agile/rl_env/mdp/`)
- **Evaluation Framework**: Random rollouts, deterministic scenarios, motion metrics, HTML reports, W&B integration
- **Sim-to-MuJoCo Transfer**: Generic framework for cross-simulator policy validation
- **Remote Training**: OSMO workflow support for cluster-based training, evaluation, and sweeps

## Quick Start

```bash
# Train a velocity tracking policy
python scripts/train.py --task Velocity-T1-v0 --num_envs 2048 --headless

# Evaluate the trained policy
python scripts/eval.py --task Velocity-T1-v0 --num_envs 32 --checkpoint <path>
```

See {doc}`getting-started` for full installation and setup instructions.

## Supported Tasks

| Category | Task IDs | Description |
|----------|----------|-------------|
| **Locomotion** | `Velocity-T1-v0`, `Velocity-G1-History-v0` | Lower-body velocity tracking on rough terrain |
| **Locomotion + Height** | `Velocity-Height-G1-v0`, distillation variants | Lower-body velocity + height tracking with teacher-student distillation |
| **Stand Up** | `StandUp-T1-v0` | Full-body recovery from fallen poses (unified whole-body policy) |
| **Pick & Place** | `G1-PickPlace-Tracking-v0` | Upper-body trajectory tracking with frozen lower-body locomotion policy |
| **Whole-Body Motion Tracking** | `Tracking-Flat-G1-v0` | Full-body motion imitation from reference trajectories (e.g., dancing) |
| **Debug** | `Debug-G1-v0`, `Debug-T1-v0` | Interactive GUI environments for debugging |

See {doc}`tasks` for detailed task documentation.

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

getting-started
```

```{toctree}
:maxdepth: 2
:caption: User Guide
:hidden:

training
evaluation
training-tips
tasks
mdp
data-recording
```

```{toctree}
:maxdepth: 2
:caption: Deployment
:hidden:

pretrained-policies
sim2mujoco
```

```{toctree}
:maxdepth: 2
:caption: Infrastructure
:hidden:

remote-training
testing
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

algorithms
```
