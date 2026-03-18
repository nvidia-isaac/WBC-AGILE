# **AGILE**: **A** **G**eneric **I**saac-**L**ab based **E**ngine for humanoid loco-manipulation learning

## Overview

**AGILE** provides a comprehensive reinforcement learning framework for training whole-body control policies with validated sim-to-real transfer capabilities. Built on NVIDIA Isaac Lab, this toolkit enables researchers and practitioners to develop loco-manipulation behaviors for humanoid robots.

**[Documentation](https://nvidia-isaac.github.io/WBC-AGILE/)**

<table align="center">
  <tr>
    <th colspan="2">Booster T1 – Stand-Up</th>
    <th colspan="2">Booster T1 – Velocity Tracking</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/videos/booster_t1_stand_up_sim2sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="docs/videos/booster_t1_stand_up_sim2real.gif" width="240"><br><em>Real</em></td>
    <td align="center"><img src="docs/videos/booster_t1_vel_sim2sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="docs/videos/booster_t1_vel_sim2real.gif" width="240"><br><em>Real</em></td>
  </tr>
  <tr>
    <th colspan="2">Unitree G1 – Velocity-Height Tracking</th>
    <th colspan="2">Unitree G1 – Sit-Down / Stand-Up</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/videos/unitree_g1_vel_height_sim2sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="docs/videos/unitree_g1_vel_height_sim2real.gif" width="240"><br><em>Real</em></td>
    <td align="center"><img src="docs/videos/unitree_g1_updown_sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="docs/videos/unitree_g1_updown.gif" width="240"><br><em>Real</em></td>
  </tr>
  <tr>
    <th colspan="2">Unitree G1 – Teleoperation</th>
    <th colspan="2">Unitree G1 – Dancing</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/videos/locomanipulation-g1-sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="docs/videos/unitree_g1_teleop.gif" width="240"><br><em>Real</em></td>
    <td align="center"><img src="docs/videos/unitree_g1_dancing_sim.gif" width="240"><br><em>Sim</em></td>
    <td align="center"><img src="docs/videos/unitree_g1_dancing.gif" width="240"><br><em>Real</em></td>
  </tr>
</table>

## Key Features

- **Multi-Robot Support**: Validated on Booster T1 and Unitree G1 with sim-to-real transfer
- **Teacher-Student Distillation**: Train with privileged observations, distill to deployable student policies
- **Self-Contained Tasks**: Each task config is a single file; MDP term functions are shared via a common library
- **Evaluation Framework**: Random rollouts, deterministic scenarios, motion metrics, HTML reports, W&B integration
- **Sim-to-MuJoCo Transfer**: Generic framework for cross-simulator policy validation
- **Remote Training**: OSMO workflow support for cluster-based training, evaluation, and sweeps

## Quick Start

**Prerequisites:** [Isaac Lab v2.3.2](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html) with Isaac Sim 5.1.

```bash
# Install AGILE
export ISAACLAB_PATH=/path/to/IsaacLab
./scripts/setup/install_deps_local.sh

# Train a velocity tracking policy
python scripts/train.py --task Velocity-T1-v0 --num_envs 2048 --headless

# Evaluate the trained policy
python scripts/eval.py --task Velocity-T1-v0 --num_envs 32 --checkpoint <path>
```

See the [full documentation](https://nvidia-isaac.github.io/WBC-AGILE/) for installation details, training guides, task descriptions, and deployment instructions.

## Office Hour and FAQ

We hosted a robotics livestream office hour providing an in-depth walkthrough of the AGILE framework.

- **[YouTube Recording](https://www.youtube.com/live/ANvkdrESIuc?si=KPd8PvXFipt8FsG9)**
- **[FAQ Document](OFFICE_HOUR_FAQ.md)**

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on how to contribute to this project.

## License

<details>
<summary> License Information</summary>

This repository contains code under two different open-source licenses:

### BSD 3-Clause License
The reinforcement learning algorithm library located in `agile/algorithms/rsl_rl/` is licensed under the **BSD 3-Clause License**.
- **Copyright holders:** ETH Zurich, NVIDIA CORPORATION & AFFILIATES
- This portion is based on the [RSL_RL](https://github.com/leggedrobotics/rsl_rl) library developed at ETH Zurich

### Apache License 2.0
All other portions of this repository are licensed under the **Apache License 2.0**.
- **Copyright holder:** NVIDIA CORPORATION & AFFILIATES

For complete license terms, see the [LICENCE](LICENCE) file.

</details>

## Core Contributors
Huihua Zhao, Rafael Cathomen, Lionel Gulich, Efe Arda Ongan, Michael Lin, Shalin Jain, Wei Liu, Xinghao Zhu, Vishal Kulkarni, Soha Pouya, Yan Chang

## Acknowledgments
We would like to acknowledge the following projects from which parts of the code in this repo are derived:
- [Beyond Mimic](https://github.com/HybridRobotics/whole_body_tracking)
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

## Citation
If you use AGILE in your research, please cite:

```bibtex
@misc{agile2025,
  title        = {AGILE: A Generic Isaac-Lab based Engine for Humanoid Loco-Manipulation Learning},
  author       = {Zhao, Huihua and Cathomen, Rafael and Gulich, Lionel and Ongan, Efe Arda and Lin, Michael and Jain, Shalin and Liu, Wei and Kulkarni, Vishal and Pouya, Soha and Chang, Yan},
  year         = {2025},
  note         = {Version compatible with Isaac Lab 2.3; accessed 2025-11-19},
  url          = {https://github.com/nvidia-isaac/WBC-AGILE/tree/main}
}
```
