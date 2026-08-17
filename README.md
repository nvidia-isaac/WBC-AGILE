# **AGILE**: **A** **G**eneric **I**saac-**L**ab based **E**ngine for humanoid loco-manipulation learning

## Overview

**AGILE** provides a comprehensive reinforcement learning framework for training whole-body control policies with validated sim-to-real transfer capabilities. Built on NVIDIA Isaac Lab, this toolkit enables researchers and practitioners to develop loco-manipulation behaviors for humanoid robots.

**[Paper](https://arxiv.org/abs/2603.20147)**

AGILE targets Isaac Lab `v3.0.0-beta2`, Isaac Sim 6.0, Python 3.12, uv-based installation, and public RSL-RL `5.4.1` with a small AGILE patch.

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

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), and a Linux workstation with an NVIDIA GPU. The first `uv run` resolves AGILE's pinned Isaac Lab `3.0.0-beta2`, Isaac Sim 6.0, LEAPP, and RSL-RL dependencies.

```bash
# Train a velocity tracking policy
uv run scripts/train.py --task Velocity-T1-v0 --num_envs 2048 --headless

# Evaluate the trained policy
uv run scripts/eval.py --task Velocity-T1-v0 --num_envs 32 --checkpoint <path>
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

This repository contains code under two open-source licenses:

### Apache License 2.0
Most AGILE source code is licensed under the **Apache License 2.0**.
- **Copyright holder:** NVIDIA CORPORATION & AFFILIATES

### BSD 3-Clause License
The RSL-RL compatibility patch in `third_party/rsl_rl/patches/` is based on
[RSL_RL](https://github.com/leggedrobotics/rsl_rl), which is licensed under the
**BSD 3-Clause License** by ETH Zurich and contributors.

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
@misc{zhao2026agilecomprehensiveworkflowhumanoid,
      title={AGILE: A Comprehensive Workflow for Humanoid Loco-Manipulation Learning},
      author={Huihua Zhao* and Rafael Cathomen* and Lionel Gulich and Wei Liu and Efe Arda Ongan and Michael Lin and Shalin Jain and Soha Pouya and Yan Chang},
      year={2026},
      eprint={2603.20147},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.20147},
}
```
