# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1-parity RSL-RL configuration for Unitree H2 history training."""

from isaaclab.utils.configclass import configclass

from agile.rl_env.tasks.locomotion.g1.agents.rsl_rl_ppo_cfg import G1VelocityPpoRunnerCfg


@configclass
class H2VelocityHistoryPpoRunnerCfg(G1VelocityPpoRunnerCfg):
    entropy_coef_annealing_start_progress = 4_000 / 50_000
    experiment_name = "velocity_h2_history"
    run_name = "velocity_h2_history"
    wandb_project = "Velocity-H2-History"
