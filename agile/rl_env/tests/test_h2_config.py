# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public H2 training configuration contracts."""

import gymnasium as gym

from agile.rl_env.assets.robots import unitree_h2
from agile.rl_env.tasks.locomotion.h2.agents.rsl_rl_ppo_cfg import H2VelocityHistoryPpoRunnerCfg
from agile.rl_env.tasks.locomotion.h2.velocity_history_env_cfg import H2LowerVelocityHistoryEnvCfg

EXPECTED_H2_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.1/Isaac/Robots/Unitree/H2plus/unitree_h2.usda"
)


def test_h2_uses_public_training_asset() -> None:
    spawn = unitree_h2.H2_DELAYED_DC_MOTOR.spawn

    assert spawn.usd_path == EXPECTED_H2_USD
    assert spawn.variants == {"Physics": "physx", "collision_profile": "feet_only", "hands": "none"}
    assert getattr(spawn, "articulation_root_prim_path", None) is None


def test_h2_training_task_uses_public_asset() -> None:
    import agile.rl_env.tasks  # noqa: F401

    spec = gym.spec("Velocity-H2-History-v0")
    assert spec.kwargs["env_cfg_entry_point"].endswith(":H2LowerVelocityHistoryEnvCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":H2VelocityHistoryPpoRunnerCfg")
    assert H2LowerVelocityHistoryEnvCfg().scene.robot.spawn.usd_path == EXPECTED_H2_USD
    assert H2VelocityHistoryPpoRunnerCfg().entropy_coef_annealing_start_progress == 4_000 / 50_000
