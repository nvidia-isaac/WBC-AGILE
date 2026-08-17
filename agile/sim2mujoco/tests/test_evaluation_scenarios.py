# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from agile.algorithms.evaluation.eval_config import EvalConfig
from agile.evaluation.evaluation_manifest import load_evaluation_spec
from agile.sim2mujoco import leapp
from agile.sim2mujoco.command_provider import HeightCommandProvider
from agile.sim2mujoco.command_scheduler import Sim2MuJoCoCommandScheduler
from agile.sim2mujoco.commands import CommandManager
from scripts import sim2mujoco_eval


def test_eval_config_expands_height_ramp(tmp_path: Path) -> None:
    scenario = tmp_path / "height.yaml"
    scenario.write_text(
        """
evaluation:
  task_name: HeightTracking-G1-v0
  num_envs: 1
  episode_length_s: 6.0
  num_episodes: 1
  environments:
    - env_ids: [0]
      name: height_down
      ramps:
        - start_time: 2.0
          duration_s: 4.0
          interval_s: 1.0
          commands:
            height:
              height: [0.72, -0.50]
"""
    )

    config = EvalConfig.from_yaml(scenario)
    schedule = config.get_env_config(0).get_full_schedule(config.episode_length_s)

    assert [(step.time, step.commands["height"]["height"]) for step in schedule] == [
        (2.0, 0.72),
        (3.0, 0.415),
        (4.0, 0.11),
        (5.0, -0.195),
        (6.0, -0.5),
    ]


def test_sim2mujoco_scheduler_applies_velocity_sequence() -> None:
    command_manager = CommandManager(torch.device("cpu"))
    config = EvalConfig(
        task_name="Velocity-G1-History-v0",
        num_envs=1,
        episode_length_s=6.0,
        environments=[
            EvalConfig.from_yaml("agile/evaluation/scenarios/sim2mujoco_velocity_sequence.yaml").get_env_config(0)
        ],
    )
    scheduler = Sim2MuJoCoCommandScheduler(
        eval_config=config,
        command_manager=command_manager,
        duration=6.0,
        command_dim=3,
        verbose=False,
    )

    scheduler.update(2.0)
    assert command_manager.get_navigation_command().tolist() == pytest.approx([0.0, 0.0, 0.5])

    scheduler.update(4.0)
    assert command_manager.get_navigation_command().tolist() == pytest.approx([0.3, 0.0, 0.0])


def test_sim2mujoco_scheduler_applies_height_command() -> None:
    provider = HeightCommandProvider(torch.device("cpu"), default_height=0.72)
    config = EvalConfig.from_yaml("agile/evaluation/scenarios/sim2mujoco_height_tracking.yaml")
    scheduler = Sim2MuJoCoCommandScheduler(
        eval_config=config,
        command_provider=provider,
        duration=10.0,
        command_dim=1,
        verbose=False,
    )

    scheduler.update(6.0)

    assert provider.get_commands().item() == -0.5


def test_height_tracking_scenario_holds_final_stand_long_enough_for_video() -> None:
    config = EvalConfig.from_yaml("agile/evaluation/scenarios/sim2mujoco_height_tracking.yaml")
    schedule = config.get_env_config(0).get_full_schedule(config.episode_length_s)

    assert config.episode_length_s >= 12.0
    assert schedule[-1].time <= config.episode_length_s - 2.0
    assert schedule[-1].commands["height"]["height"] == pytest.approx(0.72)


def test_velocity_and_tracking_scenarios_enable_balance_validation() -> None:
    for path in (
        Path("agile/evaluation/scenarios/sim2mujoco_velocity_sequence.yaml"),
        Path("agile/evaluation/scenarios/sim2mujoco_motion_tracking_g1.yaml"),
    ):
        options = sim2mujoco_eval._load_sim2mujoco_options(path)
        balance = options["validation"]["balance"]

        assert balance["min_pelvis_height"] > 0.0
        assert balance["start_time_s"] >= 0.0


def test_tracking_flat_starts_mujoco_from_reference_motion() -> None:
    options = sim2mujoco_eval._load_sim2mujoco_options(
        Path("agile/evaluation/scenarios/sim2mujoco_motion_tracking_g1.yaml")
    )

    assert options["initial_state"] == "reference_motion"


def test_balance_monitor_fails_when_pelvis_drops_below_threshold() -> None:
    monitor = sim2mujoco_eval.BalanceMonitor(min_pelvis_height=0.5, start_time_s=1.0)

    monitor.update(time_s=0.5, pelvis_height=0.2)
    monitor.update(time_s=1.0, pelvis_height=0.7)
    monitor.update(time_s=1.5, pelvis_height=0.49)

    result = monitor.result()

    assert not result.passed
    assert result.min_pelvis_height == pytest.approx(0.49)
    assert "0.490" in result.message


def test_balance_monitor_passes_when_threshold_is_disabled() -> None:
    monitor = sim2mujoco_eval.BalanceMonitor(min_pelvis_height=None, start_time_s=0.0)

    monitor.update(time_s=1.0, pelvis_height=0.0)

    assert monitor.result().passed


def test_task_specs_select_task_specific_mujoco_scenarios() -> None:
    assert load_evaluation_spec(Path("agile/evaluation/specs/Velocity-G1-History-v0.yaml")).sim2mujoco_scenario == Path(
        "agile/evaluation/scenarios/sim2mujoco_velocity_sequence.yaml"
    )
    assert load_evaluation_spec(Path("agile/evaluation/specs/HeightTracking-G1-v0.yaml")).sim2mujoco_scenario == Path(
        "agile/evaluation/scenarios/sim2mujoco_height_tracking.yaml"
    )
    assert load_evaluation_spec(Path("agile/evaluation/specs/MotionTracking-G1-v0.yaml")).sim2mujoco_scenario == Path(
        "agile/evaluation/scenarios/sim2mujoco_motion_tracking_g1.yaml"
    )


def test_motion_provider_selection_ignores_anchor_reference_inputs() -> None:
    assert not leapp._is_motion_tracking_command_input({"name": "motion_anchor_ori_b", "shape": [1, 6]})
    assert leapp._is_motion_tracking_command_input({"name": "generated_commands", "shape": [1, 58]})
