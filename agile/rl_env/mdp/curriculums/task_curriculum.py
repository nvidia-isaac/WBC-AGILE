# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Curriculum terms based on locomotion performance/ traveled distance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from agile.rl_env.mdp import HarnessAction

type Direction = Literal["above", "below"]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


class terrain_levels_vel_curriculum(ManagerTermBase):
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.num_failures = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.num_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        command_name: str = "base_velocity",
        move_up_distance: float = 6.0,
        move_down_distance: float = 3.0,
        n_failures: int = 3,
        n_successes: int = 3,
        p_random_move_up: float = 0.0,
        p_random_move_down: float = 0.0,
    ) -> torch.Tensor:
        """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

        This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
        difficulty when the robot walks less than half of the distance required by the commanded velocity.

        .. note::
            It is only possible to use this term with the terrain type ``generator``. For further information
            on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

        Returns:
            The mean terrain level for the given environment ids.
        """
        # extract the used quantities (to enable type-hinting)
        terrain: TerrainImporter = env.scene.terrain

        traveled_distance = env.command_manager._terms[command_name].metrics["traveled_distance"][env_ids]

        # move up if the robot has traveled far enou  gh
        succeeded = traveled_distance > move_up_distance
        self.num_successes[env_ids] += succeeded

        # move down if the robot has failed too many times
        failed = traveled_distance < move_down_distance
        self.num_failures[env_ids] += failed

        move_up = self.num_successes[env_ids] >= n_successes
        move_down = self.num_failures[env_ids] >= n_failures

        # reset the number of successes and failures when the robot moves up or down
        self.num_failures[env_ids[move_up | move_down]] = 0.0
        self.num_successes[env_ids[move_up | move_down]] = 0.0

        # add random move up and down
        if p_random_move_up > 0.0:
            random_move_up = (torch.rand(env_ids.shape, device=env.device) < p_random_move_up) & ~move_down
            move_up = move_up | random_move_up
        if p_random_move_down > 0.0:
            random_move_down = (torch.rand(env_ids.shape, device=env.device) < p_random_move_down) & ~move_up
            move_down = move_down | random_move_down

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class terrain_levels_successful_termination(ManagerTermBase):
    """Curriculum based on how often the robot terminates due to the specified termination term."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.num_failures = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.num_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        successful_termination_term: str,
        n_failures: int = 3,
        n_successes: int = 3,
    ) -> torch.Tensor:
        """Curriculum based on the termination condition.

        The robot moves to a more difficult terrain if it terminates due to the specified `successful_termination_term`
        `n_successes` times and it moves to simpler terrain if it does not terminate due to the specified term `n_failures` times.

        .. note::
            It is only possible to use this term with the terrain type ``generator``. For further information
            on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

        Returns:
            The mean terrain level for the given environment ids.
        """
        # extract the used quantities (to enable type-hinting)
        terrain: TerrainImporter = env.scene.terrain

        # find the envs that succeeded
        succeeded = env.termination_manager.get_term(successful_termination_term)[env_ids]
        self.num_successes[env_ids] += succeeded

        # move down if the robot has failed too many times
        self.num_failures[env_ids] += ~succeeded

        move_up = self.num_successes[env_ids] >= n_successes
        move_down = self.num_failures[env_ids] >= n_failures

        # reset the number of successes and failures when the robot moves up or down
        self.num_failures[env_ids[move_up | move_down]] = 0.0
        self.num_successes[env_ids[move_up | move_down]] = 0.0

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class terrain_levels_standing_at_timeout(ManagerTermBase):
    """Curriculum based on whether the robot is standing when the episode times out.

    This curriculum is designed for stand-up tasks where success is defined as
    being at standing height when the episode ends due to timeout (not early termination).

    Supports an optional prerequisite curriculum that must reach a threshold before
    this curriculum becomes active.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.num_failures = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.num_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self._activated = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        min_height: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
        n_failures: int = 3,
        n_successes: int = 3,
        prerequisite_curriculum: str | None = None,
        prerequisite_threshold: float = 0.0,
        prerequisite_direction: Direction = "below",
    ) -> torch.Tensor:
        """Curriculum based on standing at timeout.

        The robot moves to a more difficult terrain if it is standing at timeout
        `n_successes` times and moves to simpler terrain if it fails `n_failures` times.

        Args:
            env: The environment.
            env_ids: The environment IDs that are being reset.
            min_height: Minimum height to be considered standing.
            asset_cfg: Asset configuration for the robot.
            sensor_cfg: Optional height sensor for rough terrain adjustment.
            n_failures: Number of failures before moving to easier terrain.
            n_successes: Number of successes before moving to harder terrain.
            prerequisite_curriculum: Name of another curriculum that must reach threshold
                before this one activates. If None, this curriculum is always active.
            prerequisite_threshold: The threshold value the prerequisite must reach.
            prerequisite_direction: "below" means prerequisite must be <= threshold,
                "above" means prerequisite must be >= threshold.

        Returns:
            The mean terrain level for the given environment ids.
        """
        terrain: TerrainImporter = env.scene.terrain

        # Check prerequisite curriculum if configured (only until first activation)
        if not self._activated and prerequisite_curriculum is not None:
            prereq_value = env.curriculum_manager._curriculum_state.get(prerequisite_curriculum)
            if prereq_value is None:
                return 0.0

            if prerequisite_direction == "below":
                prereq_met = prereq_value <= prerequisite_threshold
            else:  # "above"
                prereq_met = prereq_value >= prerequisite_threshold

            if not prereq_met:
                # Prerequisite not met, don't update terrain levels
                return torch.mean(terrain.terrain_levels.float())

            # Prerequisite met for the first time, activate permanently
            self._activated = True

        # Check if these envs timed out
        is_timeout = env.termination_manager.time_outs[env_ids]

        # Check current height for these envs
        asset = env.scene[asset_cfg.name]
        if sensor_cfg is not None:
            sensor: RayCaster = env.scene[sensor_cfg.name]
            current_height = asset.data.root_pos_w.torch[env_ids, 2] - torch.mean(
                sensor.data.ray_hits_w[env_ids, ..., 2], dim=1
            )
        else:
            current_height = asset.data.root_pos_w.torch[env_ids, 2]

        is_standing = current_height > min_height

        # Success = timeout AND standing
        succeeded = is_timeout & is_standing

        self.num_successes[env_ids] += succeeded
        self.num_failures[env_ids] += ~succeeded

        move_up = self.num_successes[env_ids] >= n_successes
        move_down = self.num_failures[env_ids] >= n_failures

        # Reset counters when moving up or down
        self.num_failures[env_ids[move_up | move_down]] = 0
        self.num_successes[env_ids[move_up | move_down]] = 0

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class terrain_levels_tracking_at_timeout(ManagerTermBase):
    """Curriculum based on height command tracking error when the episode times out.

    Designed for height-tracking tasks: success is defined as having low tracking
    error at timeout, rather than reaching a fixed standing height.

    Supports an optional prerequisite curriculum that must reach a threshold before
    this curriculum becomes active.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.num_failures = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self.num_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self._activated = False
        self._command_term = env.command_manager.get_term(cfg.params["command_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        command_name: str,  # noqa: ARG002
        error_threshold: float = 0.1,
        n_failures: int = 3,
        n_successes: int = 3,
        ignore_terminations: bool = False,
        prerequisite_curriculum: str | None = None,
        prerequisite_threshold: float = 0.0,
        prerequisite_direction: Direction = "below",
        error_metric_name: str = "height_error",
    ) -> torch.Tensor:
        """Curriculum based on tracking error at timeout.

        The robot moves to harder terrain if tracking error is below threshold at
        timeout `n_successes` times, and to easier terrain if it fails `n_failures` times.

        Args:
            env: The environment.
            env_ids: The environment IDs that are being reset.
            command_name: Name of the command term.
            error_threshold: Max tracking error (meters) to count as success.
            n_failures: Number of failures before moving to easier terrain.
            n_successes: Number of successes before moving to harder terrain.
            ignore_terminations: If True, episodes that end via early termination
                (not timeout) are ignored — they don't count as success or failure.
                If False (default), non-timeout episodes count as failures.
            prerequisite_curriculum: Name of another curriculum that must reach threshold
                before this one activates. If None, this curriculum is always active.
            prerequisite_threshold: The threshold value the prerequisite must reach.
            prerequisite_direction: "below" means prerequisite must be <= threshold,
                "above" means prerequisite must be >= threshold.
            error_metric_name: Name of the error metric in command_term.metrics to use.

        Returns:
            The mean terrain level for the given environment ids.
        """
        terrain: TerrainImporter = env.scene.terrain

        # Check prerequisite curriculum if configured (only until first activation)
        if not self._activated and prerequisite_curriculum is not None:
            prereq_value = env.curriculum_manager._curriculum_state.get(prerequisite_curriculum)
            if prereq_value is None:
                return 0.0

            if prerequisite_direction == "below":
                prereq_met = prereq_value <= prerequisite_threshold
            else:  # "above"
                prereq_met = prereq_value >= prerequisite_threshold

            if not prereq_met:
                return torch.mean(terrain.terrain_levels.float())

            # Prerequisite met for the first time, activate permanently
            self._activated = True

        # Check if these envs timed out
        is_timeout = env.termination_manager.time_outs[env_ids]

        # If ignoring terminations, only evaluate envs that timed out
        if ignore_terminations:
            timeout_mask = is_timeout.nonzero(as_tuple=False).squeeze(-1)
            if timeout_mask.numel() == 0:
                return torch.mean(terrain.terrain_levels.float())
            eval_ids = env_ids[timeout_mask]

            mean_error = self._command_term.metrics[error_metric_name][eval_ids]
            is_tracking_well = mean_error < error_threshold

            self.num_successes[eval_ids] += is_tracking_well
            self.num_failures[eval_ids] += ~is_tracking_well
        else:
            # Check episode-average tracking error for these envs
            mean_error = self._command_term.metrics[error_metric_name][env_ids]
            is_tracking_well = mean_error < error_threshold

            # Success = timeout AND low average tracking error
            succeeded = is_timeout & is_tracking_well

            self.num_successes[env_ids] += succeeded
            self.num_failures[env_ids] += ~succeeded

        move_up = self.num_successes[env_ids] >= n_successes
        move_down = self.num_failures[env_ids] >= n_failures

        # Reset counters when moving up or down
        self.num_failures[env_ids[move_up | move_down]] = 0
        self.num_successes[env_ids[move_up | move_down]] = 0

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class action_limit_successful_termination(ManagerTermBase):
    """Curriculum based on the ratio of successful terminations."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ema_success_ratio = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        action_name: str,
        successful_termination_term: str,
        activate_after_steps: int = 0,
        move_up_ratio: float = 0.95,
        move_down_ratio: float = 0.8,
        update_rate: float = 0.0001,
        ema_decay: float = 0.99,
        max_action_limit: float = 1.0,
        min_action_limit: float = 0.0,
    ) -> torch.Tensor:
        """Curriculum based on the ratio of successful terminations.

        Note: This curriculum only makes sense if the action is a relative joint action.

        The action limit is increased if it terminates due to the specified `successful_termination_term` more than
        `move_up_ratio` times and it is decreased if it terminates less then `move_down_ratio` times.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            action_name: name of the action term
            successful_termination_term: name of the termination term term
            activate_after_steps: step at which to start the curriculum
            move_up_ratio: ratio of successful terminations to increase the action limit
            move_down_ratio: ratio of successful terminations to decrease the action limit
            update_rate: rate at which to update the action limit
            ema_decay: decay rate for the exponential moving average of the ratio of successful terminations
            max_action_limit: maximum action limit
            min_action_limit: minimum action limit

        """

        if env.common_step_counter < activate_after_steps:
            return 1.0

        # extract the used quantities (to enable type-hinting)
        action: RelativeJointPositionAction = env.action_manager._terms[action_name]

        # find the envs that succeeded
        succeeded = env.termination_manager.get_term(successful_termination_term)[env_ids]
        self.ema_success_ratio = ema_decay * self.ema_success_ratio + (1 - ema_decay) * succeeded.float().mean()

        if self.ema_success_ratio > move_up_ratio:
            action._clip = action._clip * (1 + (move_up_ratio - self.ema_success_ratio) * update_rate)
        elif self.ema_success_ratio < move_down_ratio:
            action._clip = action._clip * (1 + (move_down_ratio - self.ema_success_ratio) * update_rate)

        action_clip_sign = torch.sign(action._clip)
        action_clip_abs = torch.clamp(action._clip.abs(), min=min_action_limit, max=max_action_limit)
        action._clip = action_clip_sign * action_clip_abs

        return action_clip_abs.max().item()


def remove_harness(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],  # noqa: ARG001
    harness_action_name: str,
    start: int,
    num_steps: int,
    linear: bool = True,
) -> float:
    """Curriculum that reduces the harness linearly given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        harness_action_name: name of the harness action
        start: step at which to start reducing the harness
        num_steps: number of steps in which the reducing happens
        linear: if True, reduce linearly, else reduce exponentially
    """
    harness_action: HarnessAction = env.action_manager._terms[harness_action_name]

    if env.common_step_counter <= start:
        return 1.0
    elif env.common_step_counter > start + num_steps:
        harness_action.scale_forces(0.0)

        return 0.0
    else:
        if linear:
            scale = 1 - (env.common_step_counter - start) / num_steps
        else:
            current_step_in_decay = env.common_step_counter - start
            target_scale = 0.01
            log_target_scale = math.log(target_scale)
            progress = current_step_in_decay / num_steps
            current_log_scale = progress * log_target_scale
            scale = math.exp(current_log_scale)
        harness_action.scale_forces(scale)

        return scale  # type: ignore


def _check_prerequisite(
    env: ManagerBasedRLEnv,
    prerequisite_curriculum: str | None,
    prerequisite_threshold: float,
    prerequisite_direction: Direction,
) -> bool:
    """Check if a prerequisite curriculum condition is met.

    Returns True if the prerequisite is satisfied or not configured.
    """
    if prerequisite_curriculum is None:
        return True
    prereq_value = env.curriculum_manager._curriculum_state.get(prerequisite_curriculum)
    if prereq_value is None:
        return False
    if prerequisite_direction == "below":
        return bool(prereq_value <= prerequisite_threshold)
    return bool(prereq_value >= prerequisite_threshold)


class adaptive_force_decay(ManagerTermBase):
    """Adaptive curriculum that decays action forces based on a performance metric.

    Monitors a metric, smooths it with EMA, and multiplicatively decays the force
    scale when the smoothed metric crosses a threshold. The force scale is
    monotonically non-increasing: once decayed, it never increases again.

    Supported ``metric_name`` values:

    - ``"standing_ratio"``: Fraction of envs reaching standing height (from
      ``action.max_heights``). Requires ``standing_height_threshold``.
    - ``"velocity_xy"``: Instantaneous velocity tracking error norm (L2).
      Evaluated every step on all envs. Requires ``command_name``.
    - Any key in ``command_term.metrics`` (e.g. ``"height_error"``):
      Per-episode metric read at env reset. Requires ``command_name``.

    Any action with a ``scale_forces(float)`` method is supported (e.g.,
    LiftAction, HarnessAction, CommandAssistAction). Use ``scale_method``
    to target a specific component (e.g., ``"scale_velocity"``,
    ``"scale_height"``).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._force_scale: float = 1.0
        self._action = env.action_manager._terms[cfg.params["action_name"]]
        self._scale_method: str = cfg.params.get("scale_method", "scale_forces")

        decay_when = cfg.params.get("decay_when", "above")
        self._ema: float = 0.0 if decay_when == "above" else 1.0

        if "command_name" in cfg.params:
            self._command_term = env.command_manager.get_term(cfg.params["command_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        action_name: str,  # noqa: ARG002
        metric_name: str = "standing_ratio",
        decay_when: Direction = "above",
        threshold: float = 0.8,
        ema_alpha: float = 0.01,
        decay: float = 0.999,
        disable_threshold: float = 0.01,
        # Metric-specific (used at init or by specific metrics)
        command_name: str | None = None,  # noqa: ARG002
        standing_height_threshold: float = 0.7,
        # Scale method
        scale_method: str = "scale_forces",  # noqa: ARG002
        # Prerequisite
        prerequisite_curriculum: str | None = None,
        prerequisite_threshold: float = 0.0,
        prerequisite_direction: Direction = "below",
    ) -> float:
        """Update force scale based on smoothed metric.

        Args:
            env: The learning environment.
            env_ids: Environment IDs being reset (unused for ``"velocity_xy"``).
            action_name: Name of the action term to control (used at init).
            metric_name: Which metric to monitor. ``"standing_ratio"`` uses
                ``action.max_heights``; ``"velocity_xy"`` computes velocity error
                every step; anything else reads from ``command_term.metrics``.
            decay_when: ``"above"`` decays when EMA exceeds threshold (e.g. high
                standing ratio means robot learned). ``"below"`` decays when EMA
                drops below threshold (e.g. low error means robot learned).
            threshold: The value the EMA must cross to trigger decay.
            ema_alpha: EMA smoothing factor. Smaller = smoother.
            decay: Multiplicative decay per step. Scale ``*= decay``.
            disable_threshold: Set scale to 0 when it drops below this.
            command_name: Command term name (used at init for error metrics).
            standing_height_threshold: Min height to count as standing
                (``"standing_ratio"`` only).
            prerequisite_curriculum: Name of another curriculum that must reach a
                threshold before this one starts decaying.
            prerequisite_threshold: The threshold the prerequisite must reach.
            prerequisite_direction: ``"below"`` = ``<= threshold``,
                ``"above"`` = ``>= threshold``.

        Returns:
            Current force scale [0, 1].
        """
        if self._force_scale <= 0:
            return 0.0

        # Per-step metrics run every step; per-episode metrics skip when no envs reset
        per_step_metrics = {"velocity_xy", "yaw_error", "orientation_error", "height_error"}
        if metric_name not in per_step_metrics and len(env_ids) == 0:
            return self._force_scale

        if not _check_prerequisite(env, prerequisite_curriculum, prerequisite_threshold, prerequisite_direction):
            return self._force_scale

        # Compute metric
        if metric_name == "standing_ratio":
            metric = float((self._action.max_heights[env_ids] > standing_height_threshold).float().mean().item())
        elif metric_name == "velocity_xy":
            cmd_vel_xy = self._command_term.command[:, :2]
            # Use smoothed velocity from the command term (if available) so that
            # jerky high-frequency oscillations don't keep the assist alive.
            if hasattr(self._command_term, "vel_xy_smoothed"):
                cur_vel_xy = self._command_term.vel_xy_smoothed
            else:
                cur_vel_xy = env.scene["robot"].data.root_lin_vel_b.torch[:, :2]
            metric = float(torch.norm(cmd_vel_xy - cur_vel_xy, dim=1).mean().item())
        elif metric_name == "yaw_error":
            cmd_yaw = self._command_term.command[:, 2]
            # Use the smoothed yaw rate from the assist action (if available) so that
            # jerky high-frequency oscillations don't keep the assist alive.
            if hasattr(self._action, "_yaw_rate_smoothed"):
                cur_yaw = self._action._yaw_rate_smoothed
            else:
                cur_yaw = env.scene["robot"].data.root_ang_vel_w.torch[:, 2]
            metric = float(torch.abs(cmd_yaw - cur_yaw).mean().item())
        elif metric_name == "orientation_error":
            gravity = env.scene["robot"].data.projected_gravity_b.torch  # (N, 3)
            # Roll/pitch error from upright: projected gravity should be (0, 0, -1)
            metric = float(torch.norm(gravity[:, :2], dim=1).mean().item())
        elif metric_name == "height_error" and hasattr(self._command_term, "base_height"):
            metric = float(torch.abs(self._command_term.base_height - self._command_term.target_height).mean().item())
        else:
            metric = float(self._command_term.metrics[metric_name][env_ids].mean().item())

        # EMA update
        self._ema = ema_alpha * metric + (1 - ema_alpha) * self._ema

        # Decay when metric crosses threshold
        should_decay = self._ema > threshold if decay_when == "above" else self._ema < threshold
        if should_decay:
            self._force_scale *= decay
            if self._force_scale < disable_threshold:
                self._force_scale = 0.0

        getattr(self._action, self._scale_method)(self._force_scale)
        return self._force_scale


class goal_pose_force_decay(ManagerTermBase):
    """Global decay of assist forces based on success rate at command interval completion.

    Detects when envs complete a command interval (time-based resampling, excluding
    episode resets) and checks the terminal error from the previous step. Decays the
    global force scale when the fraction of successful completions (error < threshold)
    exceeds ``success_rate``.

    With 4096 envs and 5-10s intervals, ~10 envs complete per step, giving a reliable
    per-step success rate estimate. The decay fires at most once per step regardless
    of how many envs complete, so the rate is independent of ``num_envs``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._force_scale: float = 1.0
        self._action = env.action_manager._terms[cfg.params["action_name"]]
        self._scale_method: str = cfg.params.get("scale_method", "scale_forces")
        self._metric_name: str = cfg.params["metric_name"]
        self._command_term = env.command_manager.get_term(cfg.params["command_name"])
        self._dt: float = env.step_dt

        # Previous step state — large init prevents spurious detection on first step
        self._prev_time_left = torch.full((env.num_envs,), 1e6, device=env.device)
        self._prev_metric = torch.zeros(env.num_envs, device=env.device)

    def _get_metric(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Get the per-env metric tensor."""
        if self._metric_name == "orientation_error":
            gravity = env.scene["robot"].data.projected_gravity_b.torch
            return torch.norm(gravity[:, :2], dim=1)
        return self._command_term.metrics[self._metric_name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        action_name: str,  # noqa: ARG002
        command_name: str,  # noqa: ARG002
        metric_name: str,  # noqa: ARG002
        threshold: float = 0.5,
        decay: float = 0.9999,
        success_rate: float = 0.8,
        scale_method: str = "scale_forces",  # noqa: ARG002
        disable_threshold: float = 0.01,
    ) -> float:
        if self._force_scale <= 0:
            return 0.0

        time_left = self._command_term.time_left

        # Detect command interval completions: time_left jumped up (resampled)
        resampled = time_left > self._prev_time_left + self._dt

        # Exclude episode resets — those are not successful interval completions
        reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if len(env_ids) > 0:
            reset_mask[env_ids] = True
        completed = resampled & ~reset_mask

        if completed.any():
            terminal_error = self._prev_metric[completed].abs()
            rate = (terminal_error < threshold).float().mean().item()
            if rate >= success_rate:
                self._force_scale *= decay
                if self._force_scale < disable_threshold:
                    self._force_scale = 0.0

        # Store current state for next step
        self._prev_time_left[:] = time_left
        self._prev_metric[:] = self._get_metric(env)

        getattr(self._action, self._scale_method)(self._force_scale)
        return self._force_scale


class goal_distance_curriculum(ManagerTermBase):
    """Bidirectional curriculum for goal XY distance with hysteresis.

    Detects when envs complete a command interval (time-based resampling,
    excluding episode resets) and checks the terminal position error.

    - **Grow** when success rate >= ``success_rate`` (default 0.8)
    - **Shrink** when success rate < ``shrink_success_rate`` (default 0.4)
    - **Hold** in between (hysteresis dead zone prevents oscillation)

    The shrink rate is ``shrink_factor * increment`` so the curriculum backs off
    quickly when other curriculums (e.g. assist decay) make the task harder.

    The lower bound of ``pos_radius`` stays at 0 so the policy always sees a mix
    of easy and hard goals, which stabilises value function learning.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command_term = env.command_manager.get_term(cfg.params["command_name"])
        self._dt: float = env.step_dt

        # Curriculum state
        self._current_max_radius: float = cfg.params.get("initial_max_radius", 0.5)
        self._min_radius: float = cfg.params.get("initial_max_radius", 0.5)
        self._locked: bool = False  # Once max is reached, lock and never shrink back

        # Apply initial range immediately
        self._command_term.cfg.ranges.pos_radius = (0.0, self._current_max_radius)

        # Previous step state
        self._prev_time_left = torch.full((env.num_envs,), 1e6, device=env.device)
        self._prev_metric = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        command_name: str,  # noqa: ARG002
        initial_max_radius: float = 0.5,  # noqa: ARG002
        max_radius: float = 5.0,
        increment: float = 0.1,
        threshold: float = 0.3,
        success_rate: float = 0.8,
        shrink_success_rate: float = 0.4,
        shrink_factor: float = 2.0,
        lock_at_max: bool = True,
    ) -> float:
        """Adjust goal radius based on tracking performance.

        Args:
            command_name: Name of the humanoid pose command term.
            initial_max_radius: Starting max radius (used in __init__ only).
            max_radius: Upper cap for the goal radius.
            increment: Additive increase per successful batch.
            threshold: Position error (m) below which a completion counts as success.
            success_rate: Fraction above which radius grows.
            shrink_success_rate: Fraction below which radius shrinks.
            shrink_factor: Multiplier on ``increment`` for shrink speed.
            lock_at_max: If True, once max_radius is reached, never shrink back.

        Returns:
            Current maximum goal radius.
        """
        time_left = self._command_term.time_left

        # Detect command interval completions: time_left jumped up (resampled)
        resampled = time_left > self._prev_time_left + self._dt

        # Exclude episode resets
        reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if len(env_ids) > 0:
            reset_mask[env_ids] = True
        completed = resampled & ~reset_mask

        if completed.any() and not self._locked:
            terminal_error = self._prev_metric[completed].abs()
            rate = (terminal_error < threshold).float().mean().item()
            if rate >= success_rate and self._current_max_radius < max_radius:
                self._current_max_radius = min(self._current_max_radius + increment, max_radius)
                self._command_term.cfg.ranges.pos_radius = (0.0, self._current_max_radius)
                # Lock once max is reached so disturbance curriculum can take over
                if lock_at_max and self._current_max_radius >= max_radius:
                    self._locked = True
            elif rate < shrink_success_rate and self._current_max_radius > self._min_radius:
                self._current_max_radius = max(self._current_max_radius - increment * shrink_factor, self._min_radius)
                self._command_term.cfg.ranges.pos_radius = (0.0, self._current_max_radius)

        # Store current state for next step
        self._prev_time_left[:] = time_left
        self._prev_metric[:] = self._command_term.metrics["torso_position_error"]

        return self._current_max_radius


class update_event_range_step(ManagerTermBase):
    """Curriculum to update event parameter ranges given the iteration.

    This curriculum linearly interpolates between start and terminal ranges
    based on training steps, useful for gradually increasing randomization
    like object pose ranges.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.event_term: str = cfg.params["event_term"]
        self.param_name: str = cfg.params["param_name"]

        # Get the original parameter structure (could be dict of tuples)
        self.original_param = env.event_manager.get_term_cfg(self.event_term).params[self.param_name]

        # Store start and terminal ranges
        self.start_range: dict[str, tuple[float, float]] = cfg.params["start_range"]
        self.terminal_range: dict[str, tuple[float, float]] = cfg.params["terminal_range"]

        # Set initial range
        self._update_range(env, 0.0)

    def _interpolate_range(
        self, start: tuple[float, float], terminal: tuple[float, float], scale: float
    ) -> tuple[float, float]:
        """Linearly interpolate between start and terminal range."""
        return (
            start[0] + (terminal[0] - start[0]) * scale,
            start[1] + (terminal[1] - start[1]) * scale,
        )

    def _update_range(self, env: ManagerBasedRLEnv, scale: float) -> dict:
        """Update the event parameter with interpolated ranges."""
        new_range = {}
        for key in self.original_param:
            if key in self.start_range and key in self.terminal_range:
                new_range[key] = self._interpolate_range(self.start_range[key], self.terminal_range[key], scale)
            else:
                # Keep original value for keys not specified
                new_range[key] = self.original_param[key]

        env.event_manager.get_term_cfg(self.event_term).params[self.param_name] = new_range
        return new_range

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        event_term: str,  # noqa: ARG002
        param_name: str,  # noqa: ARG002
        start_range: dict[str, tuple[float, float]],  # noqa: ARG002
        terminal_range: dict[str, tuple[float, float]],  # noqa: ARG002
        start_step: int,
        num_steps: int,
    ) -> float:
        """Update event parameter range linearly over training steps.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            event_term: Name of the event term to update.
            param_name: Name of the parameter to update (e.g., "pose_range").
            start_range: Starting range dict (e.g., {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}).
            terminal_range: Terminal range dict (e.g., {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}).
            start_step: When to start updating.
            num_steps: How long to update.

        Returns:
            Current interpolation scale (0.0 to 1.0).
        """
        if env.common_step_counter <= start_step:
            return 0.0
        elif env.common_step_counter > start_step + num_steps:
            self._update_range(env, 1.0)
            return 1.0
        else:
            scale: float = (env.common_step_counter - start_step) / num_steps
            self._update_range(env, scale)
            return scale


class velocity_command_range_step(ManagerTermBase):
    """Linearly interpolate velocity command ranges over a fixed number of training steps.

    Directly modifies the command term's ranges (lin_vel_x, lin_vel_y, ang_vel_z)
    from start_ranges to terminal_ranges between start_step and start_step + num_steps.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_name = cfg.params["command_name"]
        self.start_ranges = cfg.params["start_ranges"]
        self.terminal_ranges = cfg.params["terminal_ranges"]
        command = env.command_manager._terms[self.command_name]
        for attr, (lo, hi) in self.start_ranges.items():
            setattr(command.cfg.ranges, attr, (lo, hi))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        command_name: str,  # noqa: ARG002
        start_ranges: dict[str, tuple[float, float]],  # noqa: ARG002
        terminal_ranges: dict[str, tuple[float, float]],  # noqa: ARG002
        start_step: int,
        num_steps: int,
    ) -> float:
        if env.common_step_counter <= start_step:
            return 0.0

        scale: float = min((env.common_step_counter - start_step) / num_steps, 1.0)
        command = env.command_manager._terms[self.command_name]
        for attr in self.start_ranges:
            s = self.start_ranges[attr]
            t = self.terminal_ranges[attr]
            new_range = (s[0] + (t[0] - s[0]) * scale, s[1] + (t[1] - s[1]) * scale)
            setattr(command.cfg.ranges, attr, new_range)
        return scale


class update_event_param_after_curriculum(ManagerTermBase):
    """Ramp an event parameter after a prerequisite curriculum is met.

    Supports scalars (e.g., random_fallen_ratio: 0.0 → 0.5),
    tuples (e.g., force_range: (-5, 5) → (-20, 20)),
    and dicts of tuples (e.g., velocity_range: {"x": (-0.1, 0.1), ...}).

    If ``start_value`` is provided, interpolates from it to ``terminal_value``.
    If omitted, defaults to ``0.0`` (scalar ramp from zero).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._trigger_step: int | None = None
        self._start_value = cfg.params.get("start_value", 0.0)

        # Apply initial value to the event term immediately.
        event_term = cfg.params["event_term"]
        param_name = cfg.params["param_name"]
        env.event_manager.get_term_cfg(event_term).params[param_name] = self._start_value

    def _interpolate(self, start: Any, terminal: Any, scale: float) -> Any:
        """Recursively interpolate between start and terminal. Handles scalars, tuples, and dicts."""
        if isinstance(start, dict):
            return {k: self._interpolate(start[k], terminal[k], scale) for k in start}
        if isinstance(start, tuple | list):
            return type(start)(s + (t - s) * scale for s, t in zip(start, terminal, strict=False))
        return start + (terminal - start) * scale

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        event_term: str,
        param_name: str,
        terminal_value: Any,
        prerequisite_curriculum: str,
        prerequisite_threshold: float,
        delay_steps: int,
        num_steps: int,
        start_value: Any = None,  # noqa: ARG002 — read from self._start_value
    ) -> float:
        """Ramp event param after prerequisite is met.

        Returns:
            Current interpolation scale (0.0 to 1.0).
        """
        if self._trigger_step is None:
            prereq_value = env.curriculum_manager._curriculum_state.get(prerequisite_curriculum)
            if prereq_value is not None and prereq_value >= prerequisite_threshold:
                self._trigger_step = env.common_step_counter
            else:
                return 0.0

        ramp_start = self._trigger_step + delay_steps
        if env.common_step_counter <= ramp_start:
            return 0.0

        scale = min((env.common_step_counter - ramp_start) / num_steps, 1.0)
        new_value = self._interpolate(self._start_value, terminal_value, scale)
        env.event_manager.get_term_cfg(event_term).params[param_name] = new_value
        return float(scale)


class update_reward_weight_step(ManagerTermBase):
    """Curriculum to update reward weights given the iteration."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reward_name = cfg.params["reward_name"]
        if not isinstance(reward_name, str):
            raise ValueError(f"reward_name must be a string, got {type(reward_name)}")
        self.start_weight: float = env.reward_manager.get_term_cfg(reward_name).weight

        # Validate log space parameters if use_log_space is enabled
        use_log_space = cfg.params.get("use_log_space", False)
        if use_log_space:
            terminal_weight = cfg.params["terminal_weight"]

            # Check that start and terminal weights have the same sign
            if (self.start_weight > 0) != (terminal_weight > 0):
                raise ValueError(
                    f"For log space scaling, start_weight ({self.start_weight}) and "
                    f"terminal_weight ({terminal_weight}) must have the same sign"
                )

            if self.start_weight == 0 or terminal_weight == 0:
                raise ValueError(
                    f"For log space scaling, weights cannot be zero. "
                    f"start_weight={self.start_weight}, terminal_weight={terminal_weight}"
                )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        reward_name: str,
        start_step: int,
        num_steps: int,
        terminal_weight: float,
        use_log_space: bool = False,
    ) -> float:
        """Curriculum that changes the reward weight linearly or logarithmically given number of steps.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            reward_name: reward to update
            start_step: when to start updating
            num_steps: how long to update
            terminal_weight: reward weight after curriculum is finished
            use_log_space: if True, change weight magnitude logarithmically instead of linearly.
                          Both start_weight and terminal_weight must have the same sign.
        """
        if env.common_step_counter <= start_step:
            return self.start_weight
        elif env.common_step_counter > start_step + num_steps:
            env.reward_manager.get_term_cfg(reward_name).weight = terminal_weight
            return terminal_weight
        else:
            scale = (env.common_step_counter - start_step) / num_steps

            if use_log_space:
                # Work with absolute values for log space interpolation
                abs_start = abs(self.start_weight)
                abs_terminal = abs(terminal_weight)

                # Interpolate in log space
                log_start = math.log(abs_start)
                log_terminal = math.log(abs_terminal)
                log_weight = log_start + scale * (log_terminal - log_start)
                abs_new_weight = math.exp(log_weight)

                # Apply the original sign
                new_weight = abs_new_weight if self.start_weight > 0 else -abs_new_weight
            else:
                # Linear interpolation (original behavior)
                new_weight = self.start_weight + (terminal_weight - self.start_weight) * scale

            env.reward_manager.get_term_cfg(reward_name).weight = new_weight
            return new_weight


class update_reward_weight_after_curriculum(ManagerTermBase):
    """Curriculum that ramps a reward weight after a prerequisite curriculum reaches a threshold.

    Once the prerequisite is met, waits `delay_steps`, then ramps the reward
    weight from its current value to `terminal_weight` over `num_steps`.
    Supports linear or log-space interpolation.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reward_name = cfg.params["reward_name"]
        self.start_weight: float = env.reward_manager.get_term_cfg(reward_name).weight
        self._trigger_step: int | None = None

        # Validate log space constraints at init time
        terminal_weight = cfg.params["terminal_weight"]
        use_log_space = cfg.params.get("use_log_space", False)
        if use_log_space:
            if (self.start_weight > 0) != (terminal_weight > 0):
                raise ValueError(
                    f"For log space scaling, start_weight ({self.start_weight}) and "
                    f"terminal_weight ({terminal_weight}) must have the same sign"
                )
            if self.start_weight == 0 or terminal_weight == 0:
                raise ValueError(
                    f"For log space scaling, weights cannot be zero. "
                    f"start_weight={self.start_weight}, terminal_weight={terminal_weight}"
                )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        reward_name: str,
        prerequisite_curriculum: str,
        prerequisite_threshold: float,
        delay_steps: int,
        num_steps: int,
        terminal_weight: float,
        use_log_space: bool = False,
    ) -> float:
        """Ramp reward weight after a prerequisite curriculum is satisfied.

        Args:
            env: The learning environment.
            env_ids: Not used.
            reward_name: Reward term to update.
            prerequisite_curriculum: Name of the curriculum to watch.
            prerequisite_threshold: Value the prerequisite must reach (>=).
            delay_steps: Steps to wait after prerequisite is met before ramping.
            num_steps: Steps over which to ramp to terminal_weight.
            terminal_weight: Final reward weight.
            use_log_space: If True, interpolate in log space (both weights must have same sign).
        """
        # Check if prerequisite is met
        if self._trigger_step is None:
            prereq_value = env.curriculum_manager._curriculum_state.get(prerequisite_curriculum)
            if prereq_value is not None and prereq_value >= prerequisite_threshold:
                self._trigger_step = env.common_step_counter
            else:
                return self.start_weight

        ramp_start = self._trigger_step + delay_steps
        if env.common_step_counter <= ramp_start:
            return self.start_weight
        elif env.common_step_counter >= ramp_start + num_steps:
            env.reward_manager.get_term_cfg(reward_name).weight = terminal_weight
            return terminal_weight
        else:
            scale = (env.common_step_counter - ramp_start) / num_steps

            if use_log_space:
                abs_start = abs(self.start_weight)
                abs_terminal = abs(terminal_weight)
                log_weight = math.log(abs_start) + scale * (math.log(abs_terminal) - math.log(abs_start))
                abs_new_weight = math.exp(log_weight)
                new_weight = abs_new_weight if self.start_weight > 0 else -abs_new_weight
            else:
                new_weight = self.start_weight + (terminal_weight - self.start_weight) * scale

            env.reward_manager.get_term_cfg(reward_name).weight = new_weight
            return float(new_weight)


class update_action_scale_step(ManagerTermBase):
    """Curriculum to decay the action scale of a joint action term (linear or log-space)."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name = cfg.params["action_name"]
        action_term = env.action_manager._terms[action_name]
        self.start_scale: float = (
            float(action_term._scale)
            if isinstance(action_term._scale, int | float)
            else float(action_term._scale.mean())
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        action_name: str,
        start_step: int,
        num_steps: int,
        terminal_scale: float,
        use_log_space: bool = False,
    ) -> float:
        """Interpolate the action scale from its initial value to terminal_scale.

        Args:
            env: The learning environment.
            env_ids: Not used since all environments are affected.
            action_name: Name of the action term to update.
            start_step: Step count at which to begin decaying.
            num_steps: Number of steps over which to decay.
            terminal_scale: Final action scale value.
            use_log_space: If True, interpolate in log space so that equal time
                intervals produce equal *relative* changes (e.g. 0.1→0.05 takes
                the same time as 0.05→0.025).
        """
        action_term = env.action_manager._terms[action_name]

        if env.common_step_counter <= start_step:
            return self.start_scale
        elif env.common_step_counter > start_step + num_steps:
            action_term._scale = terminal_scale
            return terminal_scale
        else:
            t = (env.common_step_counter - start_step) / num_steps
            if use_log_space:
                log_start = math.log(self.start_scale)
                log_terminal = math.log(terminal_scale)
                new_scale = math.exp(log_start + t * (log_terminal - log_start))
            else:
                new_scale = self.start_scale + (terminal_scale - self.start_scale) * t
            action_term._scale = new_scale
            return new_scale


def switch_action_to_target_mode(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],  # noqa: ARG001
    action_name: str,
    switch_step: int,
) -> float:
    """Switch a SwitchableRelativeJointPositionAction from delta to target mode at a given step.

    Returns 1.0 after switching, 0.0 before.
    """
    action_term = env.action_manager._terms[action_name]
    if env.common_step_counter >= switch_step and not action_term.use_target_mode:
        action_term.switch_to_target_mode()
    return float(action_term.use_target_mode)


class update_episode_length_step(ManagerTermBase):
    """Curriculum that linearly ramps the max episode length (in seconds).

    Modifies ``env.cfg.episode_length_s`` at runtime so the built-in
    ``time_out`` termination picks up the new horizon automatically.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],  # noqa: ARG002
        start_step: int,
        num_steps: int,
        start_length_s: float,
        terminal_length_s: float,
    ) -> float:
        """Linearly interpolate episode length from start to terminal over num_steps.

        Args:
            start_step: Step to begin ramping.
            num_steps: Steps over which to ramp.
            start_length_s: Initial episode length in seconds.
            terminal_length_s: Final episode length in seconds.
        """
        if env.common_step_counter <= start_step:
            length_s = start_length_s
        elif env.common_step_counter > start_step + num_steps:
            length_s = terminal_length_s
        else:
            frac = (env.common_step_counter - start_step) / num_steps
            length_s = start_length_s + frac * (terminal_length_s - start_length_s)

        env.cfg.episode_length_s = length_s
        return length_s
