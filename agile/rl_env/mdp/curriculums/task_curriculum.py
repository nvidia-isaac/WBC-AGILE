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
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import RelativeJointPositionAction
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.terrains import TerrainImporter

from agile.rl_env.mdp import HarnessAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

        Note: This curriculum only makes sense if the action is a relative joint action
        Note: This curriculum changes the action clip range which means that the exported IO descriptor will not be correct.

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
