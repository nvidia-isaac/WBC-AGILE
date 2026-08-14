# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Curriculum terms for progressive pose tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


class progressive_body_tracking_curriculum(ManagerTermBase):
    """Curriculum that progressively enables tracking for different body parts.

    This curriculum starts with only torso tracking, then adds hands one by one
    as the robot improves. This is done by scaling the reward weights.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Store initial reward weights
        self.reward_terms = cfg.params["reward_terms"]
        self.initial_weights = {}
        for term_name in self.reward_terms:
            self.initial_weights[term_name] = env.reward_manager.get_term_cfg(term_name).weight

        # Initialize stages
        self.current_stage = 0
        self.stages = cfg.params.get(
            "stages",
            [
                {"torso_position": 1.0, "torso_yaw": 1.0},  # Stage 0: Only torso
                {
                    "torso_position": 1.0,
                    "torso_yaw": 1.0,
                    "left_hand_position": 0.5,
                    "left_hand_orientation": 0.5,
                },  # Stage 1: Add left hand
                {
                    "torso_position": 1.0,
                    "torso_yaw": 1.0,
                    "left_hand_position": 1.0,
                    "left_hand_orientation": 1.0,
                    "right_hand_position": 0.5,
                    "right_hand_orientation": 0.5,
                },  # Stage 2: Full left, add right
                {
                    "torso_position": 1.0,
                    "torso_yaw": 1.0,
                    "left_hand_position": 1.0,
                    "left_hand_orientation": 1.0,
                    "right_hand_position": 1.0,
                    "right_hand_orientation": 1.0,
                },  # Stage 3: Everything full
            ],
        )

        # Performance tracking
        self.stage_successes = 0
        self.stage_failures = 0

        # Apply initial stage
        self._apply_stage(0)

    def _apply_stage(self, stage_idx: int) -> None:
        """Apply reward weights for a specific stage."""
        if stage_idx >= len(self.stages):
            stage_idx = len(self.stages) - 1

        stage_weights = self.stages[stage_idx]

        # Set all tracked rewards to zero first
        for term_name in self.reward_terms:
            env = self._env
            env.reward_manager.get_term_cfg(term_name).weight = 0.0

        # Then set the active ones for this stage
        for term_name, scale in stage_weights.items():
            if term_name in self.initial_weights:
                env = self._env
                new_weight = self.initial_weights[term_name] * scale
                env.reward_manager.get_term_cfg(term_name).weight = new_weight

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        reward_terms: list[str],  # noqa: ARG002
        position_error_threshold: float = 0.1,
        orientation_error_threshold: float = 0.3,
        successes_to_advance: int = 100,
        failures_to_retreat: int = 50,
        stages: list[dict] | None = None,
    ) -> int:
        """Progressively enable body part tracking based on performance.

        Args:
            env: The environment instance.
            env_ids: Environment IDs being reset.
            command_name: Name of the command term.
            reward_terms: List of reward term names to control.
            position_error_threshold: Position error threshold for success.
            orientation_error_threshold: Orientation error threshold for success.
            successes_to_advance: Number of successful episodes to advance stage.
            failures_to_retreat: Number of failed episodes to go back a stage.
            stages: Optional custom stage definitions.

        Returns:
            Current stage index.
        """
        if stages is not None:
            self.stages = stages

        # Get tracking errors for current stage
        command_term = env.command_manager.get_term(command_name)
        metrics = command_term.metrics

        # Determine success based on active body parts in current stage
        stage_config = self.stages[self.current_stage]

        # Check performance for active body parts
        is_successful = True

        if "torso_position" in stage_config and stage_config["torso_position"] > 0:
            if metrics["torso_position_error"][env_ids].mean() > position_error_threshold:
                is_successful = False

        if "torso_yaw" in stage_config and stage_config["torso_yaw"] > 0:
            if (
                metrics.get("torso_yaw_error", metrics.get("torso_orientation_error", torch.zeros(len(env_ids))))[
                    env_ids
                ].mean()
                > orientation_error_threshold
            ):
                is_successful = False

        if "left_hand_position" in stage_config and stage_config["left_hand_position"] > 0.9:
            if metrics["left_hand_position_error"][env_ids].mean() > position_error_threshold:
                is_successful = False

        if "left_hand_orientation" in stage_config and stage_config["left_hand_orientation"] > 0.9:
            if metrics["left_hand_orientation_error"][env_ids].mean() > orientation_error_threshold:
                is_successful = False

        if "right_hand_position" in stage_config and stage_config["right_hand_position"] > 0.9:
            if metrics["right_hand_position_error"][env_ids].mean() > position_error_threshold:
                is_successful = False

        if "right_hand_orientation" in stage_config and stage_config["right_hand_orientation"] > 0.9:
            if metrics["right_hand_orientation_error"][env_ids].mean() > orientation_error_threshold:
                is_successful = False

        # Update success/failure counts
        if is_successful:
            self.stage_successes += len(env_ids)
            self.stage_failures = 0  # Reset failures on success
        else:
            self.stage_failures += len(env_ids)

        # Check for stage transition
        if self.stage_successes >= successes_to_advance and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.stage_successes = 0
            self.stage_failures = 0
            self._apply_stage(self.current_stage)
            print(f"[Curriculum] Advanced to stage {self.current_stage}")
        elif self.stage_failures >= failures_to_retreat and self.current_stage > 0:
            self.current_stage -= 1
            self.stage_successes = 0
            self.stage_failures = 0
            self._apply_stage(self.current_stage)
            print(f"[Curriculum] Retreated to stage {self.current_stage}")

        return self.current_stage
