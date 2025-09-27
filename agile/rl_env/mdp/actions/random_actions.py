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

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg
    from .velocity_profiles import VelocityProfileBase

# Import velocity profile classes
from .velocity_profiles import (
    EMAVelocityProfile,
    EMAVelocityProfileCfg,
    LinearVelocityProfile,
    LinearVelocityProfileCfg,
    TrapezoidalVelocityProfile,
    TrapezoidalVelocityProfileCfg,
    VelocityProfileBaseCfg,
)


class RandomPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.RandomActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RandomActionCfg, env: ManagerBasedEnv):
        if bool(cfg.joint_names) + bool(cfg.joint_names_exclude) + bool(cfg.actuator_group) != 1:
            raise ValueError(
                "Only one of joint_names, joint_names_exclude, or actuator_group must be provided, but not both."
            )

        if cfg.joint_names_exclude:
            asset = env.scene[cfg.asset_name]
            _, joints_to_exclude = asset.find_joints(cfg.joint_names_exclude, preserve_order=cfg.preserve_order)

            joint_names = [name for name in asset.joint_names if name not in joints_to_exclude]
            cfg.joint_names = joint_names
            cfg.joint_names_exclude = []

        elif cfg.actuator_group:
            asset = env.scene[cfg.asset_name]
            if cfg.actuator_group not in asset.actuators:
                raise ValueError(
                    f"Actuator group {cfg.actuator_group} not found in asset {cfg.asset_name}. "
                    f"Available actuator groups: {list(asset.actuators.keys())}"
                )
            cfg.joint_names = asset.actuators[cfg.actuator_group].joint_names
            cfg.joint_names_exclude = []
        else:
            cfg.joint_names_exclude = []

        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._joint_limits = self._asset.data.joint_pos_limits[:, self._joint_ids].clone()
        self._velocity_limits = self._asset.data.joint_vel_limits[:, self._joint_ids].clone()
        self._sample_range = cfg.sample_range

        # Override joint limits if specified in config
        if cfg.joint_pos_limits is not None:
            for joint_name, (lower_limit, upper_limit) in cfg.joint_pos_limits.items():
                if joint_name in cfg.joint_names:
                    joint_idx = cfg.joint_names.index(joint_name)
                    self._joint_limits[:, joint_idx, 0] = lower_limit
                    self._joint_limits[:, joint_idx, 1] = upper_limit
                else:
                    raise ValueError(
                        f"Joint '{joint_name}' specified in joint_pos_limits not found in action term's joint list. "
                        f"Available joints: {cfg.joint_names}"
                    )

        # Count the time since the last sample
        self._time_since_last_sample = torch.zeros(self._env.num_envs, device=self._asset.device)
        # The time between samples is also random
        self._time_to_resample_sample = (
            torch.rand(self._env.num_envs, device=self._asset.device) * (self._sample_range[1] - self._sample_range[0])
            + self._sample_range[0]
        )

        # Create velocity profile based on configuration
        self._velocity_profile = self._create_velocity_profile(
            cfg.velocity_profile_cfg,
            num_envs=self._env.num_envs,
            num_joints=self._num_joints,
            device=self._asset.device,
            joint_limits=self._joint_limits,
            velocity_limits=self._velocity_limits,
        )

        self._processed_actions = self._offset.clone()
        self._target_processed_actions = self._offset.clone()

        # Do not export the IO descriptor for this action term.
        self._export_IO_descriptor = False

    def _create_velocity_profile(self, profile_cfg: VelocityProfileBaseCfg, **kwargs: Any) -> VelocityProfileBase:
        """Factory method to create appropriate velocity profile.

        Args:
            profile_cfg: Configuration for the velocity profile.
            **kwargs: Additional arguments passed to profile constructor.

        Returns:
            Instance of the appropriate velocity profile.

        Raises:
            ValueError: If unknown profile configuration type.
        """
        if isinstance(profile_cfg, EMAVelocityProfileCfg):
            return EMAVelocityProfile(profile_cfg, **kwargs)
        elif isinstance(profile_cfg, LinearVelocityProfileCfg):
            return LinearVelocityProfile(profile_cfg, **kwargs)
        elif isinstance(profile_cfg, TrapezoidalVelocityProfileCfg):
            return TrapezoidalVelocityProfile(profile_cfg, **kwargs)
        else:
            raise ValueError(f"Unknown profile config type: {type(profile_cfg)}")

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 0  # no action is applied, just randomization

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        """Sample random actions for the joints in the action term."""
        self._time_since_last_sample += self._env.step_dt
        resample_action_mask = self._time_since_last_sample > self._time_to_resample_sample
        self._time_since_last_sample[resample_action_mask] = 0.0
        self._time_to_resample_sample[resample_action_mask] = (
            torch.rand(resample_action_mask.sum(), device=self._asset.device)  # type: ignore[call-overload]
            * (self._sample_range[1] - self._sample_range[0])
            + self._sample_range[0]
        )

        # Sample random joint position targets within the joint limits
        if resample_action_mask.any():
            # Sample new target positions
            new_targets = (
                torch.rand(
                    resample_action_mask.sum(),
                    self._num_joints,
                    device=self._asset.device,
                )  # type: ignore[call-overload]
                * (self._joint_limits[resample_action_mask, :, 1] - self._joint_limits[resample_action_mask, :, 0])
                + self._joint_limits[resample_action_mask, :, 0]
            )

            self._target_processed_actions[resample_action_mask] = new_targets

            # Get current positions for environments that need new targets
            current_positions = self._processed_actions[resample_action_mask]

            # Set new targets in velocity profile
            env_ids = resample_action_mask.nonzero().squeeze(-1)
            self._velocity_profile.set_target(current_positions, new_targets, env_ids=env_ids)

        # Handle no_random_when_walking logic
        command_term = self._env.command_manager.get_term(self.cfg.command_name)
        is_null_cmd = (command_term.command[:, :3] == 0).all(dim=1)
        if self.cfg.no_random_when_walking:
            # Set target to default positions for environments that are walking (not null command)
            is_walking = ~is_null_cmd
            if is_walking.any():
                walking_ids = is_walking.nonzero().squeeze(-1)
                self._target_processed_actions[is_walking] = self._offset[is_walking]
                # Update velocity profile targets for walking environments
                self._velocity_profile.set_target(
                    self._processed_actions[is_walking], self._offset[is_walking], env_ids=walking_ids
                )

        # Update positions using velocity profile
        self._processed_actions = self._velocity_profile.compute_next_position(dt=self._env.step_dt)

    def apply_actions(self) -> None:
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
