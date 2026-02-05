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

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DeltaJointPositionAction(JointAction):
    """Joint action term that applies the delta actions to the articulation's joints as position commands.

    actions = prev_targets + raw_actions * scale
    """

    cfg: actions_cfg.DeltaJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.DeltaJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # store the previous targets
        self._prev_targets = self._asset.data.joint_pos[:, self._joint_ids]

        self._steady_joint_ids, _ = self._asset.find_joints(self.cfg.steady_joint_names)
        self._asset.set_joint_position_target(
            self._asset.data.default_joint_pos[..., self._steady_joint_ids],
            joint_ids=self._steady_joint_ids,
        )

        # Build combined joint limits tensor
        # Priority: robot default limits -> clip (global) -> joint_limits (per-joint override)
        self._combined_limits: torch.Tensor = self._build_combined_limits_tensor()

    def _build_combined_limits_tensor(self) -> torch.Tensor:
        """Build a combined tensor of joint limits from clip, robot limits, and joint_limits config.

        Priority for each joint (from lowest to highest):
        1. Robot's default joint position limits (from asset)
        2. If clip is defined, use clip limits
        3. If joint is specified in joint_limits, use those limits (must be within default limits)

        Returns:
            Tensor of shape (1, num_joints, 2) containing [lower, upper] limits for each joint.
        """
        # Get joint names for this action term
        joint_names = [self._asset.joint_names[idx] for idx in self._joint_ids]

        # Step 1: Initialize with robot's default joint position limits
        # joint_pos_limits shape: (num_instances, num_joints, 2) where 2 is [lower, upper]
        default_limits = self._asset.data.joint_pos_limits[0, self._joint_ids, :].clone()  # (num_joints, 2)
        limits = default_limits.unsqueeze(0)  # (1, num_joints, 2)

        # Step 2: Override with clip limits if defined, clamped to respect default limits
        if self.cfg.clip is not None:
            # self._clip is already built by parent class with shape (1, num_joints, 2)
            # Clamp clip values to stay within robot's default limits
            default_lower = limits[..., 0].clone()
            default_upper = limits[..., 1].clone()
            limits[..., 0] = torch.clamp(self._clip[..., 0], min=default_lower, max=default_upper)
            limits[..., 1] = torch.clamp(self._clip[..., 1], min=default_lower, max=default_upper)

        # Step 3: Override with joint_limits for specific joints (assuming within default limits)
        if self.cfg.joint_limits is not None:
            for pattern, (lower, upper) in self.cfg.joint_limits.items():
                # Check if pattern is a regex or exact match
                for i, joint_name in enumerate(joint_names):
                    if re.fullmatch(pattern, joint_name) or pattern == joint_name:
                        # Clamp user-provided limits to be within the current limits
                        limits[0, i, 0] = max(lower, limits[0, i, 0].item())
                        limits[0, i, 1] = min(upper, limits[0, i, 1].item())

        return limits

    @property
    def active_joint_ids(self) -> torch.Tensor:
        """The joint ids of the joints that are active."""
        return self._scale[0].nonzero().view(-1)

    @property
    def prev_targets(self) -> torch.Tensor:
        """The previous targets of the joints."""
        return self._prev_targets

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self.prev_targets + self._raw_actions * self._scale
        # apply combined joint limits (robot limits -> clip -> joint_limits)
        self._processed_actions = torch.clamp(
            self._processed_actions, min=self._combined_limits[:, :, 0], max=self._combined_limits[:, :, 1]
        )
        # update the previous targets
        self._prev_targets = self._processed_actions.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_targets[env_ids] = self._asset.data.joint_pos[env_ids][..., self._joint_ids].clone()
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self) -> None:
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
