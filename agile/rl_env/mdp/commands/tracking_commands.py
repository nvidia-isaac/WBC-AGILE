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

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import yaml

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils import math as math_utils

# Only import the command class during type checking
if TYPE_CHECKING:
    from .tracking_commands_cfg import TrackingCommandCfg


class TrackingCommand(CommandTerm):
    """Command term that generates pose commands for tracking task."""

    cfg: TrackingCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: TrackingCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Cache env origins for later use (self._env.scene doesn't work in properties)
        self._env_origins = env.scene.env_origins

        # Find anchor body index if specified
        if cfg.anchor_body_name is not None:
            self._anchor_body_ids, _ = self.robot.find_bodies(cfg.anchor_body_name)
            if len(self._anchor_body_ids) == 0:
                raise ValueError(f"Anchor body '{cfg.anchor_body_name}' not found in robot.")
            self._anchor_body_id = self._anchor_body_ids[0]
        else:
            self._anchor_body_id = None

        # Get object reference if specified
        if cfg.object_name is not None:
            self.object: RigidObject = env.scene[cfg.object_name]
        else:
            self.object = None

        # Load motion data from YAML
        if not os.path.exists(cfg.file_path):
            raise FileNotFoundError(f"Motion file not found: {cfg.file_path}")
        with open(cfg.file_path) as f:
            motion_data = yaml.safe_load(f)
        qpos_data = torch.tensor(motion_data["qpos"], device=self.device)
        object_position = torch.tensor(motion_data["object_position"], device=self.device)
        object_wxyz = torch.tensor(motion_data["object_wxyz"], device=self.device)

        self.num_timesteps = len(qpos_data)
        self.pos_trajectory_w = qpos_data[:, :3] + torch.tensor(cfg.pos_offset, dtype=torch.float, device=self.device)
        self.quat_trajectory_w = qpos_data[:, 3:7]
        self.object_pos_trajectory_w = object_position + torch.tensor(
            cfg.pos_offset, dtype=torch.float, device=self.device
        )
        self.object_quat_trajectory_w = object_wxyz

        # Precompute object height peak for phase detection in rewards
        # This identifies when the object reaches maximum height (grasp/lift peak)
        object_heights = self.object_pos_trajectory_w[:, 2]
        self.object_height_peak_timestep = torch.argmax(object_heights).item()
        self.object_height_peak_value = object_heights[self.object_height_peak_timestep].item()

        # Tracked joints (e.g., upper body)
        # YAML file already has joint positions in Isaac order, so use tracked_joint_ids directly
        self.tracked_joint_ids, self.tracked_joint_names = self.robot.find_joints(self.cfg.joint_names)
        self.target_full_body_joint_pos = qpos_data[:, 7:]
        self.target_tracked_joint_pos = self.target_full_body_joint_pos[:, self.tracked_joint_ids]

        # Non-tracked joints (all other joints) - cache their default positions for reset
        all_joint_ids = list(range(self.robot.num_joints))
        tracked_joint_ids_set = set(self.tracked_joint_ids)
        self.other_joint_ids = [jid for jid in all_joint_ids if jid not in tracked_joint_ids_set]
        if len(self.other_joint_ids) > 0:
            self.default_other_joint_pos = self.robot.data.default_joint_pos[:, self.other_joint_ids].clone()
        else:
            self.default_other_joint_pos = None

        # -- buffer
        self.timestep_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_e = torch.zeros(self.num_envs, 4, device=self.device)
        self.tracked_joint_pos_command = torch.zeros(self.num_envs, len(self.tracked_joint_names), device=self.device)
        self.object_pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_quat_command_e = torch.zeros(self.num_envs, 4, device=self.device)

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_pos_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """String representation of the command term."""
        msg = "TrackingCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command_tracked_joint_pos(self) -> torch.Tensor:
        """The commanded tracked joint positions. Shape is (num_envs, num_tracked_joints)."""
        return self.tracked_joint_pos_command

    @property
    def command_anchor_pos_w(self) -> torch.Tensor:
        """The commanded anchor position in world frame. Shape is (num_envs, 3)."""
        return self.pos_command_e + self._env_origins

    @property
    def command_anchor_quat_w(self) -> torch.Tensor:
        """The commanded anchor quaternion in world frame. Shape is (num_envs, 4)."""
        return self.quat_command_e

    @property
    def command_anchor_yaw_w(self) -> torch.Tensor:
        """The commanded anchor yaw angle in world frame. Shape is (num_envs,)."""
        return math_utils.euler_xyz_from_quat(self.quat_command_e)[2]

    @property
    def command_object_pos_w(self) -> torch.Tensor:
        """The commanded object position in world frame. Shape is (num_envs, 3)."""
        return self.object_pos_command_e + self._env_origins

    @property
    def command_object_quat_w(self) -> torch.Tensor:
        """The commanded object quaternion in world frame. Shape is (num_envs, 4)."""
        return self.object_quat_command_e

    @property
    def object_pos_w(self) -> torch.Tensor:
        """The actual object position in world frame. Shape is (num_envs, 3).

        Returns zeros if object_name is not configured.
        """
        if self.object is not None:
            return self.object.data.root_pos_w
        return torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def object_quat_w(self) -> torch.Tensor:
        """The actual object quaternion in world frame. Shape is (num_envs, 4).

        Returns identity quaternion if object_name is not configured.
        """
        if self.object is not None:
            return self.object.data.root_quat_w
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """The robot anchor position in world frame. Shape is (num_envs, 3)."""
        if self._anchor_body_id is not None:
            return self.robot.data.body_pos_w[:, self._anchor_body_id, :]
        return self.robot.data.root_pos_w

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """The robot anchor quaternion in world frame. Shape is (num_envs, 4)."""
        if self._anchor_body_id is not None:
            return self.robot.data.body_quat_w[:, self._anchor_body_id, :]
        return self.robot.data.root_quat_w

    @property
    def robot_anchor_yaw_w(self) -> torch.Tensor:
        """The robot anchor yaw angle in world frame. Shape is (num_envs,)."""
        return math_utils.euler_xyz_from_quat(self.robot_anchor_quat_w)[2]

    @property
    def robot_tracked_joint_pos(self) -> torch.Tensor:
        """The robot tracked joint positions. Shape is (num_envs, num_tracked_joints)."""
        return self.robot.data.joint_pos[..., self.tracked_joint_ids]

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 38)."""
        return torch.cat(
            (
                self.command_anchor_pos_w,
                self.command_anchor_quat_w,
                self.command_tracked_joint_pos,
            ),
            dim=-1,
        )

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics."""
        self.metrics["position_error"] = torch.norm(self.robot.data.root_pos_w - self.command_anchor_pos_w, dim=1)

        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.robot.data.root_quat_w, self.quat_command_e
        )

        joint_pos_error = self.robot_tracked_joint_pos - self.tracked_joint_pos_command
        self.metrics["joint_pos_error"] = torch.norm(joint_pos_error, dim=1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command.

        Moved to reset_robot_to_trajectory in events.py to allow action manager to see the reset.
        """
        del env_ids  # unused

    def _update_command(self) -> None:
        """Update the command."""
        if self.cfg.update_goal_on_reach:
            pos_error = torch.norm(self.robot.data.root_pos_w - self.command_anchor_pos_w, dim=1)
            reached = pos_error < self.cfg.goal_reach_threshold
            self.timestep_counter[reached] += 1
        else:
            self.timestep_counter += 1

        # Clamp timestep counter to valid range to prevent index out of bounds
        self.timestep_counter = torch.clamp(self.timestep_counter, 0, self.num_timesteps - 1)

        self.pos_command_e = self.pos_trajectory_w[self.timestep_counter].float()
        self.quat_command_e = (
            math_utils.quat_unique(self.quat_trajectory_w[self.timestep_counter].float())
            if self.cfg.make_quat_unique
            else self.quat_trajectory_w[self.timestep_counter].float()
        )
        self.tracked_joint_pos_command = self.target_tracked_joint_pos[self.timestep_counter].float()

        self.object_pos_command_e = self.object_pos_trajectory_w[self.timestep_counter].float()
        self.object_quat_command_e = (
            math_utils.quat_unique(self.object_quat_trajectory_w[self.timestep_counter].float())
            if self.cfg.make_quat_unique
            else self.object_quat_trajectory_w[self.timestep_counter].float()
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set the debug visibility."""
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.object_goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
            self.object_goal_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)
            self.object_goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event: Any) -> None:
        """Visualize the goal marker."""
        del event  # unused
        # visualize the goal marker (convert from env frame to world frame)
        self.goal_pose_visualizer.visualize(
            translations=self.command_anchor_pos_w,
            orientations=self.command_anchor_quat_w,
        )
        self.object_goal_pose_visualizer.visualize(
            translations=self.command_object_pos_w,
            orientations=self.command_object_quat_w,
        )
