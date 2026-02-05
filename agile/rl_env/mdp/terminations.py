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


import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import subtract_frame_transforms

from agile.rl_env.mdp.utils import get_robot_cfg


def illegal_ground_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """Terminate when the contact force exceeds the force threshold and the asset is below the min_height."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold

    asset_height = torch.min(env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2], dim=1)[0]
    on_ground = asset_height < min_height

    in_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )

    return in_contact & on_ground


def illegal_base_height(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_measurement_sensor"),
) -> torch.Tensor:
    """Terminate if the base height is below the threshold."""
    robot, _ = get_robot_cfg(env, asset_cfg)
    sensor: RayCaster = env.scene[sensor_cfg.name]
    base_height = robot.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    return base_height < height_threshold


def link_distance(
    env: ManagerBasedRLEnv,
    min_distance_threshold: float = 0.05,
    max_distance_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Terminate if the distance between the two links is outside the allowed range.

    Args:
        env: Environment instance
        min_distance_threshold: Minimum distance threshold. Terminate if links closer than this.
        max_distance_threshold: Maximum distance threshold. Terminate if links farther than this. None to disable.
        asset_cfg: Asset configuration (must specify exactly 2 links)

    Returns:
        Boolean tensor indicating which environments should terminate
    """
    robot, _ = get_robot_cfg(env, asset_cfg)
    link_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]

    assert len(asset_cfg.body_ids) == 2, "Link distance is only supported for 2 links"
    link_distance = torch.norm(link_pos[:, 0] - link_pos[:, 1], dim=1)

    # Check minimum distance
    too_close = link_distance < min_distance_threshold

    # Check maximum distance if specified
    if max_distance_threshold is not None:
        too_far = link_distance > max_distance_threshold
        return too_close | too_far

    return too_close


class standing(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.standing_timer = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        min_height: float,
        duration_s: float,
        sensor_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """Terminate if the robot stands for a given time."""

        asset: RigidObject = env.scene[asset_cfg.name]
        if sensor_cfg is not None:
            sensor: RayCaster = env.scene[sensor_cfg.name]
            # Adjust the target height using the sensor data
            current_height = asset.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        else:
            # Use the provided target height directly for flat terrain
            current_height = asset.data.root_pos_w[:, 2]

        is_standing = current_height > min_height

        self.standing_timer += 1
        self.standing_timer[~is_standing] = 0

        return self.standing_timer > int(duration_s / env.step_dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs)
        self.standing_timer[env_ids] = 0


def bad_base_pose(
    env: ManagerBasedRLEnv,
    base_pos_threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the robot is away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    base_pos_error = torch.norm(command.command_anchor_pos_w - command.robot_anchor_pos_w, dim=-1)
    return base_pos_error > base_pos_threshold


def bad_base_rotation(
    env: ManagerBasedRLEnv,
    base_ori_threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the robot is away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    base_ori_error = math_utils.quat_error_magnitude(command.command_anchor_quat_w, command.robot_anchor_quat_w)
    return base_ori_error > base_ori_threshold


def bad_joint_pos(
    env: ManagerBasedRLEnv,
    joint_pos_threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the robot is away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    joint_pos_error = torch.norm(command.command_tracked_joint_pos - command.robot_tracked_joint_pos, dim=-1)
    return joint_pos_error > joint_pos_threshold


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},  # noqa: B006
    reference_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range.
        reference_asset_cfg: Optional reference asset configuration with body_names specified.
            If provided, the body_names must resolve to exactly one body to use as reference frame.

    Raises:
        ValueError: If reference_asset_cfg is provided but body_names is not specified or resolves to multiple bodies.
    """
    object: RigidObject = env.scene[asset_cfg.name]
    # Default to (-inf, inf) for unspecified dimensions - always in bounds
    range_list = [in_bound_range.get(key, (float("-inf"), float("inf"))) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    # Get object position in world frame
    object_pos_w = object.data.root_pos_w

    # Transform object position to reference frame
    if reference_asset_cfg is not None:
        # Use asset link as reference frame
        reference_asset: Articulation = env.scene[reference_asset_cfg.name]
        # Get the body IDs from the reference_asset_cfg
        body_ids = reference_asset_cfg.body_ids
        if body_ids is None or len(body_ids) == 0:
            raise ValueError(
                f"reference_asset_cfg must have body_names specified. "
                f"Please provide body_names in SceneEntityCfg for asset '{reference_asset_cfg.name}'."
            )
        if len(body_ids) != 1:
            raise ValueError(
                f"reference_asset_cfg must resolve to exactly one body, but got {len(body_ids)} bodies: {body_ids}. "
                f"Please specify a single body name in reference_asset_cfg."
            )
        body_id = body_ids[0]
        # Get the reference link pose in world frame
        reference_pos_w = reference_asset.data.body_pos_w[:, body_id, :]
        reference_quat_w = reference_asset.data.body_quat_w[:, body_id, :]
        # Transform object position from world frame to reference link frame
        object_pos_local, _ = subtract_frame_transforms(reference_pos_w, reference_quat_w, object_pos_w)
    else:
        # Use environment origins as reference frame (no rotation)
        object_pos_local = object_pos_w - env.scene.env_origins

    # Check if object is outside bounds
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds
