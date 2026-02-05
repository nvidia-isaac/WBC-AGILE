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
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.assets import retrieve_file_path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def load_torchscript_model(model_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """Load a TorchScript model from a file. Code borrowed from IsaacLab 2.3.0."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TorchScript model file not found: {model_path}")

    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print(f"Successfully loaded TorchScript model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        raise e


class AgileBasedLowerBodyAction(ActionTerm):
    """Action term that is based on Agile lower body RL policy."""

    cfg: ActionTermCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the agile action term."""
        super().__init__(cfg, env)

        # Save the observation config from cfg
        self._observation_cfg = env.cfg.observations
        self._obs_group_name = cfg.obs_group_name

        # Load policy here if needed
        _temp_policy_path = retrieve_file_path(cfg.policy_path)
        self._policy = load_torchscript_model(_temp_policy_path, device=env.device)
        self._env = env

        # Find joint ids for the lower body joints
        assert len(cfg.joint_names) > 0, "Joint names must be provided."
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)

        # Get the scale and offset from the configuration
        policy_output_scale = {
            name: scale
            for name, scale in cfg.policy_output_scale.items()
            if ("hip" in name or "knee" in name or "ankle" in name)
        }
        index_list, _, value_list = string_utils.resolve_matching_names_values(policy_output_scale, self._joint_names)
        self._policy_output_scale = torch.ones(self.num_envs, len(self._joint_names), device=self.device)
        self._policy_output_scale[:, index_list] = torch.tensor(value_list, device=self.device)

        self._policy_output_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # Create tensors to store raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

        # Base command names for clip parsing: [vx, vy, wz, height]
        self._base_command_names = ["vx", "vy", "wz", "height"]

        # parse clip for base command
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._base_command_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Lower Body Action: [vx, vy, wz, hip_height]."""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions with Agile policy."""
        return self._processed_actions

    def _compose_policy_input(self, base_command: torch.Tensor, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Compose the policy input by concatenating repeated commands with observations.

        Args:
            base_command: The base command tensor [vx, vy, wz, hip_height].
            obs_tensor: The observation tensor from the environment.

        Returns:
            The composed policy input tensor with repeated commands concatenated to observations.
        """
        # Get history length from observation configuration
        history_length = getattr(self._observation_cfg, self._obs_group_name).history_length
        history_length = history_length if history_length is not None else 1

        # Repeat commands based on history length and concatenate with observations
        repeated_commands = base_command.unsqueeze(1).repeat(1, history_length, 1).reshape(base_command.shape[0], -1)
        policy_input = torch.cat([repeated_commands, obs_tensor], dim=-1)

        return policy_input

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the input actions using the locomotion policy."""
        # Extract base command from the action tensor
        # Assuming the first 4 elements are the base command [vx, vy, wz, hip_height]
        base_command = actions[:, :4]

        # Clip base command if configured
        if self.cfg.clip is not None:
            base_command = torch.clamp(
                base_command,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        obs_tensor = self._env.obs_buf[self._obs_group_name]

        # Compose policy input using helper function
        policy_input = self._compose_policy_input(base_command, obs_tensor)

        with torch.inference_mode():
            joint_actions = self._policy(policy_input)

        self._raw_actions[:] = joint_actions

        # Apply scaling and offset to the raw actions from the policy
        self._processed_actions = joint_actions * self._policy_output_scale + self._policy_output_offset

    def apply_actions(self) -> None:
        """Apply the actions to the environment."""
        # Store the raw actions
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
