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

"""Action processing."""

import torch

from agile.sim2mujoco.simulation import JointCommand


class ActionTerm:
    """Single action term (e.g., JointPositionAction)."""

    def __init__(self, name: str, config: dict, full_joint_names: list[str], device: torch.device):
        """
        Initialize action term.

        Args:
            name: Name of the action term.
            config: Configuration dictionary for this term.
            full_joint_names: List of all joint names from simulation.
            device: Torch device.
        """
        self.name = name
        self.config = config
        self.device = device

        # Parse action configuration.
        self.action_joint_names = config.get("joint_names", [])

        # Convert scale to tensor if it's a list or nested list
        scale_raw = config.get("scale", 1.0)
        if isinstance(scale_raw, list | tuple):
            # Flatten if nested (e.g., [[1, 2, 3]] -> [1, 2, 3])
            if isinstance(scale_raw[0], list | tuple):
                scale_raw = scale_raw[0]
            self.scale = torch.tensor(scale_raw, device=device, dtype=torch.float32)
        else:
            self.scale = float(scale_raw)  # Keep as scalar if single value

        self.offset_list = config.get("offset", [0.0] * len(self.action_joint_names))
        self.offset = torch.tensor(self.offset_list, device=device, dtype=torch.float32)
        self.clip = config.get("clip", None)

        # Map action joints to full robot joint indices.
        self.joint_indices = [full_joint_names.index(jn) for jn in self.action_joint_names]

        # Action dimension.
        self.action_dim = len(self.action_joint_names)

    def process(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Process raw actions to joint positions.

        Args:
            actions: Raw actions from policy, shape (action_dim,).

        Returns:
            Processed joint positions, shape (action_dim,).
        """
        # Apply clipping if specified.
        if self.clip is not None:
            if isinstance(self.clip[0], list):
                # Per-joint clipping.
                clip_min = torch.tensor([c[0] for c in self.clip], device=self.device, dtype=torch.float32)
                clip_max = torch.tensor([c[1] for c in self.clip], device=self.device, dtype=torch.float32)
                actions = torch.clamp(actions, clip_min, clip_max)
            else:
                # Uniform clipping.
                actions = torch.clamp(actions, self.clip[0], self.clip[1])

        # Apply scale and offset.
        joint_positions = actions * self.scale + self.offset

        return joint_positions


class ActionProcessor:
    """Flexible action processor that handles arbitrary action terms."""

    def __init__(self, config: dict, joint_names: list[str], device: torch.device):
        """
        Initialize action processor.

        Args:
            config: Configuration dictionary from YAML.
            joint_names: List of joint names from simulation.
            device: Torch device.
        """
        self.device = device
        self.joint_names = joint_names
        self.num_joints = len(joint_names)

        # Parse action terms from YAML.
        self.action_terms = []
        action_configs = config["actions"]

        for action_config in action_configs:
            term = ActionTerm(action_config["name"], action_config, joint_names, device)

            # Validate joint ordering
            if hasattr(term, "joint_indices") and term.joint_indices is not None:
                # Verify joints exist and ordering is as expected
                actual_joint_order = [joint_names[idx] for idx in term.joint_indices]
                if actual_joint_order != term.action_joint_names:
                    raise ValueError(
                        f"❌ ERROR: Joint ordering mismatch for action term '{term.name}'!\n"
                        f"   YAML specifies: {term.action_joint_names[:5]}...\n"
                        f"   Mapping gives:  {actual_joint_order[:5]}...\n"
                        f"   Joint names in YAML don't match MJCF."
                    )

            self.action_terms.append(term)

        # Total action dimension.
        self.total_action_dim = sum(term.action_dim for term in self.action_terms)

        # Compute action ranges for slicing.
        action_dims = [term.action_dim for term in self.action_terms]
        action_cumsum = [0] + list(torch.cumsum(torch.tensor(action_dims), dim=0).tolist())
        self.action_ranges = [(int(action_cumsum[i]), int(action_cumsum[i + 1])) for i in range(len(action_dims))]

        # Load PD gains and default positions.
        robot_config = config["articulations"]["robot"]
        self.default_joint_pos = self._parse_joint_values(
            robot_config["default_joint_pos"], robot_config["joint_names"], joint_names
        )
        self.kp = self._parse_joint_values(
            robot_config["default_joint_stiffness"], robot_config["joint_names"], joint_names
        )
        self.kd = self._parse_joint_values(
            robot_config["default_joint_damping"], robot_config["joint_names"], joint_names
        )

    def _parse_joint_values(
        self, values: list, yaml_joint_names: list[str], mjcf_joint_names: list[str]
    ) -> torch.Tensor:
        """
        Map joint values from YAML order to MJCF order.

        Args:
            values: List of values in YAML joint order.
            yaml_joint_names: Joint names from YAML.
            mjcf_joint_names: Joint names from MJCF (simulation).

        Returns:
            Tensor of values in MJCF order.
        """
        # Initialize with zeros.
        result = torch.zeros(len(mjcf_joint_names), device=self.device, dtype=torch.float32)

        # Map values.
        for yaml_idx, joint_name in enumerate(yaml_joint_names):
            if joint_name in mjcf_joint_names:
                mjcf_idx = mjcf_joint_names.index(joint_name)
                result[mjcf_idx] = values[yaml_idx]

        return result

    def process(self, actions: torch.Tensor) -> JointCommand:
        """
        Process policy actions to joint commands.

        Args:
            actions: Raw actions from policy, shape (total_action_dim,).

        Returns:
            JointCommand with positions, kp, kd for all joints.
        """
        # Start with zeros
        joint_positions = torch.zeros_like(self.default_joint_pos)

        # Apply each action term.
        for term, (start, end) in zip(self.action_terms, self.action_ranges, strict=False):
            action_slice = actions[start:end]
            processed = term.process(action_slice)

            # Write to appropriate joint indices.
            joint_positions[term.joint_indices] = processed

        return JointCommand(position=joint_positions, kp=self.kp, kd=self.kd)
