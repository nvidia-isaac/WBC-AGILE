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

"""Observation computation with history support."""

import torch

from agile.sim2mujoco.utils import quat_rotate_inverse


class HistoryBuffer:
    """Circular buffer for observation history."""

    def __init__(self, obs_dim: int, history_length: int, device: torch.device):
        """
        Initialize history buffer.

        Args:
            obs_dim: Dimension of the observation.
            history_length: Number of history steps (0 means no history).
            device: Torch device.
        """
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.device = device

        if history_length > 0:
            # Buffer shape: (history_length, obs_dim).
            self.buffer = torch.zeros(history_length, obs_dim, device=device)
            self.index = 0
            self.filled = False
        else:
            self.buffer = None

    def push(self, obs: torch.Tensor):
        """Add new observation to history."""
        if self.history_length == 0:
            return

        self.buffer[self.index] = obs
        self.index = (self.index + 1) % self.history_length

        if self.index == 0:
            self.filled = True

    def get(self, flatten: bool = True) -> torch.Tensor:
        """
        Get current history.

        Returns:
            If history_length == 0: returns None (should not be called).
            If history_length > 0 and flatten: (history_length * obs_dim,).
            If history_length > 0 and not flatten: (history_length, obs_dim).

        Note:
            Always returns the full history_length dimension, padding with zeros
            if the buffer is not yet fully filled. This ensures consistent dimensions
            for the policy from the very first step.
        """
        if self.history_length == 0:
            raise ValueError("Cannot get history when history_length is 0")

        # Always return full history_length dimension.
        # Return in chronological order (oldest to newest).
        if self.filled:
            # Buffer is full, reorder to get chronological order
            ordered = torch.cat([self.buffer[self.index :], self.buffer[: self.index]], dim=0)
        else:
            # Buffer not fully filled yet, return full buffer (zeros + filled entries)
            # The buffer is already initialized with zeros, so unfilled entries are zero
            ordered = self.buffer

        if flatten:
            return ordered.flatten()
        else:
            return ordered

    def reset(self):
        """Reset history buffer to zeros."""
        if self.history_length > 0:
            self.buffer.zero_()
            self.index = 0
            self.filled = False


class ObservationTerm:
    """Single observation term with history support."""

    def __init__(self, name: str, config: dict, joint_names: list[str], device: torch.device):
        """
        Initialize observation term.

        Args:
            name: Name of the observation term.
            config: Configuration dictionary for this term.
            joint_names: List of all joint names from simulation.
            device: Torch device.
        """
        self.name = name
        self.config = config
        self.device = device

        # Parse overloads.
        overloads = config.get("overloads", {})
        self.history_length = overloads.get("history_length", 0)
        self.flatten_history = overloads.get("flatten_history_dim", True)
        self.clip = overloads.get("clip", None)
        self.scale = overloads.get("scale", None)

        # Determine observation dimension.
        self.obs_dim = self._compute_obs_dim(config)

        # Create history buffer.
        self.history = HistoryBuffer(self.obs_dim, self.history_length, device)

        # Store joint indices if needed.
        if "joint_names" in config:
            term_joint_names = config["joint_names"]
            self.joint_names = term_joint_names  # Store for debugging/validation
            self.joint_indices = [joint_names.index(jn) for jn in term_joint_names]

            # Handle position offsets if present.
            if "joint_pos_offsets" in config:
                offsets = config["joint_pos_offsets"]
                self.joint_offsets = torch.tensor(offsets, device=device, dtype=torch.float32)
            else:
                self.joint_offsets = None

            # Handle velocity offsets if present.
            if "joint_vel_offsets" in config:
                vel_offsets = config["joint_vel_offsets"]
                self.joint_vel_offsets = torch.tensor(vel_offsets, device=device, dtype=torch.float32)
            else:
                self.joint_vel_offsets = None
        else:
            self.joint_names = None
            self.joint_indices = None
            self.joint_offsets = None
            self.joint_vel_offsets = None

        # Compute function will be assigned by the factory.
        self.compute_raw_fn = None

    def _compute_obs_dim(self, config: dict) -> int:
        """Compute the base observation dimension (without history)."""
        shape = config.get("shape", [1])
        if isinstance(shape, list):
            return shape[0] if len(shape) == 1 else int(torch.prod(torch.tensor(shape)).item())
        return shape

    def compute(self, sim_state) -> torch.Tensor:
        """
        Compute observation with history, scaling, and clipping.

        Args:
            sim_state: Current simulation state.

        Returns:
            Tensor of shape (obs_dim,) if history_length == 0.
            Tensor of shape (history_length * obs_dim,) if history_length > 0.
        """
        # Compute raw observation.
        raw_obs = self.compute_raw_fn(sim_state)

        # Apply scaling.
        if self.scale is not None:
            if isinstance(self.scale, list):
                scale_tensor = torch.tensor(self.scale, device=self.device, dtype=torch.float32)
                raw_obs = raw_obs * scale_tensor
            else:
                raw_obs = raw_obs * self.scale

        # Apply clipping.
        if self.clip is not None:
            raw_obs = torch.clamp(raw_obs, self.clip[0], self.clip[1])

        # Handle history.
        if self.history_length == 0:
            return raw_obs
        else:
            self.history.push(raw_obs)
            return self.history.get(flatten=self.flatten_history)

    def reset(self):
        """Reset history buffer."""
        self.history.reset()

    def output_dim(self) -> int:
        """Get the final output dimension (including history)."""
        if self.history_length == 0:
            return self.obs_dim
        else:
            return self.obs_dim * self.history_length


class ObservationProcessor:
    """Flexible observation processor that handles arbitrary observation terms."""

    def __init__(self, config: dict, joint_names: list[str], device: torch.device, command_manager=None):
        """
        Initialize observation processor.

        Args:
            config: Configuration dictionary from YAML.
            joint_names: List of joint names from simulation.
            device: Torch device.
            command_manager: Optional CommandManager for interactive commands.
        """
        self.device = device
        self.joint_names = joint_names
        self.command_manager = command_manager
        self.terms = []

        # Parse all observation terms from YAML.
        obs_policy = config["observations"]["policy"]

        for obs_config in obs_policy:
            term_name = obs_config["name"]
            expected_shape = obs_config.get("shape", [1])
            base_dim = expected_shape[0] if isinstance(expected_shape, list) else expected_shape

            # Account for history in expected dimension
            overloads = obs_config.get("overloads", {})
            history_length = overloads.get("history_length", 0)
            if history_length > 0:
                expected_dim = base_dim * history_length
            else:
                expected_dim = base_dim

            # Create observation term.
            term = self._create_term(term_name, obs_config, joint_names, device)

            if term is None:
                # Error instead of warning: unsupported observation term
                raise ValueError(
                    f"❌ ERROR: Unsupported observation term '{term_name}' found in config.\n"
                    f"   Available terms: projected_gravity, base_ang_vel, base_lin_vel, "
                    f"joint_pos_rel, joint_pos, joint_vel, joint_vel_rel, last_action, "
                    f"navigation_command, locomotion_command, velocity_and_height_command, "
                    f"generated_commands, zero_padding"
                )

            # Validate dimension matches expected shape from YAML (including history)
            actual_dim = term.output_dim()
            if actual_dim != expected_dim:
                history_info = f" × history_length={history_length}" if history_length > 0 else ""
                raise ValueError(
                    f"❌ ERROR: Dimension mismatch for observation term '{term_name}'!\n"
                    f"   Expected dimension: {expected_dim} (from YAML shape: {expected_shape}{history_info})\n"
                    f"   Actual dimension:   {actual_dim}\n"
                    f"   This may indicate a mismatch between the config and the robot model."
                )

            # Validate joint ordering for joint-based observations
            if hasattr(term, "joint_indices") and term.joint_indices is not None:
                if hasattr(term, "joint_names") and term.joint_names:
                    # Verify that joint_indices maps correctly to joint_names
                    actual_joint_order = [joint_names[idx] for idx in term.joint_indices]
                    if actual_joint_order != term.joint_names:
                        raise ValueError(
                            f"❌ ERROR: Joint ordering mismatch for observation term '{term_name}'!\n"
                            f"   YAML specifies these joints (in order): {term.joint_names[:5]}...\n"
                            f"   But mapping produces this order:        {actual_joint_order[:5]}...\n"
                            f"   This indicates the joint names in YAML don't match MJCF."
                        )

            self.terms.append(term)

        # Compute total observation dimension.
        self.total_obs_dim = sum(term.output_dim() for term in self.terms)

        # Store last action for last_action terms.
        self.last_action = None

    def _create_term(self, name: str, config: dict, joint_names: list[str], device: torch.device) -> ObservationTerm:
        """
        Factory method to create observation terms.

        Args:
            name: Name of the observation term.
            config: Configuration for the term.
            joint_names: List of joint names.
            device: Torch device.

        Returns:
            ObservationTerm instance or None if unknown.
        """
        # Create term.
        term = ObservationTerm(name, config, joint_names, device)

        # Assign compute function based on name.
        if name == "projected_gravity":
            term.compute_raw_fn = lambda sim_state: self._compute_projected_gravity(sim_state)
        elif name == "base_ang_vel":
            term.compute_raw_fn = lambda sim_state: self._compute_base_ang_vel(sim_state)
        elif name == "base_lin_vel":
            term.compute_raw_fn = lambda sim_state: self._compute_base_lin_vel(sim_state)
        elif name == "joint_pos_rel":
            term.compute_raw_fn = lambda sim_state: self._compute_joint_pos_rel(term, sim_state)
        elif name == "joint_pos":
            term.compute_raw_fn = lambda sim_state: self._compute_joint_pos(term, sim_state)
        elif name == "joint_vel":
            term.compute_raw_fn = lambda sim_state: self._compute_joint_vel(term, sim_state)
        elif name == "joint_vel_rel":
            term.compute_raw_fn = lambda sim_state: self._compute_joint_vel_rel(term, sim_state)
        elif name == "last_action":
            term.compute_raw_fn = lambda sim_state: self._compute_last_action(term)
        elif name in ["navigation_command", "locomotion_command"]:
            term.compute_raw_fn = lambda sim_state: self._compute_navigation_command()
        elif name == "zero_padding":
            term.compute_raw_fn = lambda sim_state: self._compute_zero_padding(term)
        elif name in ["velocity_and_height_command", "generated_commands"]:
            term.compute_raw_fn = lambda sim_state: self._compute_velocity_height_command()
        else:
            return None

        return term

    def _compute_projected_gravity(self, sim_state) -> torch.Tensor:
        """Compute gravity vector in robot frame."""
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        # Rotate by inverse of root quaternion.
        gravity_robot = quat_rotate_inverse(sim_state.root_quat.float(), gravity_world)
        return gravity_robot

    def _compute_base_ang_vel(self, sim_state) -> torch.Tensor:
        """Compute base angular velocity."""
        return sim_state.root_ang_vel.float()

    def _compute_base_lin_vel(self, sim_state) -> torch.Tensor:
        """Compute base linear velocity."""
        return sim_state.root_lin_vel.float()

    def _compute_joint_pos_rel(self, term: ObservationTerm, sim_state) -> torch.Tensor:
        """Compute relative joint positions (with offsets).

        Note: This should match IsaacLab's joint_pos_rel which computes:
              joint_pos - default_joint_pos
        The joint_offsets in the config should equal default_joint_pos for the controlled joints.
        """
        joint_pos = sim_state.joint_pos[term.joint_indices]
        if term.joint_offsets is not None:
            result = (joint_pos - term.joint_offsets).float()
            return result
        return joint_pos.float()

    def _compute_joint_pos(self, term: ObservationTerm, sim_state) -> torch.Tensor:
        """Compute absolute joint positions."""
        return sim_state.joint_pos[term.joint_indices].float()

    def _compute_joint_vel(self, term: ObservationTerm, sim_state) -> torch.Tensor:
        """Compute joint velocities."""
        return sim_state.joint_vel[term.joint_indices].float()

    def _compute_joint_vel_rel(self, term: ObservationTerm, sim_state) -> torch.Tensor:
        """Compute relative joint velocities (with offsets)."""
        joint_vel = sim_state.joint_vel[term.joint_indices]
        if term.joint_vel_offsets is not None:
            return (joint_vel - term.joint_vel_offsets).float()
        return joint_vel.float()

    def _compute_last_action(self, term: ObservationTerm) -> torch.Tensor:
        """Return last action."""
        if self.last_action is None:
            return torch.zeros(term.obs_dim, device=self.device, dtype=torch.float32)
        return self.last_action.float()

    def _compute_navigation_command(self) -> torch.Tensor:
        """Return navigation command (3D: vx, vy, wz)."""
        if self.command_manager is not None:
            return self.command_manager.get_navigation_command()
        # Fallback to zeros (backward compatibility)
        return torch.zeros(3, device=self.device, dtype=torch.float32)

    def _compute_velocity_height_command(self) -> torch.Tensor:
        """4D command: [lin_vel_x, lin_vel_y, ang_vel_z, height]."""
        if self.command_manager is not None:
            return self.command_manager.get_command()
        # Fallback to default (backward compatibility)
        return torch.tensor([0.0, 0.0, 0.0, 0.72], device=self.device, dtype=torch.float32)

    def _compute_zero_padding(self, term: ObservationTerm) -> torch.Tensor:
        """Return zero padding."""
        return torch.zeros(term.obs_dim, device=self.device, dtype=torch.float32)

    def compute(self, sim_state) -> torch.Tensor:
        """
        Compute full observation vector.

        Args:
            sim_state: Current simulation state.

        Returns:
            Concatenated observation tensor of shape (total_obs_dim,).
        """
        obs_list = []
        for term in self.terms:
            obs = term.compute(sim_state)
            obs_list.append(obs)

        return torch.cat(obs_list, dim=0)

    def set_last_action(self, action: torch.Tensor):
        """Update last action for observation terms that need it."""
        self.last_action = action

    def reset(self):
        """Reset all history buffers."""
        for term in self.terms:
            term.reset()
        self.last_action = None
