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

"""Unified command provider interface for sim2mujoco evaluation.

Provides a single polymorphic interface for reading commands regardless of
whether the policy is driven by velocity/height commands or motion tracking
references.  The eval script and data logger consume this interface without
needing to know the underlying command source.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from agile.sim2mujoco.commands import CommandManager


@runtime_checkable
class CommandProvider(Protocol):
    """Uniform read interface for policy commands (velocity, motion tracking, etc.)."""

    @property
    def command_type(self) -> str:
        """Short identifier: ``"velocity"``, ``"velocity_height"``, ``"motion_tracking"``."""
        ...

    @property
    def command_dim(self) -> int:
        """Number of scalar command dimensions."""
        ...

    @property
    def command_names(self) -> list[str]:
        """Human-readable label for each dimension (length == :attr:`command_dim`)."""
        ...

    def get_commands(self) -> torch.Tensor:
        """Current command tensor, shape ``(command_dim,)``."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class VelocityCommandProvider:
    """Wraps :class:`CommandManager` for 3-D velocity or 4-D velocity+height commands."""

    def __init__(self, command_manager: CommandManager, dim: int):
        if dim not in (3, 4):
            raise ValueError(f"VelocityCommandProvider supports dim 3 or 4, got {dim}")
        self._manager = command_manager
        self._dim = dim

    @property
    def command_type(self) -> str:
        return "velocity" if self._dim == 3 else "velocity_height"

    @property
    def command_dim(self) -> int:
        return self._dim

    @property
    def command_names(self) -> list[str]:
        if self._dim == 3:
            return ["vx", "vy", "wz"]
        return ["vx", "vy", "wz", "height"]

    def get_commands(self) -> torch.Tensor:
        if self._dim == 3:
            return self._manager.get_navigation_command()
        return self._manager.get_command()

    @property
    def manager(self) -> CommandManager:
        """Access the underlying :class:`CommandManager` (for keyboard callbacks, scheduler, etc.)."""
        return self._manager


class MotionCommandProvider:
    """Wraps :class:`~agile.sim2mujoco.observations.MotionTracker` as a command provider.

    The command tensor is ``cat([target_joint_pos, target_joint_vel])`` from the
    reference motion file, in the policy's joint ordering.
    """

    def __init__(self, motion_tracker):
        self._tracker = motion_tracker
        n = self._tracker.data.joint_pos.shape[1]
        self._names = [f"ref_joint_pos_{i}" for i in range(n)] + [f"ref_joint_vel_{i}" for i in range(n)]

    @property
    def command_type(self) -> str:
        return "motion_tracking"

    @property
    def command_dim(self) -> int:
        return len(self._names)

    @property
    def command_names(self) -> list[str]:
        return self._names

    def get_commands(self) -> torch.Tensor:
        return self._tracker.get_command()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_NAV_COMMAND_NAMES = {"navigation_command", "locomotion_command"}
_VEL_HEIGHT_COMMAND_NAMES = {"velocity_and_height_command", "generated_commands"}


def create_command_provider(
    config: dict,
    device: torch.device,
    *,
    motion_tracker=None,
) -> CommandProvider | None:
    """Inspect *config* and return the appropriate :class:`CommandProvider`, or ``None``.

    Decision logic:
      1. If a ``motion_tracking`` section is present **and** a motion tracker was
         created → :class:`MotionCommandProvider`.
      2. If a velocity / velocity+height observation term is found →
         :class:`VelocityCommandProvider` (creates a :class:`CommandManager` internally).
      3. Otherwise → ``None`` (policy has no command input).

    Args:
        config: Full YAML configuration dictionary.
        device: Torch device for tensor creation.
        motion_tracker: An already-created
            :class:`~agile.sim2mujoco.observations.MotionTracker` instance, if
            the observation processor built one.

    Returns:
        A :class:`CommandProvider` instance, or ``None``.
    """
    if motion_tracker is not None:
        return MotionCommandProvider(motion_tracker)

    obs_terms = config.get("observations", {}).get("policy", [])
    for term in obs_terms:
        name = term.get("name", "")
        shape = term.get("shape", [1])
        dim = shape[0] if isinstance(shape, list) and len(shape) >= 1 else 0

        if name in _NAV_COMMAND_NAMES:
            mgr = CommandManager(
                device=device,
                defaults={"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0, "height": 0.72},
            )
            return VelocityCommandProvider(mgr, min(dim, 3))

        if name in _VEL_HEIGHT_COMMAND_NAMES:
            mgr = CommandManager(
                device=device,
                defaults={"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0, "height": 0.74},
            )
            return VelocityCommandProvider(mgr, min(dim, 4))

    return None
