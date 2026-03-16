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

"""Shared motion-data loader for .npz motion clips.

This module is Isaac-Lab-free so it can be imported from both the training
environment (``agile.rl_env``) and the standalone MuJoCo evaluator
(``agile.sim2mujoco``).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch


class MotionData:
    """Loads an .npz motion clip and optionally remaps joint/body ordering.

    The .npz file is expected to contain at least:
        ``fps``, ``joint_pos``, ``joint_vel``,
        ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w``.

    Args:
        motion_file: Path to the ``.npz`` file.
        body_indices: Indices into the body dimension to keep.  When ``None``
            all bodies are retained.
        joint_remap_idx: If provided, ``joint_pos`` and ``joint_vel`` are
            reindexed along the joint axis using this index tensor so that the
            resulting ordering matches the consumer's expected joint order.
        device: Torch device string.
    """

    def __init__(
        self,
        motion_file: str,
        body_indices: Sequence[int] | None = None,
        joint_remap_idx: torch.Tensor | None = None,
        device: str = "cpu",
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.fps: float = float(data["fps"])

        joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)

        if joint_remap_idx is not None:
            joint_pos = joint_pos[:, joint_remap_idx]
            joint_vel = joint_vel[:, joint_remap_idx]

        self.joint_pos: torch.Tensor = joint_pos
        self.joint_vel: torch.Tensor = joint_vel

        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

        if body_indices is not None:
            self._body_indices: Sequence[int] | slice = body_indices
        else:
            self._body_indices = slice(None)

        self.time_step_total: int = self.joint_pos.shape[0]

    # ------------------------------------------------------------------
    # Body-data properties (filtered by body_indices)
    # ------------------------------------------------------------------

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indices]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indices]

    # ------------------------------------------------------------------
    # Static helpers for building remap / index arguments
    # ------------------------------------------------------------------

    @staticmethod
    def build_joint_remap_idx(
        target_joint_names: list[str],
        source_joint_names: list[str],
        device: str = "cpu",
    ) -> torch.Tensor:
        """Build an index tensor that remaps joints from *source* to *target* order.

        For each name in ``target_joint_names``, the returned tensor contains the
        index of that name in ``source_joint_names``.

        Raises:
            ValueError: If a target joint name is not found in the source list.
        """
        remap: list[int] = []
        for name in target_joint_names:
            if name not in source_joint_names:
                raise ValueError(
                    f"Target joint '{name}' not found in source joint names. Available: {source_joint_names}"
                )
            remap.append(source_joint_names.index(name))
        return torch.tensor(remap, dtype=torch.long, device=device)

    @staticmethod
    def build_body_indices(
        tracked_body_names: list[str],
        source_body_names: list[str],
    ) -> list[int]:
        """Build a list of body indices to select *tracked* bodies from *source* ordering.

        Raises:
            ValueError: If a tracked body name is not found in the source list.
        """
        indices: list[int] = []
        for name in tracked_body_names:
            if name not in source_body_names:
                raise ValueError(
                    f"Tracked body '{name}' not found in source body names. Available: {source_body_names}"
                )
            indices.append(source_body_names.index(name))
        return indices
