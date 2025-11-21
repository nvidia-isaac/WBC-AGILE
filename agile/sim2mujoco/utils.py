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

"""Utility functions for sim2mujoco evaluation."""

from pathlib import Path

import torch
import yaml


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by the inverse of a quaternion.

    Uses the EXACT same implementation as isaac-deploy (core/math/geometry.py:624).
    This ensures numerical consistency.

    Args:
        q: Quaternion [w, x, y, z], shape (4,)
        v: Vector to rotate, shape (3,)

    Returns:
        Rotated vector, shape (3,)
    """
    # Extract quaternion components.
    q_w = q[0]
    q_vec = q[1:]

    # Apply the inverse rotation formula (matching isaac-deploy exactly).
    # v' = v * (2*w^2 - 1) - 2*w*(q_vec x v) + 2*(q_vec · v)*q_vec
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=0) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v) * 2.0

    return a - b + c


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse (conjugate) of a quaternion.

    Args:
        q: Quaternion [w, x, y, z], shape (4,)

    Returns:
        Inverse quaternion [w, -x, -y, -z], shape (4,)
    """
    return torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device, dtype=q.dtype)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply a quaternion rotation to a vector.

    Uses the cross-product formula matching isaac-deploy (core/math/geometry.py:559).

    Args:
        q: Quaternion [w, x, y, z], shape (4,)
        v: Vector to rotate, shape (3,)

    Returns:
        Rotated vector, shape (3,)
    """
    # Extract quaternion vector part.
    q_w = q[0]
    q_vec = q[1:]

    # Cross-product formula: v' = v + 2*w*(q_vec × v) + 2*(q_vec × (q_vec × v))
    t = torch.cross(q_vec, v, dim=0) * 2.0
    return v + q_w * t + torch.cross(q_vec, t, dim=0)


def load_config(yaml_path: Path) -> dict:
    """
    Load YAML configuration file.

    Args:
        yaml_path: Path to YAML file.

    Returns:
        Dictionary containing configuration.
    """
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def default_device() -> torch.device:
    """
    Get default device (CUDA if available, else CPU).

    Returns:
        torch.device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
