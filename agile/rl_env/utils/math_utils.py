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


@torch.jit.script
def interpolate_linear(low: torch.Tensor, high: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between two tensors.

    Args:
        low: The lower value tensor, shape is (N, ...).
        high: The higher value tensor, shape is (N, ...).
        weight: The weight tensor, shape is (N).

    Returns:
        The interpolated value tensor, with shape (N, ...).
    """
    # Reshape weight to have singleton dimensions for each additional axis in low/high.
    new_shape = [weight.size(0)] + [1] * (low.dim() - 1)
    weight = weight.reshape(new_shape)
    return low + (high - low) * weight


@torch.jit.script
def angular_velocity_from_quats(q1: torch.Tensor, q2: torch.Tensor, dt: float):
    """
    Calculate the angular velocity of a body given two consecutive quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (N, 4).
        q2: The second quaternion in (w, x, y, z). Shape is (N, 4).
        dt: The time step between the two quaternions.

    Returns:
        The angular velocity of the body in (x, y, z). Shape is (N, 3).
    """
    assert q1.shape == q2.shape
    assert q1.dim() == 2
    q1_inv = math_utils.quat_inv(q1)
    q1_T_q2 = math_utils.quat_mul(q1_inv, q2)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(q1_T_q2)
    return (1.0 / dt) * torch.stack([roll, pitch, yaw], dim=-1)


@torch.jit.script
def quat_from_axis(axis: torch.Tensor) -> torch.Tensor:
    """Convert an axis representation to a quaternion.

    The angle is assumed to be the magnitude of the axis.

    Args:
        axis: A tensor of shape (..., 3) representing the axis of rotation,
              where the magnitude of the axis corresponds to the angle.

    Returns:
        A tensor representing the quaternion corresponding to the given axis-angle,
        with shape (..., 4) in (w, x, y, z) format.
    """
    assert axis.shape[-1] == 3
    angle = torch.linalg.norm(axis, dim=-1)
    quat = math_utils.quat_from_angle_axis(angle=angle, axis=axis)
    return quat


@torch.jit.script
def angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    Compute the rotation angle from a quaternion.

    Args:
        quat: A tensor of shape (..., 4) representing the quaternion.

    Returns:
        A tensor of shape (...) representing the angle in radians.
    """
    w = quat[..., 0]
    angle = 2.0 * torch.acos(w)
    return angle


@torch.jit.script
def angle_along_axis_from_quat(quat: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle along an axis from a quaternion.

    Args:
        quat: A tensor of shape (..., 4) representing the quaternion in (w,x,y,z) format.
        axis: A tensor of shape (..., 3) representing the axis of rotation. Should be
            normalized.

    Returns:
        A tensor of shape (...) representing the signed angle in radians along the
        specified axis.
    """
    # Extract quaternion components
    w = quat[..., 0]
    v = quat[..., 1:]  # x,y,z components

    # Project quaternion vector onto axis
    v_proj = torch.sum(v * axis, dim=-1)

    # Use atan2 to get signed angle
    angle = 2.0 * torch.atan2(v_proj, w)

    return angle


@torch.jit.script
def quat_to_tangent_normal(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a tangent and normal vector representation.

    Args:
        quat: A tensor of shape (..., 4) representing the quaternion in (w,x,y,z) format.

    Returns:
        A tensor of shape (..., 6) representing the tangent and normal vectors.
    """
    ref_tangent = torch.zeros_like(quat[..., 0:3])
    ref_tangent[..., 0] = 1
    tangent = math_utils.quat_apply(quat, ref_tangent)

    ref_normal = torch.zeros_like(quat[..., 0:3])
    ref_normal[..., -1] = 1
    normal = math_utils.quat_apply(quat, ref_normal)

    tangent_normal = torch.cat([tangent, normal], dim=len(tangent.shape) - 1)
    return tangent_normal


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    return torch.abs(torch.ones_like(other) * mag) * torch.sign(other)


@torch.jit.script
def euler_xyz_from_quat(
    quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (...,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_w, q_x, q_y, q_z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = math_utils.wrap_to_pi(torch.atan2(sin_roll, cos_roll))

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = math_utils.wrap_to_pi(
        torch.where(
            torch.abs(sin_pitch) >= 1,
            copysign(torch.pi / 2.0, sin_pitch),
            torch.asin(sin_pitch),
        )
    )

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math_utils.wrap_to_pi(torch.atan2(sin_yaw, cos_yaw))

    return roll, pitch, yaw
