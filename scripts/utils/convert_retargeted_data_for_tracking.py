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

"""Convert retargeted data for tracking task.

Remaps keys, resamples to target fps (default 50), and recomputes velocities.

Usage:
    python scripts/utils/convert_retargeted_data_for_tracking.py \
        --input_file ../g1/ACCAD/Female1General_c3d/A1-Stand_poses_120_jpos.npz \
        --output_file ./motions/A1-Stand.npz \
        --output_fps 50
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Isaac Lab npz to BeyondMimic format.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to Isaac Lab format npz.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for output BeyondMimic npz.")
    parser.add_argument("--output_fps", type=int, default=50, help="Target fps for output (default: 50).")
    return parser.parse_args()


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation: a*(1-t) + b*t. t can be broadcastable."""
    return a * (1.0 - t) + b * t


def slerp_quaternions(quats: np.ndarray, input_times: np.ndarray, output_times: np.ndarray) -> np.ndarray:
    """Slerp quaternions from input_times to output_times.

    Args:
        quats: (N_in, ..., 4) quaternions in wxyz order.
        input_times: (N_in,) monotonic timestamps.
        output_times: (N_out,) target timestamps.

    Returns:
        (N_out, ..., 4) interpolated quaternions in wxyz order.
    """
    original_shape = quats.shape  # (N_in, ..., 4)
    n_in = original_shape[0]

    # Flatten middle dims: (N_in, M, 4)
    if quats.ndim == 2:
        quats_flat = quats[:, np.newaxis, :]  # (N_in, 1, 4)
    else:
        quats_flat = quats.reshape(n_in, -1, 4)  # (N_in, M, 4)

    m = quats_flat.shape[1]
    n_out = len(output_times)

    # Output shape
    if quats.ndim == 2:
        out_shape = (n_out, 4)
    else:
        out_shape = (n_out,) + original_shape[1:]

    result = np.zeros((n_out, m, 4), dtype=np.float64)

    for j in range(m):
        # Convert wxyz -> xyzw for scipy
        q_xyzw = quats_flat[:, j, [1, 2, 3, 0]]
        rots = Rotation.from_quat(q_xyzw)
        slerp_fn = Slerp(input_times, rots)
        interp_rots = slerp_fn(output_times)
        q_out_xyzw = interp_rots.as_quat()  # (N_out, 4) in xyzw
        # Convert xyzw -> wxyz
        result[:, j, :] = q_out_xyzw[:, [3, 0, 1, 2]]

    return result.reshape(out_shape).astype(np.float32)


def compute_velocity_finite_diff(data: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity via central finite differences (numpy gradient)."""
    return np.gradient(data, dt, axis=0).astype(np.float32)


def compute_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    """Compute angular velocity from quaternion sequence via finite differences.

    Uses q_rel = q_next * q_prev^{-1}, then converts to axis-angle / (2*dt).
    Mirrors csv_to_npz.py _so3_derivative approach.

    Args:
        quats: (N, ..., 4) quaternions in wxyz order.
        dt: time step.

    Returns:
        (N, ..., 3) angular velocities.
    """
    original_shape = quats.shape  # (N, ..., 4)
    n = original_shape[0]

    if quats.ndim == 2:
        quats_flat = quats[:, np.newaxis, :]
    else:
        quats_flat = quats.reshape(n, -1, 4)

    m = quats_flat.shape[1]

    if quats.ndim == 2:
        out_shape = (n, 3)
    else:
        out_shape = original_shape[:-1] + (3,)

    ang_vel = np.zeros((n, m, 3), dtype=np.float64)

    for j in range(m):
        # Convert wxyz -> xyzw for scipy
        q_xyzw = quats_flat[:, j, [1, 2, 3, 0]]

        q_prev = Rotation.from_quat(q_xyzw[:-2])
        q_next = Rotation.from_quat(q_xyzw[2:])
        q_rel = q_next * q_prev.inv()

        omega = q_rel.as_rotvec() / (2.0 * dt)  # (N-2, 3)

        # Pad: repeat first and last
        omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
        ang_vel[:, j, :] = omega

    return ang_vel.reshape(out_shape).astype(np.float32)


def resample_motion(
    data: dict[str, np.ndarray],
    input_fps: float,
    output_fps: int,
) -> dict[str, np.ndarray]:
    """Resample motion data from input_fps to output_fps."""
    n_input = data["dof_positions"].shape[0]
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (n_input - 1) * input_dt

    input_times = np.arange(n_input) * input_dt
    output_times = np.arange(0, duration, output_dt)
    n_output = len(output_times)

    print(f"Resampling: {n_input} frames @ {input_fps} fps -> {n_output} frames @ {output_fps} fps")
    print(f"Duration: {duration:.3f} s")

    # Compute blend factors (matching csv_to_npz.py approach)
    phase = output_times / duration
    index_0 = np.floor(phase * (n_input - 1)).astype(int)
    index_1 = np.minimum(index_0 + 1, n_input - 1)
    blend = (phase * (n_input - 1) - index_0).astype(np.float32)

    # Interpolate positions (lerp)
    dof_pos = lerp(
        data["dof_positions"][index_0],
        data["dof_positions"][index_1],
        blend[:, np.newaxis],
    )

    body_pos = lerp(
        data["body_positions"][index_0],
        data["body_positions"][index_1],
        blend[:, np.newaxis, np.newaxis],
    )

    # Interpolate quaternions (slerp)
    body_quat = slerp_quaternions(data["body_rotations"], input_times, output_times)

    # Recompute velocities from interpolated data
    dof_vel = compute_velocity_finite_diff(dof_pos, output_dt)
    body_lin_vel = compute_velocity_finite_diff(body_pos, output_dt)
    body_ang_vel = compute_angular_velocity(body_quat, output_dt)

    return {
        "fps": np.array(output_fps, dtype=np.float32),
        "joint_pos": dof_pos.astype(np.float32),
        "joint_vel": dof_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }


def main():
    args = parse_args()

    # Load input
    input_path = Path(args.input_file)
    assert input_path.exists(), f"Input file not found: {input_path}"
    data = dict(np.load(str(input_path), allow_pickle=True))

    # Validate input keys
    required_keys = ["fps", "dof_positions", "dof_velocities", "body_positions", "body_rotations"]
    missing = [k for k in required_keys if k not in data]
    assert not missing, f"Missing keys in input: {missing}. Found: {list(data.keys())}"

    input_fps = float(data["fps"].item())
    print(f"Input: {input_path}")
    print(f"  fps: {input_fps}")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Resample and remap
    output = resample_motion(data, input_fps, args.output_fps)

    # Validate output
    n_frames = output["joint_pos"].shape[0]
    n_joints = output["joint_pos"].shape[1]
    n_bodies = output["body_pos_w"].shape[1]
    print(f"\nOutput: {n_frames} frames, {n_joints} joints, {n_bodies} bodies, {args.output_fps} fps")
    for k, v in output.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **output)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
