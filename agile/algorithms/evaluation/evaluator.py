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


from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from agile.algorithms.evaluation.episode_buffer import EpisodeBuffer, Frame
from agile.algorithms.evaluation.motion_metrics_analyzer import MotionMetricsAnalyzer
from agile.algorithms.evaluation.trajectory_logger import TrajectoryLogger


class PolicyEvaluator:
    """Evaluates policy performance with focus on motion smoothness and success rate.

    Handles:
    - Data collection from environment steps
    - Tracking termination conditions
    - Computing smoothness metrics
    - Aggregating results across envs
    """

    def __init__(
        self,
        env,
        task_name: str,
        metrics_path: str | None = None,
        total_envs_target: int = 1,
        verbose: bool = False,
        save_trajectories: bool = False,
        trajectory_fields: list[str] | None = None,
        joint_group_config: dict | None = None,
    ):
        """Initialize a policy evaluator.

        Args:
            env: Environment to evaluate policy in
            task_name: Name of the task
            metrics_path: Path to save metrics to (optional). If not provided and save_trajectories
                         is True, will auto-create: logs/evaluation/{task_name}_{datetime}
            total_envs_target: Number of envs to evaluate (default: 1)
            verbose: Whether to print detailed diagnostic information (default: False)
            save_trajectories: Whether to save episode trajectory data to parquet files (default: False)
            trajectory_fields: List of field names to save in trajectories. None = save all.
                              Example: ["joint_pos", "joint_vel", "root_pos"]
            joint_group_config: Optional dict defining joint groups from eval_config.yaml.
                              Format: {"upper_body": ["joint1", ".*_shoulder_.*", ...], ...}
                              Each value is a list of joint names/patterns supporting wildcards.
                              If None, all joints go to "default" group.
        """
        self._env = env.unwrapped if hasattr(env, "unwrapped") else env
        self._num_envs = self._env.num_envs
        # Ensure device is a torch.device object (not a string)
        self._device = torch.device(env.device) if isinstance(env.device, str) else env.device
        self._verbose = verbose

        # Get episode length information
        self._max_episode_len = getattr(env, "max_episode_length", 100)

        # Build joint groups from config
        self._joint_groups = self._build_joint_groups(joint_group_config)

        # Initialize metrics calculator
        self._metrics = MotionMetricsAnalyzer(
            max_episode_length=self._max_episode_len, joint_groups=self._joint_groups, verbose=verbose
        )
        self._metrics_path = metrics_path

        # Initialize trajectory logger if requested
        self._trajectory_logger = None
        if save_trajectories:
            # Determine output directory
            if metrics_path is None:
                # Auto-generate path: logs/evaluation/{task_name}_{datetime}
                import time
                from pathlib import Path

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self._metrics_path = Path("logs") / "evaluation" / f"{task_name}_{timestamp}"

            print(f"[INFO] Initializing TrajectoryLogger with output_dir: {self._metrics_path}")
            # Use step_dt (control timestep) instead of physics_dt to properly account for decimation
            control_dt = self._env.step_dt if hasattr(self._env, "step_dt") else self._env.physics_dt
            self._trajectory_logger = TrajectoryLogger(
                output_dir=self._metrics_path,
                physics_dt=control_dt,
                env=self._env,  # Pass environment for metadata extraction
                fields_to_save=trajectory_fields,
                joint_groups=self._joint_groups,  # Pass joint groups for metadata
                verbose=verbose,
            )

        # Episode tracking
        self._episode_buffer = EpisodeBuffer(self._num_envs, self._max_episode_len, self._device)
        self._num_envs_evaluated = 0

        # Set target number of environments to evaluate
        self._total_envs_target = total_envs_target

        # Store previous frame to handle terminal states correctly
        # When done=True, observations contain reset state, so we use previous frame
        self._previous_frame = None

        # Progress tracking
        self._pbar = tqdm(
            total=self._total_envs_target,
            desc="Evaluating policy",
            position=0,
            leave=True,
        )

        # Info about episode numbering mode
        if self._total_envs_target == self._num_envs:
            print(
                "[INFO] Episode numbering mode: env_id-based (episode_id = env_id)\n"
                "       Episode N will contain data from environment N"
            )
        else:
            print(
                f"[INFO] Episode numbering mode: sequential (multiple episodes per env)\n"
                f"       {self._num_envs} envs × {self._total_envs_target // self._num_envs} episodes = {self._total_envs_target} total"
            )

    def _build_joint_groups(self, joint_group_config: dict | None) -> dict[str, list[int]]:
        """Build joint groups from configuration.

        Args:
            joint_group_config: Dict mapping group names to list of joint name patterns.
                              Each pattern supports wildcards (e.g., ".*_hip_.*", "waist_yaw").
                              If None, all joints go to "default" group.

        Returns:
            Dictionary mapping group names to lists of joint indices.
            Always includes groups for assigned joints, plus "default" for any unassigned.

        Example joint_group_config:
            {"upper_body": ["waist_.*", ".*_shoulder_.*", ".*_elbow.*"],
             "lower_body": [".*_hip_.*", ".*_knee.*", ".*_ankle.*"]}
        """
        # Access robot from scene (InteractiveScene is dict-like but doesn't have .get())
        try:
            robot = self._env.scene["robot"]
        except (KeyError, AttributeError, TypeError):
            if self._verbose:
                print("[WARNING] No robot in scene, cannot create joint groups")
            return {}

        all_joint_names = robot.joint_names
        num_joints = len(all_joint_names)

        if self._verbose:
            print(f"[INFO] Building joint groups from {num_joints} joints")

        # Track which joints are assigned to groups
        assigned_joints = set()
        joint_groups = {}

        # If no config provided, use default group with all joints
        if joint_group_config is None:
            if self._verbose:
                print("[INFO] No joint groups specified, using 'default' group with all joints")
            return {"default": list(range(num_joints))}

        # Process each configured group
        for group_name, patterns in joint_group_config.items():
            if not isinstance(patterns, list):
                if self._verbose:
                    print(f"[WARNING] Invalid group spec for '{group_name}' (expected list), skipping")
                continue

            group_indices = []

            # Use robot.find_joints() for pattern matching (supports wildcards)
            try:
                matched_indices = robot.find_joints(patterns)[0]
                for idx in matched_indices:
                    if idx not in assigned_joints:
                        group_indices.append(idx)
                        assigned_joints.add(idx)
            except Exception as e:
                if self._verbose:
                    print(f"[WARNING] Pattern matching failed for group '{group_name}': {e}")

            # Store group if it has any joints
            if group_indices:
                joint_groups[group_name] = sorted(group_indices)
                if self._verbose:
                    matched_names = [all_joint_names[i] for i in group_indices]
                    print(f"[INFO] Group '{group_name}': {len(group_indices)} joints - {matched_names}")
            elif self._verbose:
                print(f"[WARNING] Joint group '{group_name}' is empty (no matching joints)")

        # Add remaining unassigned joints to "default" group
        unassigned = [i for i in range(num_joints) if i not in assigned_joints]
        if unassigned:
            joint_groups["default"] = unassigned
            if self._verbose:
                print(f"[INFO] Group 'default': {len(unassigned)} unassigned joints")

        return joint_groups

    def _create_hybrid_frame(self, current_frame: Frame, previous_frame: Frame, terminated_ids: torch.Tensor) -> Frame:
        """Create a hybrid frame using previous data for terminated envs, current for others.

        When environments terminate, the observations contain reset state. To avoid
        contaminating metrics with reset-induced acceleration spikes, we use the
        previous frame's observations for terminated environments.

        Args:
            current_frame: Frame with current observations (contains reset data for terminated envs)
            previous_frame: Frame with previous observations (contains valid terminal data)
            terminated_ids: Indices of environments that terminated

        Returns:
            Hybrid frame with correct data for all environments
        """
        # Create a copy of current frame data
        hybrid_data = {}

        for key, current_tensor in current_frame.__dict__.items():
            if current_tensor is None:
                hybrid_data[key] = None
                continue

            # Skip tensors that are not torch tensors
            if not isinstance(current_tensor, torch.Tensor):
                hybrid_data[key] = current_tensor
                continue

            # Skip empty tensors or tensors with size 0
            if current_tensor.numel() == 0:
                hybrid_data[key] = current_tensor
                continue

            # Get corresponding tensor from previous frame
            previous_tensor = getattr(previous_frame, key, None)
            if previous_tensor is None or not isinstance(previous_tensor, torch.Tensor):
                # If previous frame doesn't have this key, use current
                hybrid_data[key] = current_tensor
                continue

            # Skip if previous tensor is empty
            if previous_tensor.numel() == 0:
                hybrid_data[key] = current_tensor
                continue

            # Clone current tensor to avoid modifying the original
            hybrid_tensor = current_tensor.clone()

            # Ensure terminated_ids is on the same device as the tensor
            terminated_ids_device = terminated_ids.to(hybrid_tensor.device)

            # Replace terminated environment data with previous frame data
            # Handle different tensor shapes (e.g., [num_envs, ...] or [num_envs])
            try:
                if current_tensor.dim() == 1:
                    hybrid_tensor[terminated_ids_device] = previous_tensor[terminated_ids_device]
                else:
                    hybrid_tensor[terminated_ids_device, ...] = previous_tensor[terminated_ids_device, ...]
            except (IndexError, RuntimeError) as e:
                # If indexing fails, just use current tensor (likely a shape mismatch)
                print(f"Warning: Could not replace terminated data for field '{key}': {e}")
                print(f"  Current shape: {current_tensor.shape}, Previous shape: {previous_tensor.shape}")
                hybrid_data[key] = current_tensor
                continue

            hybrid_data[key] = hybrid_tensor

        return Frame(**hybrid_data)

    def collect(self, dones: torch.Tensor, info: dict[str, Any]) -> bool:
        """Collect data from an environment step.

        Args:
            dones: Boolean tensor indicating which environments are done
            info: Dictionary of additional information from environment

        Returns:
            Whether evaluation is complete and no more collection is needed
        """
        # Check if evaluation is already complete - make function a no-op if so
        if self._num_envs_evaluated >= self._total_envs_target:
            return True

        # Extract observation data and create current frame
        frame_data = self._extract_frame_data(info)
        current_frame = Frame.from_dict(frame_data)

        # Get terminated environment IDs
        terminated_ids = dones.nonzero(as_tuple=False).squeeze(-1) if dones.any() else None

        # If we have more terminations than needed, only take what we need to reach the target
        if terminated_ids is not None and len(terminated_ids) > 0:
            remaining_needed = self._total_envs_target - self._num_envs_evaluated
            if remaining_needed < len(terminated_ids):
                terminated_ids = terminated_ids[:remaining_needed]

        # Determine which frame to add to the buffer
        # For non-terminated environments: use current frame (contains valid observations)
        # For terminated environments: use previous frame (current contains reset observations)
        if self._previous_frame is not None and terminated_ids is not None and len(terminated_ids) > 0:
            # Create a hybrid frame: use previous frame data for terminated envs, current for others
            frame_to_add = self._create_hybrid_frame(current_frame, self._previous_frame, terminated_ids)
        else:
            # First step or no terminations: use current frame
            frame_to_add = current_frame

        # Add frame to episode buffer and get terminated episode data
        terminated_data = self._episode_buffer.add_frame(frame_to_add, terminated_ids)

        # Process terminated data if any
        if terminated_data is not None:
            # Update metrics with terminated data
            self._metrics.update(terminated_data)

            # Log trajectories if enabled
            if self._trajectory_logger:
                # Get frame counts to determine success
                frame_counts = terminated_data.get("frame_counts")
                is_success = (
                    frame_counts == self._max_episode_len
                    if frame_counts is not None
                    else torch.zeros_like(terminated_ids, dtype=torch.bool)
                )

                # Generate episode numbers for this batch
                # For deterministic evaluation with one episode per env, use env_id as episode_id
                # This ensures Episode N always contains env_id=N's data, making visualization intuitive
                if self._total_envs_target == self._num_envs:
                    # Single episode per env: use env_id directly as episode_id
                    episode_numbers = terminated_ids.cpu().tolist()
                else:
                    # Multiple episodes per env: use sequential numbering as before
                    episode_numbers = list(
                        range(self._num_envs_evaluated, self._num_envs_evaluated + len(terminated_ids))
                    )

                if self._verbose:
                    mode = "env_id-based" if self._total_envs_target == self._num_envs else "sequential"
                    print(f"[DEBUG] Logging {len(terminated_ids)} episodes ({mode} numbering): {episode_numbers}")
                    print(f"[DEBUG] Env IDs: {terminated_ids.cpu().tolist()}")
                    print(f"[DEBUG] Frame counts: {frame_counts.tolist() if frame_counts is not None else 'None'}")
                    print(f"[DEBUG] Success flags: {is_success.tolist()}")

                # Log episodes
                self._trajectory_logger.log_episodes(
                    episode_data=terminated_data,
                    env_ids=terminated_ids,
                    episode_numbers=episode_numbers,
                    is_success=is_success,
                )

            # Count newly terminated environments
            num_terminations = len(terminated_ids)
            self._num_envs_evaluated += num_terminations

            # Update progress bar
            if self._pbar:
                self._pbar.update(num_terminations)

        # Store current frame as previous for next iteration
        self._previous_frame = current_frame

        # Update progress display
        self._update_status_bar()

        # Check if we've reached the target number of environments
        if self._num_envs_evaluated >= self._total_envs_target:
            if self._pbar:
                self._pbar.set_description(f"Evaluation complete: {self._num_envs_evaluated} environments evaluated")
            return True

        # Return whether evaluation is complete
        return False

    def _extract_frame_data(self, info: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract frame data from environment info.

        Args:
            info: Dictionary of environment information

        Returns:
            Dictionary of trajectory data for this frame

        Raises:
            ValueError: If "eval" observation group doesn't exist or is missing required terms
        """
        # Extract observation data
        obs_dict = info.get("observations", {})
        if not obs_dict:
            raise ValueError("No observations found in environment info")

        # Access observation manager to get observation structure
        obs_manager = getattr(self._env, "observation_manager", None)
        if not obs_manager:
            raise ValueError("Environment does not have an observation_manager")

        # Check if "eval" observation group exists
        if "eval" not in obs_dict or "eval" not in obs_manager.active_terms:
            available_groups = list(obs_dict.keys())
            raise ValueError(
                f"PolicyEvaluator requires an 'eval' observation group in the environment, "
                f"but it was not found. Available observation groups: {available_groups}. "
                f"Please add an 'eval' observation group to your environment configuration "
                f"that includes: joint_pos, joint_vel, joint_acc, root_lin_vel, root_rot, root_pos"
            )

        # Required fields for evaluation (must be present)
        required_fields = [
            "joint_pos",
            "joint_vel",
            "joint_acc",
            "root_lin_vel",
            "root_ang_vel",
            "root_rot",
            "root_pos",
            "commands",
            "actions",
        ]

        # Optional fields (nice to have for analysis but not required)
        optional_fields = []

        # Extract data for each required field
        frame_data = {}
        missing_fields = []
        for field in required_fields:
            term_data = self._get_term_obs_data("eval", field, obs_dict, obs_manager)
            if term_data is not None:
                frame_data[field] = term_data
            else:
                missing_fields.append(field)

        # Check if we got all required fields
        if missing_fields:
            available_terms = obs_manager.active_terms.get("eval", [])
            raise ValueError(
                f"PolicyEvaluator requires the following observation terms in the 'eval' group: "
                f"{required_fields}, but the following are missing: {missing_fields}. "
                f"Available terms in 'eval' group: {available_terms}"
            )

        # Extract optional fields (don't error if missing)
        for field in optional_fields:
            term_data = self._get_term_obs_data("eval", field, obs_dict, obs_manager)
            if term_data is not None:
                frame_data[field] = term_data
            elif self._verbose:
                print(f"Optional field '{field}' not found in 'eval' observations")

        # Transform linear velocity from world frame to robot yaw-aligned frame
        # This is critical for comparing with commands, which are in the robot frame
        # Note: Angular velocity (yaw rate) is the same in both world and yaw-aligned frames, so no transformation needed
        if "root_lin_vel" in frame_data and "root_rot" in frame_data:
            # Extract yaw-only rotation (zero out pitch and roll)
            root_yaw_quat = yaw_quat(frame_data["root_rot"])

            # Transform linear velocity to robot frame: [v_forward, v_left, v_up]
            frame_data["root_lin_vel_robot"] = quat_apply_inverse(root_yaw_quat, frame_data["root_lin_vel"])

        return frame_data

    def _get_term_obs_data(
        self,
        group_name: str,
        term_name: str,
        obs_dict: dict[str, torch.Tensor],
        obs_manager,
    ) -> torch.Tensor | None:
        """Extract specific term data from observation dictionary.

        Args:
            group_name: Name of observation group
            term_name: Name of specific term to extract
            obs_dict: Dictionary of observations
            obs_manager: Observation manager from environment

        Returns:
            Tensor data for the requested term
        """
        # Ensure group exists in observations
        if group_name not in obs_dict or group_name not in obs_manager.active_terms:
            return None

        # Get term information
        group_term_names = obs_manager.active_terms[group_name]
        group_term_dims = obs_manager.group_obs_term_dim[group_name]
        group_obs_data = obs_dict[group_name]

        # Find term index
        if term_name not in group_term_names:
            return None

        term_idx = group_term_names.index(term_name)

        # Calculate start and end indices
        def get_dim(dim_value):
            return dim_value[0] if isinstance(dim_value, tuple) else dim_value

        term_dim_start = sum(get_dim(dim) for dim in group_term_dims[:term_idx])
        term_dim_end = term_dim_start + get_dim(group_term_dims[term_idx])

        # Extract data
        return group_obs_data[:, term_dim_start:term_dim_end]

    def _update_status_bar(self) -> str:
        """Update the progress bar with current status.

        Returns:
            Status string for display
        """
        if not self._pbar:
            return ""

        # Calculate current success rate
        if self._num_envs_evaluated > 0:
            # Get success rate from metrics which tracks successful environments
            current_success_rate = self._metrics.num_envs_successful / self._num_envs_evaluated
        else:
            current_success_rate = 0.0

        status = (
            f"Evaluated: {self._num_envs_evaluated}/{self._total_envs_target} | "
            f"Success rate: {current_success_rate:.3f}"
        )

        self._pbar.set_description(status)
        return status

    def conclude(self) -> dict[str, Any]:
        """Conclude evaluation and return results.

        Returns:
            Dictionary of evaluation results
        """
        # Close progress bar
        if self._pbar:
            self._pbar.close()
            self._pbar = None

        # Calculate final metrics
        self._metrics.conclude()

        # Print and save metrics
        self._metrics.print()
        if self._metrics_path:
            self._metrics.save(self._metrics_path, "metrics.json")

        # Return metrics
        return self._metrics.get_metrics()
