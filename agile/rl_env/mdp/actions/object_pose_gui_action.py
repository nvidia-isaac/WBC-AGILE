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

"""Object Pose GUI Action for interactive control of rigid objects."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .object_pose_gui_action_cfg import ObjectPoseGUIActionCfg


class ObjectPoseGUIAction(ActionTerm):
    """Object pose GUI action.

    This action term allows controlling a rigid object's 6DoF pose interactively via a
    DearPyGui window. A separate thread is spawned for the GUI so that the physics
    simulation can continue to run in the main thread.

    The pose is specified as:
    - Position: x, y, z in meters (world frame)
    - Orientation: roll, pitch, yaw in radians (Euler XYZ convention)

    Usage:
        Add this action term in the environment's action configuration. The GUI
        values are authoritative - any RL actions sent to this term are ignored.
    """

    cfg: ObjectPoseGUIActionCfg

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def __init__(self, cfg: ObjectPoseGUIActionCfg, env: ManagerBasedEnv):
        """Initialize the object pose GUI action.

        Args:
            cfg: Configuration for the action term.
            env: The environment instance.
        """
        # Store config and environment (don't call super().__init__ as it expects an articulation)
        self.cfg = cfg
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs

        # Get the rigid object from the scene
        self._object: RigidObject = env.scene[cfg.asset_name]

        # Initialize desired pose from current object state
        current_pos = self._object.data.root_link_pos_w.torch[0].clone()
        current_quat = self._object.data.root_link_quat_w.torch[0].clone()

        # Convert quaternion to Euler angles for GUI
        roll, pitch, yaw = euler_xyz_from_quat(current_quat.unsqueeze(0))

        # Store desired pose (on CPU for GUI thread access)
        self._desired_pos = current_pos.cpu()
        self._desired_euler = torch.tensor([roll.item(), pitch.item(), yaw.item()])

        # Store initial/default pose for reset
        self._default_pos = self._desired_pos.clone()
        self._default_euler = self._desired_euler.clone()

        # Thread-safe lock for accessing pose from GUI
        self._lock = threading.Lock()

        # Launch GUI in a daemon thread
        self._gui_thread = threading.Thread(
            target=self._launch_gui, name=f"ObjectPoseGUI_{cfg.asset_name}", daemon=True
        )
        self._gui_thread.start()

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Dimension of the action term (not used for GUI action)."""
        return 0  # No RL action dimension - GUI is authoritative

    @property
    def device(self) -> str:
        """Device for tensors."""
        return str(self._device)

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return int(self._num_envs)

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term (unused for GUI)."""
        return torch.empty(0, device=self._device)

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (returns current desired pose)."""
        with self._lock:
            pos = self._desired_pos.clone()
            euler = self._desired_euler.clone()
        return torch.cat([pos, euler])

    # ---------------------------------------------------------------------
    # GUI Implementation
    # ---------------------------------------------------------------------

    def _launch_gui(self) -> None:
        """Create the DearPyGui window with pose control sliders."""
        import time

        import dearpygui.dearpygui as dpg

        # Wait a bit to let joint GUI initialize first if it exists
        time.sleep(0.5)

        # Check if DearPyGui context already exists (from joint GUI)
        try:
            # Try to check if context exists by testing if we can get viewport info
            context_exists = dpg.is_dearpygui_running()
        except Exception:
            context_exists = False

        # Only create context if it doesn't exist
        owns_context = False
        if not context_exists:
            try:
                dpg.create_context()
                dpg.create_viewport(
                    title="Debug Controller",
                    width=600,
                    height=800,
                )
                owns_context = True
            except Exception:
                # Context might have been created by another thread in the meantime
                pass

        # Get limits from config
        pos_limits = self.cfg.position_limits
        rot_limits = self.cfg.rotation_limits

        # Store slider tags for programmatic updates
        pos_slider_tags: dict[str, int] = {}
        euler_slider_tags: dict[str, int] = {}

        # Use unique window tag to avoid conflicts
        window_tag = f"object_pose_window_{self.cfg.asset_name}"

        with dpg.window(label=self.cfg.gui_window_title, tag=window_tag, width=480, height=350, pos=(10, 450)):
            dpg.add_text(f"Control pose of '{self.cfg.asset_name}'")
            dpg.add_separator()

            # --- Buttons ---
            def _reset_pose_cb() -> None:
                """Reset object to default pose."""
                with self._lock:
                    self._desired_pos[:] = self._default_pos.clone()
                    self._desired_euler[:] = self._default_euler.clone()
                    # Update GUI sliders
                    for i, axis in enumerate(["x", "y", "z"]):
                        dpg.set_value(pos_slider_tags[axis], float(self._desired_pos[i]))
                    for i, axis in enumerate(["roll", "pitch", "yaw"]):
                        dpg.set_value(euler_slider_tags[axis], float(self._desired_euler[i]))

            def _randomize_pose_cb() -> None:
                """Randomize object pose within limits."""
                with self._lock:
                    # Random position
                    for i, axis in enumerate(["x", "y", "z"]):
                        low, high = pos_limits[axis]
                        self._desired_pos[i] = low + (high - low) * torch.rand(1).item()
                        dpg.set_value(pos_slider_tags[axis], float(self._desired_pos[i]))
                    # Random orientation
                    for i, axis in enumerate(["roll", "pitch", "yaw"]):
                        low, high = rot_limits[axis]
                        self._desired_euler[i] = low + (high - low) * torch.rand(1).item()
                        dpg.set_value(euler_slider_tags[axis], float(self._desired_euler[i]))

            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset to Default", callback=_reset_pose_cb)
                dpg.add_button(label="Randomize Pose", callback=_randomize_pose_cb)

            dpg.add_separator()

            # --- Position Sliders ---
            dpg.add_text("POSITION (meters)")

            for i, axis in enumerate(["x", "y", "z"]):
                low, high = pos_limits[axis]
                current_val = float(self._desired_pos[i])

                def _pos_slider_cb(sender: int, app_data: float, user_data: int) -> None:  # noqa: ARG001
                    idx = user_data
                    with self._lock:
                        self._desired_pos[idx] = float(app_data)

                slider_tag = dpg.add_slider_float(
                    label=f"{axis.upper()}",
                    min_value=low,
                    max_value=high,
                    default_value=current_val,
                    callback=_pos_slider_cb,
                    user_data=i,
                    format="%.3f m",
                    width=350,
                )
                pos_slider_tags[axis] = slider_tag

            dpg.add_separator()

            # --- Orientation Sliders ---
            dpg.add_text("ORIENTATION (Euler XYZ, radians)")

            for i, axis in enumerate(["roll", "pitch", "yaw"]):
                low, high = rot_limits[axis]
                current_val = float(self._desired_euler[i])

                def _euler_slider_cb(sender: int, app_data: float, user_data: int) -> None:  # noqa: ARG001
                    idx = user_data
                    with self._lock:
                        self._desired_euler[idx] = float(app_data)

                slider_tag = dpg.add_slider_float(
                    label=f"{axis.capitalize()}",
                    min_value=low,
                    max_value=high,
                    default_value=current_val,
                    callback=_euler_slider_cb,
                    user_data=i,
                    format="%.3f rad",
                    width=350,
                )
                euler_slider_tags[axis] = slider_tag

            dpg.add_separator()

            # --- Current Pose Display (read-only) ---
            dpg.add_text("CURRENT POSE (read-only)")
            pos_text_tag = dpg.add_text("Pos: [0.000, 0.000, 0.000]")
            quat_text_tag = dpg.add_text("Quat: [1.000, 0.000, 0.000, 0.000]")

        # Only run our own event loop if we own the context
        if owns_context:
            # Setup and show
            dpg.setup_dearpygui()
            dpg.show_viewport()

            # Main GUI loop
            while dpg.is_dearpygui_running():
                # Update current pose display
                try:
                    current_pos = self._object.data.root_link_pos_w.torch[0].cpu()
                    current_quat = self._object.data.root_link_quat_w.torch[0].cpu()
                    dpg.set_value(
                        pos_text_tag,
                        f"Pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]",
                    )
                    dpg.set_value(
                        quat_text_tag,
                        f"Quat: [{current_quat[0]:.3f}, {current_quat[1]:.3f}, "
                        f"{current_quat[2]:.3f}, {current_quat[3]:.3f}]",
                    )
                except Exception:
                    pass  # Object may not be ready yet

                dpg.render_dearpygui_frame()

            dpg.destroy_context()
        else:
            # Another GUI owns the context - just keep updating our display in a loop
            while True:
                try:
                    if not dpg.is_dearpygui_running():
                        break
                    current_pos = self._object.data.root_link_pos_w.torch[0].cpu()
                    current_quat = self._object.data.root_link_quat_w.torch[0].cpu()
                    dpg.set_value(
                        pos_text_tag,
                        f"Pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]",
                    )
                    dpg.set_value(
                        quat_text_tag,
                        f"Quat: [{current_quat[0]:.3f}, {current_quat[1]:.3f}, "
                        f"{current_quat[2]:.3f}, {current_quat[3]:.3f}]",
                    )
                    time.sleep(0.05)  # Small delay to avoid busy loop
                except Exception:
                    time.sleep(0.1)  # Wait if not ready

    # ---------------------------------------------------------------------
    # ActionTerm Interface
    # ---------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        """Ignore incoming RL actions; GUI values are authoritative."""
        pass

    def apply_actions(self) -> None:
        """Apply the GUI-specified pose to the rigid object."""
        with self._lock:
            pos = self._desired_pos.clone()
            euler = self._desired_euler.clone()

        # Convert to device
        pos = pos.to(self._device)
        euler = euler.to(self._device)

        # Convert Euler to quaternion (wxyz format)
        quat = quat_from_euler_xyz(
            euler[0:1],  # roll
            euler[1:2],  # pitch
            euler[2:3],  # yaw
        ).squeeze(0)

        # Build pose tensor [x, y, z, w, qx, qy, qz]
        pose = torch.cat([pos, quat])

        # Expand for all environments
        pose_batch = pose.unsqueeze(0).expand(self._num_envs, -1)

        # Write pose to simulation
        self._object.write_root_link_pose_to_sim(pose_batch)

        # Zero out velocities to prevent drift
        zero_vel = torch.zeros(self._num_envs, 6, device=self._device)
        self._object.write_root_com_velocity_to_sim(zero_vel)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term (called on environment reset).

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        # Sync GUI with current object pose on reset
        if env_ids is None or 0 in env_ids:
            with self._lock:
                current_pos = self._object.data.root_link_pos_w.torch[0].cpu()
                current_quat = self._object.data.root_link_quat_w.torch[0].cpu()
                roll, pitch, yaw = euler_xyz_from_quat(current_quat.unsqueeze(0))

                self._desired_pos[:] = current_pos
                self._desired_euler[0] = roll.item()
                self._desired_euler[1] = pitch.item()
                self._desired_euler[2] = yaw.item()

                # Update defaults
                self._default_pos[:] = self._desired_pos.clone()
                self._default_euler[:] = self._desired_euler.clone()
