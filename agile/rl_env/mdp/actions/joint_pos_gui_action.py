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

import math
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction  # type: ignore

from agile.rl_env.mdp.symmetry.symmetry_g1 import lr_mirror_G1
from agile.rl_env.mdp.symmetry.symmetry_t1 import lr_mirror_T1

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointPositionGUIAction(JointPositionAction):
    """Joint position GUI action.

    This action term allows controlling the robot's joint positions interactively via a
    DearPyGui window.  A separate thread is spawned for the GUI so that the physics
    simulation can continue to run in the main thread.

    Usage:
        Add this action term in the environment's action configuration instead of a
        regular joint-position action.  All environment instances will receive the
        same joint targets defined in the GUI.
    """

    cfg: actions_cfg.JointPositionGUIActionCfg  # type: ignore[valid-type]

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def __init__(self, cfg: actions_cfg.JointPositionGUIActionCfg, env: ManagerBasedEnv):
        # Initialize parent class (resolves joints, etc.)
        super().__init__(cfg, env)  # type: ignore[arg-type]

        # Helper to safely slice/select joints on tensors living on either CPU or GPU
        # (defined below __init__).
        self._desired_pos = self._select_joints(self._asset.data.joint_pos.torch).clone()

        # Map action columns to their owning actuator columns once. Articulation joint order
        # is not necessarily grouped by actuator (for example, G1 waist joints are
        # interleaved with its legs), so actuator tensors must never be concatenated and
        # treated as if they were in articulation order.
        self._actuator_bindings = self._resolve_actuator_bindings()
        self._desired_stiffness = self._read_selected_actuator_property("stiffness")
        self._desired_damping = self._read_selected_actuator_property("damping")

        # Store default gains for reset
        self._default_stiffness = self._desired_stiffness.clone()
        self._default_damping = self._desired_damping.clone()
        self._mirror_actions = cfg.mirror_actions
        self._robot_type = cfg.robot_type

        if not math.isfinite(cfg.pose_transition_duration_s) or cfg.pose_transition_duration_s <= 0.0:
            raise ValueError("pose_transition_duration_s must be finite and greater than zero")
        self._pose_transition_duration_s = float(cfg.pose_transition_duration_s)
        self._position_transition_start = self._desired_pos.clone()
        self._position_transition_target = self._desired_pos.clone()
        self._position_transition_elapsed_s = 0.0
        self._position_transition_duration_s = self._pose_transition_duration_s
        self._position_transition_active = False

        if self._robot_type == "g1":
            self._symmetry_augmentation_func = lr_mirror_G1
        elif self._robot_type == "t1":
            self._symmetry_augmentation_func = lr_mirror_T1
        else:
            raise ValueError(f"Invalid robot type: {self._robot_type}")

        # Thread-safe lock for accessing ``_desired_pos`` from GUI
        self._lock = threading.Lock()
        # Launch GUI in a daemon thread so that it quits automatically with Python
        self._gui_thread = threading.Thread(target=self._launch_gui, name="JointGUI", daemon=True)
        self._gui_thread.start()

    # ---------------------------------------------------------------------
    # GUI helpers
    # ---------------------------------------------------------------------

    def _launch_gui(self) -> None:  # noqa: D401
        """Create the DearPyGui window and sliders for each joint."""
        import dearpygui.dearpygui as dpg

        # Create context and viewport (sized to fit both joint and object pose windows)
        dpg.create_context()
        dpg.create_viewport(title="Debug Controller", width=600, height=1200)

        with dpg.window(label="Joint Position Controller", tag="joint_window", width=580, height=700, pos=(10, 10)):
            dpg.add_text("Adjust joint positions (radians)")
            dpg.add_separator()

            # We need a list to store slider tags so we can programmatically update them
            pos_slider_tags: list[int] = []
            stiffness_slider_tags: list[int] = []
            damping_slider_tags: list[int] = []
            effort_bar_tags: list[int] = []

            # Create a theme to make the effort plot thicker
            with dpg.theme() as effort_bar_theme:
                with dpg.theme_component(dpg.mvStemSeries):
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 5, category=dpg.mvThemeCat_Plots)

            effort_limits = self._read_selected_actuator_property("effort_limit")[0].cpu()

            def _reset_joints_cb() -> None:
                """Smoothly reset all joints to their default positions and gains."""
                self._start_default_position_transition()
                with self._lock:
                    desired_stiffness = self._desired_stiffness[0].cpu()
                    desired_damping = self._desired_damping[0].cpu()

                # Position sliders follow the transition in the GUI loop. Gain sliders
                # can be updated immediately because gains are not interpolated.
                for i in range(len(self._joint_names)):
                    dpg.set_value(stiffness_slider_tags[i], float(desired_stiffness[i]))
                    dpg.set_value(damping_slider_tags[i], float(desired_damping[i]))

            def _randomize_joints_cb() -> None:
                """Smoothly move all joints to a random pose within their limits."""
                limits = self._select_joints(self._asset.data.soft_joint_pos_limits.torch[0].T).T.cpu()
                low = limits[:, 0]
                high = limits[:, 1]
                random_pos = low + (high - low) * torch.rand_like(low)
                self._start_position_transition(random_pos)

            def _transition_duration_cb(sender, app_data, user_data) -> None:  # type: ignore  # noqa: ARG001
                duration_s = float(app_data)
                if not math.isfinite(duration_s) or duration_s <= 0.0:
                    with self._lock:
                        duration_s = self._pose_transition_duration_s
                    dpg.set_value(sender, duration_s)
                    return
                with self._lock:
                    self._pose_transition_duration_s = duration_s

            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset to Default", callback=_reset_joints_cb)
                dpg.add_button(label="Randomize", callback=_randomize_joints_cb)
            dpg.add_input_float(
                label="Pose transition duration (s)",
                default_value=self._pose_transition_duration_s,
                min_value=0.1,
                min_clamped=True,
                step=0.25,
                step_fast=1.0,
                format="%.2f",
                width=100,
                callback=_transition_duration_cb,
            )
            dpg.add_separator()

            # Iterate over joints
            for local_id, joint_name in enumerate(self._joint_names):
                with dpg.group(horizontal=True):
                    # Group for sliders on the left
                    with dpg.group():
                        # Resolve the global joint index when self._joint_ids is a slice
                        joint_idx = local_id if isinstance(self._joint_ids, slice) else self._joint_ids[local_id]

                        # Fetch soft limits (first environment)
                        limits = self._asset.data.soft_joint_pos_limits.torch[0, joint_idx].cpu()
                        low, high = float(limits[0]), float(limits[1])
                        current_val = float(self._desired_pos[0, local_id].cpu())
                        current_stiffness = float(self._desired_stiffness[0, local_id].cpu())
                        current_damping = float(self._desired_damping[0, local_id].cpu())

                        # -- Position slider
                        def _slider_cb(sender, app_data, user_data) -> None:  # type: ignore  # noqa: ARG001
                            self._set_desired_joint_position(user_data, float(app_data))

                        pos_slider_tag = dpg.add_slider_float(
                            label=f"[{local_id}] {joint_name}",
                            min_value=low,
                            max_value=high,
                            default_value=current_val,
                            callback=_slider_cb,
                            user_data=local_id,
                            format="%.3f",
                            width=300,
                        )
                        pos_slider_tags.append(pos_slider_tag)

                        # -- Stiffness slider
                        def _stiffness_slider_cb(sender, app_data, user_data) -> None:  # type: ignore  # noqa: ARG001
                            idx: int = user_data
                            with self._lock:
                                self._desired_stiffness[:, idx] = float(app_data)

                        stiffness_slider_tag = dpg.add_slider_float(
                            label="P-Gain",
                            min_value=0.0,
                            max_value=max(self.cfg.max_stiffness, current_stiffness),
                            default_value=current_stiffness,
                            callback=_stiffness_slider_cb,
                            user_data=local_id,
                            format="%.1f",
                            indent=20,
                            width=280,
                        )
                        stiffness_slider_tags.append(stiffness_slider_tag)

                        # -- Damping slider
                        def _damping_slider_cb(sender, app_data, user_data) -> None:  # type: ignore  # noqa: ARG001
                            idx: int = user_data
                            with self._lock:
                                self._desired_damping[:, idx] = float(app_data)

                        damping_slider_tag = dpg.add_slider_float(
                            label="D-Gain",
                            min_value=0.0,
                            max_value=max(self.cfg.max_damping, current_damping),
                            default_value=current_damping,
                            callback=_damping_slider_cb,
                            user_data=local_id,
                            format="%.1f",
                            indent=20,
                            width=280,
                        )
                        damping_slider_tags.append(damping_slider_tag)

                    # Group for the effort plot on the right
                    with dpg.group():
                        # -- Effort visualization
                        effort_limit = effort_limits[local_id].item()
                        with dpg.plot(
                            no_title=True,
                            no_menus=True,
                            no_box_select=True,
                            no_mouse_pos=True,
                            height=100,
                            width=50,
                        ):
                            # For a vertical bar, the X axis is position, Y axis is height.
                            dpg.add_plot_axis(
                                dpg.mvXAxis,
                                no_gridlines=True,
                                no_tick_marks=True,
                                no_tick_labels=True,
                            )
                            dpg.set_axis_limits(dpg.last_item(), -1, 1)

                            with dpg.plot_axis(
                                dpg.mvYAxis,
                                no_gridlines=True,
                                no_tick_marks=True,
                                no_tick_labels=True,
                            ) as y_axis:
                                dpg.set_axis_limits(y_axis, -effort_limit, effort_limit)
                                # Use a vertical stem series
                                bar_tag = dpg.add_stem_series([0.0], [0.0])
                                dpg.bind_item_theme(bar_tag, effort_bar_theme)
                                effort_bar_tags.append(bar_tag)

                dpg.add_separator()

        # Finalize and start event loop (blocking in this thread only)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            # Read data from asset
            with self._lock:
                desired_pos = self._desired_pos[0].clone()
                # applied_effort is on the asset's device
                applied_effort = self._select_joints(self._asset.data.applied_torque.torch).clone()
            # Move data to CPU
            desired_pos_cpu = desired_pos.cpu()
            applied_effort_cpu = applied_effort.cpu()
            # Keep position sliders synchronized with an active transition and update effort bars.
            for i in range(len(self._joint_names)):
                dpg.set_value(pos_slider_tags[i], float(desired_pos_cpu[i]))
                effort_val = float(applied_effort_cpu[0, i])
                # For vertical bar, we update the y-value (height) and keep x-value (position) constant.
                dpg.set_value(effort_bar_tags[i], [[0.0], [effort_val]])

            # Render the GUI frame
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    # ---------------------------------------------------------------------
    # Overridden ActionTerm methods
    # ---------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: D401, ARG002
        """Ignore incoming policy actions; GUI values are authoritative."""
        # Advance once per control step so the duration is independent of physics
        # decimation and rendering speed.
        self._advance_position_transition(self._env.step_dt)

    def apply_actions(self) -> None:  # noqa: D401
        """Send joint targets from GUI to the articulation."""
        with self._lock:
            target_pos = self._desired_pos.clone()
            target_stiffness = self._desired_stiffness.clone()
            target_damping = self._desired_damping.clone()
        # Ensure tensor is on the correct device
        target_pos = target_pos.to(device=self.device)
        target_stiffness = target_stiffness.to(device=self.device)
        target_damping = target_damping.to(device=self.device)

        # Mirror actions (if enabled).  The mirror function is expected to work on the
        # provided joint subset length.  When all joints are selected, *target_pos*
        # already contains the full vector; for subsets, the mirror function should be
        # defined for that specific subset ordering.

        if self._mirror_actions:
            _, augmented_actions = self._symmetry_augmentation_func(self._env, None, target_pos, "policy")
            mirrored_pos = augmented_actions.chunk(2, dim=0)[1]  # type: ignore
        else:
            mirrored_pos = target_pos

        mask = (torch.arange(target_pos.shape[0], device=target_pos.device) % 2 == 0).unsqueeze(1)

        full_pos = torch.where(mask, target_pos, mirrored_pos)

        self._asset.set_joint_position_target(full_pos, joint_ids=self._joint_ids)

        self._write_selected_actuator_property("stiffness", target_stiffness)
        self._write_selected_actuator_property("damping", target_damping)

    # ---------------------------------------------------------------------
    # Misc helpers
    # ---------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:  # noqa: D401
        """Reset the action term (called on environment reset)."""
        super().reset(env_ids)
        if env_ids is None:
            resets_first_env = True
        elif isinstance(env_ids, slice):
            resets_first_env = 0 in range(self.num_envs)[env_ids]
        else:
            resets_first_env = 0 in env_ids

        if resets_first_env:
            with self._lock:
                self._desired_pos[...] = self._select_joints(self._asset.data.joint_pos.torch).clone()
                self._desired_stiffness[...] = self._default_stiffness.clone()
                self._desired_damping[...] = self._default_damping.clone()
                self._position_transition_active = False

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _start_position_transition(self, target_position: torch.Tensor, *, restore_default_gains: bool = False) -> None:
        """Start a smooth transition from the current command to ``target_position``."""
        target_position = torch.as_tensor(
            target_position,
            device=self._desired_pos.device,
            dtype=self._desired_pos.dtype,
        )
        if target_position.ndim == 1:
            target_position = target_position.unsqueeze(0).expand_as(self._desired_pos)
        elif target_position.shape == (1, self._desired_pos.shape[1]):
            target_position = target_position.expand_as(self._desired_pos)
        elif target_position.shape != self._desired_pos.shape:
            raise ValueError(
                f"Expected target position shape {tuple(self._desired_pos.shape)}, got {tuple(target_position.shape)}"
            )

        with self._lock:
            self._position_transition_start.copy_(self._desired_pos)
            self._position_transition_target.copy_(target_position)
            self._position_transition_elapsed_s = 0.0
            self._position_transition_duration_s = self._pose_transition_duration_s
            self._position_transition_active = True
            if restore_default_gains:
                self._desired_stiffness.copy_(self._default_stiffness)
                self._desired_damping.copy_(self._default_damping)

    def _start_default_position_transition(self) -> None:
        """Smoothly return to the default pose and immediately restore default gains."""
        default_position = self._select_joints(self._asset.data.default_joint_pos.torch)
        self._start_position_transition(default_position, restore_default_gains=True)

    def _advance_position_transition(self, dt: float) -> None:
        """Advance the active pose transition by one control timestep."""
        with self._lock:
            if not self._position_transition_active:
                return

            self._position_transition_elapsed_s = min(
                self._position_transition_elapsed_s + float(dt), self._position_transition_duration_s
            )
            progress = self._position_transition_elapsed_s / self._position_transition_duration_s
            # Quintic minimum-jerk blend: zero velocity and acceleration at both ends.
            blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
            self._desired_pos.copy_(
                torch.lerp(self._position_transition_start, self._position_transition_target, blend)
            )

            if progress >= 1.0:
                self._desired_pos.copy_(self._position_transition_target)
                self._position_transition_active = False

    def _set_desired_joint_position(self, joint_index: int, position: float) -> None:
        """Apply a manual slider edit and cancel any active pose transition."""
        with self._lock:
            self._position_transition_active = False
            self._desired_pos[:, joint_index] = position

    def _resolve_actuator_bindings(self) -> list[tuple[object, torch.Tensor, torch.Tensor]]:
        """Map each selected action joint to its column in the owning actuator."""
        action_columns = {joint_name: column for column, joint_name in enumerate(self._joint_names)}
        if len(action_columns) != len(self._joint_names):
            raise ValueError(f"Duplicate joint names in GUI action: {self._joint_names}")

        bindings = []
        covered_joint_names = set()
        for actuator in self._asset.actuators.values():
            actuator_columns = []
            selected_columns = []
            for actuator_column, joint_name in enumerate(actuator.joint_names):
                selected_column = action_columns.get(joint_name)
                if selected_column is not None:
                    actuator_columns.append(actuator_column)
                    selected_columns.append(selected_column)
                    covered_joint_names.add(joint_name)

            if actuator_columns:
                bindings.append(
                    (
                        actuator,
                        torch.tensor(actuator_columns, dtype=torch.long, device=self.device),
                        torch.tensor(selected_columns, dtype=torch.long, device=self.device),
                    )
                )

        uncovered_joint_names = set(self._joint_names) - covered_joint_names
        if uncovered_joint_names:
            raise ValueError(f"GUI action joints have no actuator: {sorted(uncovered_joint_names)}")

        return bindings

    def _read_selected_actuator_property(self, property_name: str) -> torch.Tensor:
        """Read an actuator property in the selected action-joint order."""
        first_actuator = self._actuator_bindings[0][0]
        selected_values = getattr(first_actuator, property_name).new_zeros((self.num_envs, len(self._joint_names)))
        for actuator, actuator_columns, selected_columns in self._actuator_bindings:
            actuator_values = getattr(actuator, property_name)
            selected_values[:, selected_columns] = actuator_values[:, actuator_columns]
        return selected_values

    def _write_selected_actuator_property(self, property_name: str, selected_values: torch.Tensor) -> None:
        """Write selected action-joint values back to their owning actuators."""
        for actuator, actuator_columns, selected_columns in self._actuator_bindings:
            actuator_values = getattr(actuator, property_name)
            actuator_values[:, actuator_columns] = selected_values[:, selected_columns]

    def _select_joints(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely select the joint columns specified by *self._joint_ids* from *tensor*.

        This utility handles both slice objects and lists/tuples/tensors of indices and
        works transparently on CPU and GPU tensors.  The joint dimension is assumed to
        be the second axis (index 1).  The returned tensor shares the same device as
        the input tensor.
        """

        if isinstance(self._joint_ids, slice):
            # Prepare a slicing tuple that keeps all other dimensions intact.
            slicer = [slice(None)] * tensor.ndim
            slicer[1] = self._joint_ids
            return tensor[tuple(slicer)]

        # Convert list/tuple to a *LongTensor* that lives on the same device as *tensor*.
        index_tensor = torch.as_tensor(self._joint_ids, dtype=torch.long, device=tensor.device)
        return tensor.index_select(1, index_tensor)
