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

import threading
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
        self._desired_pos = self._select_joints(self._asset.data.joint_pos).clone()
        # Desired PD gains for the robot actuators
        self._desired_stiffness = self._select_joints(
            torch.cat([v.stiffness for v in self._asset.actuators.values()], dim=-1)
        ).clone()
        self._desired_damping = self._select_joints(
            torch.cat([v.damping for v in self._asset.actuators.values()], dim=-1)
        ).clone()
        # Store default gains for reset
        self._default_stiffness = self._desired_stiffness.clone()
        self._default_damping = self._desired_damping.clone()
        self._mirror_actions = cfg.mirror_actions
        self._robot_type = cfg.robot_type

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

        # Create context and viewport
        dpg.create_context()
        dpg.create_viewport(title="Joint Position Controller", width=600, height=1000)

        with dpg.window(label="Joint Position Controller", tag="primary_window"):
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

            def _reset_joints_cb() -> None:
                """Reset all joints to their default positions and gains."""
                with self._lock:
                    # Default positions for all envs
                    default_pos = self._select_joints(self._asset.data.default_joint_pos)
                    self._desired_pos[:] = default_pos.clone()
                    self._desired_stiffness[:] = self._default_stiffness.clone()
                    self._desired_damping[:] = self._default_damping.clone()
                    # Update all GUI sliders
                    for i in range(len(self._joint_names)):
                        dpg.set_value(pos_slider_tags[i], float(self._desired_pos[0, i].cpu()))
                        dpg.set_value(
                            stiffness_slider_tags[i],
                            float(self._desired_stiffness[0, i].cpu()),
                        )
                        dpg.set_value(
                            damping_slider_tags[i],
                            float(self._desired_damping[0, i].cpu()),
                        )

            def _randomize_joints_cb() -> None:
                """Randomize all joints within their limits."""
                with self._lock:
                    limits = self._select_joints(self._asset.data.soft_joint_pos_limits[0].T).T.cpu()
                    low = limits[:, 0]
                    high = limits[:, 1]
                    # Generate random positions
                    random_pos = low + (high - low) * torch.rand_like(low)
                    # Update desired positions for all environments
                    self._desired_pos[:] = random_pos.unsqueeze(0)
                    # Update GUI sliders
                    for i in range(len(self._joint_names)):
                        dpg.set_value(pos_slider_tags[i], float(self._desired_pos[0, i].cpu()))

            # Add reset button
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset to Default", callback=_reset_joints_cb)
                dpg.add_button(label="Randomize", callback=_randomize_joints_cb)
            dpg.add_separator()

            # Iterate over joints
            for local_id, joint_name in enumerate(self._joint_names):
                with dpg.group(horizontal=True):
                    # Group for sliders on the left
                    with dpg.group():
                        # Resolve the global joint index when self._joint_ids is a slice
                        joint_idx = local_id if isinstance(self._joint_ids, slice) else self._joint_ids[local_id]

                        # Fetch soft limits (first environment)
                        limits = self._asset.data.soft_joint_pos_limits[0, joint_idx].cpu()
                        low, high = float(limits[0]), float(limits[1])
                        current_val = float(self._desired_pos[0, local_id].cpu())
                        current_stiffness = float(self._desired_stiffness[0, local_id].cpu())
                        current_damping = float(self._desired_damping[0, local_id].cpu())

                        # -- Position slider
                        def _slider_cb(sender, app_data, user_data) -> None:  # type: ignore  # noqa: ARG001
                            idx: int = user_data
                            with self._lock:
                                self._desired_pos[:, idx] = float(app_data)

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
                            max_value=self.cfg.max_stiffness,
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
                            max_value=self.cfg.max_damping,
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
                        effort_limit = torch.cat(
                            [v.effort_limit for v in self._asset.actuators.values()],
                            dim=-1,
                        )[0, joint_idx].item()
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

        # Set the window as primary so it auto-resizes with the viewport
        dpg.set_primary_window("primary_window", True)

        # Finalize and start event loop (blocking in this thread only)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            # Read data from asset
            with self._lock:
                # applied_effort is on the asset's device
                applied_effort = self._select_joints(self._asset.data.applied_torque).clone()
            # Move data to CPU
            applied_effort_cpu = applied_effort.cpu()
            # Update all effort bars
            for i in range(len(self._joint_names)):
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
        # No-op to ensure that RL actions don't override GUI commands.
        return None

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

        # Apply PD gains
        if isinstance(self._joint_ids, slice):
            # Create a tensor representing the range of the slice
            joint_ids_tensor = torch.arange(
                self._joint_ids.start or 0,
                self._joint_ids.stop,
                self._joint_ids.step or 1,
                device=self.device,
            )
        else:
            joint_ids_tensor = torch.tensor(self._joint_ids, device=self.device)

        # Create a full tensor for all stiffness and damping values
        full_stiffness = torch.cat(
            [actuator.stiffness.clone() for actuator in self._asset.actuators.values()],
            dim=1,
        )
        full_damping = torch.cat(
            [actuator.damping.clone() for actuator in self._asset.actuators.values()],
            dim=1,
        )

        # Update the values for the selected joints
        full_stiffness[:, joint_ids_tensor] = target_stiffness
        full_damping[:, joint_ids_tensor] = target_damping

        # Distribute the updated values back to the actuators
        offset = 0
        for actuator in self._asset.actuators.values():
            num_dof = actuator.stiffness.shape[1]
            actuator.stiffness[:] = full_stiffness.narrow(1, offset, num_dof)
            actuator.damping[:] = full_damping.narrow(1, offset, num_dof)
            offset += num_dof

    # ---------------------------------------------------------------------
    # Misc helpers
    # ---------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:  # noqa: D401
        """Reset the action term (called on environment reset)."""
        super().reset(env_ids)
        # Synchronize GUI sliders with environment after reset (optional)
        if env_ids is None or 0 in env_ids:
            with self._lock:
                self._desired_pos[...] = self._select_joints(self._asset.data.joint_pos).clone()
                self._desired_stiffness[...] = self._default_stiffness.clone()
                self._desired_damping[...] = self._default_damping.clone()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

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
