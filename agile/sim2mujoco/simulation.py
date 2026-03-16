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

"""MuJoCo simulation wrapper."""

import os
import time as _time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import torch


@dataclass
class SimState:
    """State of the simulation."""

    joint_pos: torch.Tensor
    """Joint positions, shape (num_joints,)."""
    joint_vel: torch.Tensor
    """Joint velocities, shape (num_joints,)."""
    root_pos: torch.Tensor
    """Root position in world frame, shape (3,)."""
    root_quat: torch.Tensor
    """Root orientation quaternion [w, x, y, z], shape (4,)."""
    root_lin_vel: torch.Tensor
    """Root linear velocity in root frame, shape (3,)."""
    root_ang_vel: torch.Tensor
    """Root angular velocity in root frame, shape (3,)."""
    joint_effort: torch.Tensor | None = None
    """Joint efforts/torques, shape (num_joints,). Optional."""
    anchor_body_pos: torch.Tensor | None = None
    """Anchor body position in world frame, shape (3,). Populated when motion_tracking is configured."""
    anchor_body_quat: torch.Tensor | None = None
    """Anchor body orientation quaternion [w, x, y, z], shape (4,). Populated when motion_tracking is configured."""


@dataclass
class JointCommand:
    """Joint command with PD control parameters."""

    position: torch.Tensor
    """Desired joint positions, shape (num_joints,)."""
    kp: torch.Tensor
    """Proportional gains, shape (num_joints,)."""
    kd: torch.Tensor
    """Derivative gains, shape (num_joints,)."""


class DummyViewer:
    """Dummy viewer for headless operation."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def sync(self):
        pass

    def is_running(self) -> bool:
        return True


class MuJocoSimulation:
    """MuJoCo simulation wrapper."""

    def __init__(
        self,
        config: dict,
        device: torch.device,
        enable_viewer: bool = True,
        mjcf_path: Path | None = None,
        command_manager=None,
    ):
        """
        Initialize MuJoCo simulation.

        Args:
            config: Configuration dictionary from YAML.
            device: Device for torch tensors.
            enable_viewer: Whether to enable the viewer.
            mjcf_path: Path to MJCF file (overrides config).
            command_manager: Optional CommandManager for interactive control.
        """
        self.device = device
        self.config = config
        self.command_manager = command_manager

        # Pause / single-step control (toggled via keyboard).
        self.paused = enable_viewer
        self.step_once = False

        # Get MJCF path.
        if mjcf_path is not None:
            self.mjcf_path = mjcf_path
        elif "mjcf_path" in config:
            self.mjcf_path = Path(config["mjcf_path"])
        else:
            raise ValueError("MJCF path must be provided via config or argument")

        # Load MuJoCo model.
        print(f"Loading MJCF from: {self.mjcf_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.mj_data = mujoco.MjData(self.mj_model)

        # Set timestep from config.
        scene_config = config.get("scene", {})

        # Use physics_dt if available, otherwise fall back to dt
        if "physics_dt" in scene_config:
            self.mj_model.opt.timestep = scene_config["physics_dt"]
        elif "dt" in scene_config:
            # If only dt is provided, assume it's the physics timestep
            self.mj_model.opt.timestep = scene_config["dt"]

        self.physics_dt = self.mj_model.opt.timestep
        self.decimation = scene_config.get("decimation", 4)

        # Control dt is physics_dt * decimation
        self.dt = self.physics_dt * self.decimation

        # Extract joint names (skip freejoint if present).
        self.joint_names = []
        for i in range(self.mj_model.njnt):
            joint_type = self.mj_model.jnt_type[i]
            if joint_type != mujoco.mjtJoint.mjJNT_FREE:  # Skip freejoint.
                joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                self.joint_names.append(joint_name)

        self.num_joints = len(self.joint_names)
        print(f"Found {self.num_joints} actuated joints: {self.joint_names}")

        # Determine if fixed base.
        self.fixed_base = not self._has_freejoint()

        # Detect actuator type.
        self._detect_actuator_type()

        # Create mapping from joint names to actuator/control indices.
        # This matches isaac-deploy's _joint_to_ctrl_indices implementation.
        # The actuators in MuJoCo might be in different order than joints.
        self._joint_to_ctrl_indices = []
        for joint_name in self.joint_names:
            # Find the actuator index that controls this joint.
            for i in range(self.mj_model.nu):
                actuator_joint_id = self.mj_model.actuator_trnid[i, 0]
                if actuator_joint_id >= 0:  # Valid joint actuator.
                    actuator_joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_joint_id)
                    if actuator_joint_name == joint_name:
                        self._joint_to_ctrl_indices.append(i)
                        break

        if len(self._joint_to_ctrl_indices) != self.num_joints:
            print(
                f"Warning: Found {len(self._joint_to_ctrl_indices)} actuators "
                f"for {self.num_joints} joints. Some joints may not be actuated."
            )

        print(f"  Joint to actuator mapping: {self._joint_to_ctrl_indices}")

        # Setup viewer.
        if enable_viewer and "DISPLAY" in os.environ:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self._key_callback)
            # Configure camera.
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0, 0, 1.0]
                self.viewer.cam.distance = 3.0
                self.viewer.cam.azimuth = 135.0
                self.viewer.cam.elevation = -20.0

            self._print_keyboard_help()
        else:
            self.viewer = DummyViewer()
            self.viewer.__enter__()
            self.paused = False

        # Setup anchor body tracking for motion tracking tasks.
        motion_tracking = config.get("motion_tracking")
        if motion_tracking and "anchor_body_name" in motion_tracking:
            anchor_name = motion_tracking["anchor_body_name"]
            self._anchor_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, anchor_name)
            if self._anchor_body_id < 0:
                raise ValueError(f"Anchor body '{anchor_name}' not found in MJCF model")
            print(f"  Anchor body: {anchor_name} (body_id={self._anchor_body_id})")
        else:
            self._anchor_body_id = None

        # Setup velocity sensors.
        self._setup_sensors()

        # Set initial configuration.
        self._set_initial_configuration(config)

        # Push force disturbance state.
        self._push_force_magnitude = 100.0  # Newtons
        self._push_sign = 0.0  # +1 or -1
        self._push_axis = 0  # 0 = body x (fwd/bwd), 1 = body y (left/right)
        self._push_expiry = 0.0
        push_body_name = "torso_link"
        self._push_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, push_body_name)
        if self._push_body_id < 0:
            self._push_body_id = 1  # fallback to first non-world body

    def _has_freejoint(self) -> bool:
        """Check if model has a freejoint (floating base)."""
        for i in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                return True
        return False

    def _world_to_root_frame(self, world_vec: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        """
        Transform a vector from world frame to root frame.

        Args:
            world_vec: Vector in world frame, shape (3,).
            root_quat: Root orientation quaternion [w, x, y, z], shape (4,).

        Returns:
            Vector in root frame, shape (3,).
        """
        # Import the function from utils.
        from agile.sim2mujoco.utils import quat_rotate_inverse

        return quat_rotate_inverse(root_quat, world_vec)

    def _detect_actuator_type(self):
        """
        Detect actuator type from MuJoCo model.

        Actuators can be:
        - 'position': Built-in PD control (biastype == 1)
        - 'torque': Direct torque control (biastype == 0)
        """
        if self.mj_model.nu == 0:
            self.actuator_type = "torque"
            return

        # Check bias types.
        actuator_bias_types = [self.mj_model.actuator(i).biastype for i in range(self.mj_model.nu)]

        # Verify all actuators have the same type.
        if not all(bias_type == actuator_bias_types[0] for bias_type in actuator_bias_types):
            print("Warning: Mixed actuator types detected, using first type")

        if actuator_bias_types[0] == 0:  # Bias: None -> Torque mode.
            self.actuator_type = "torque"
        elif actuator_bias_types[0] == 1:  # Bias: Affine -> Position mode with PD.
            self.actuator_type = "position"
        else:
            print(f"Warning: Unknown actuator bias type {actuator_bias_types[0]}, defaulting to torque")
            self.actuator_type = "torque"

        print(f"  Actuator type: {self.actuator_type}")

    def _setup_sensors(self):
        """Setup velocity sensors - matching isaac-deploy implementation."""
        # Initialize sensors to None.
        self._root_linear_velocity_sensor = None
        self._root_angular_velocity_sensor = None
        self._root_linear_acceleration_sensor = None

        # Try to find sensors by name (matching isaac-deploy naming).
        try:
            self._root_linear_velocity_sensor = self.mj_data.sensor("linear-velocity")
        except KeyError:
            print("Warning: 'linear-velocity' sensor not found, will compute from qvel")

        try:
            self._root_angular_velocity_sensor = self.mj_data.sensor("angular-velocity")
        except KeyError:
            print("Warning: 'angular-velocity' sensor not found, will compute from qvel")

        try:
            self._root_linear_acceleration_sensor = self.mj_data.sensor("linear-acceleration")
        except KeyError:
            print("Warning: 'linear-acceleration' sensor not found")

    def _set_initial_configuration(self, config: dict):
        """Set initial joint positions from config.

        Reads ``articulations.robot.default_joint_pos`` (in IsaacLab joint order)
        and maps them to MuJoCo joint order so the robot starts in the correct
        default pose.  The mapped positions are stored in ``_default_qpos`` and
        written to ``mj_data.qpos`` on every ``reset()`` call.
        """
        # Default qpos for the full configuration vector.
        import numpy as np

        self._default_qpos = np.zeros(self.mj_model.nq)

        # Set default root position for floating base robots.
        if not self.fixed_base:
            self._default_qpos[:3] = [0.0, 0.0, 0.76]
            self._default_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Apply default joint positions from config (YAML/policy joint order -> MJCF order).
        robot_cfg = config.get("articulations", {}).get("robot", {})
        default_joint_pos = robot_cfg.get("default_joint_pos")
        yaml_joint_names = robot_cfg.get("joint_names")

        if default_joint_pos is not None and yaml_joint_names is not None:
            if len(default_joint_pos) != len(yaml_joint_names):
                print(
                    f"Warning: default_joint_pos ({len(default_joint_pos)}) and "
                    f"joint_names ({len(yaml_joint_names)}) have different lengths"
                )
                return

            # Map from YAML joint order to MJCF joint order using jnt_qposadr.
            for yaml_idx, jname in enumerate(yaml_joint_names):
                jnt_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jnt_id >= 0:
                    qpos_addr = self.mj_model.jnt_qposadr[jnt_id]
                    self._default_qpos[qpos_addr] = default_joint_pos[yaml_idx]
                else:
                    print(f"Warning: joint '{jname}' from config not found in MJCF model")

            print(f"  Applied default joint positions from config ({len(yaml_joint_names)} joints)")

        # Override MJCF joint frictionloss and damping for all actuated joints.
        # frictionloss: MJCF has 0.2, set to 0.1 (matching holosoma G1) for
        # realistic Coulomb friction while being less aggressive than the MJCF default.
        # damping: MJCF has 0.05, zero it out since the PD controller already
        # handles all damping via kd gains.
        for i in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                dof_addr = self.mj_model.jnt_dofadr[i]
                self.mj_model.dof_frictionloss[dof_addr] = 0.1
                self.mj_model.dof_damping[dof_addr] = 0.0
        print("  Set joint frictionloss=0.1, damping=0.0 for all actuated joints")

        # Apply per-joint armature from config to match training dynamics.
        # The MJCF uses a flat armature=0.01 for all joints, but training uses
        # per-joint values ranging from 0.0036 to 0.025.
        default_joint_armature = robot_cfg.get("default_joint_armature")
        if default_joint_armature is not None and yaml_joint_names is not None:
            for yaml_idx, jname in enumerate(yaml_joint_names):
                jnt_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jnt_id >= 0:
                    dof_addr = self.mj_model.jnt_dofadr[jnt_id]
                    self.mj_model.dof_armature[dof_addr] = default_joint_armature[yaml_idx]
            print(f"  Applied joint armature from config ({len(yaml_joint_names)} joints)")

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # Apply stored default configuration directly to qpos.
        self.mj_data.qpos[:] = self._default_qpos
        self.mj_data.ctrl[:] = 0
        self._push_sign = 0.0
        # Forward kinematics to compute derived quantities.
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_state(self) -> SimState:
        """
        Get current simulation state.

        Returns:
            SimState object with current state.
        """
        if self.fixed_base:
            # Fixed base robot.
            joint_pos = torch.from_numpy(self.mj_data.qpos.copy()).to(self.device)
            joint_vel = torch.from_numpy(self.mj_data.qvel.copy()).to(self.device)
            root_pos = torch.zeros(3, device=self.device)
            root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            root_lin_vel = torch.zeros(3, device=self.device)
            root_ang_vel = torch.zeros(3, device=self.device)
        else:
            # Floating base robot.
            root_pos = torch.from_numpy(self.mj_data.qpos[:3].copy()).to(self.device)
            root_quat = torch.from_numpy(self.mj_data.qpos[3:7].copy()).to(self.device)
            joint_pos = torch.from_numpy(self.mj_data.qpos[7:].copy()).to(self.device)

            joint_vel = torch.from_numpy(self.mj_data.qvel[6:].copy()).to(self.device)

            # Get velocities from sensors (matching isaac-deploy implementation).
            # Sensors provide velocities in root frame directly from MuJoCo.
            if self._root_linear_velocity_sensor is not None:
                root_lin_vel = torch.from_numpy(self._root_linear_velocity_sensor.data.copy()).to(self.device)
            else:
                # Fallback: Compute COM velocity from body origin velocity.
                # qvel[:3] is the world-frame velocity of the body origin (joint frame),
                # but IsaacLab reports root_lin_vel as the COM velocity.
                # v_com = v_origin + omega x (R * com_local)
                from agile.sim2mujoco.utils import quat_apply

                world_lin_vel = torch.from_numpy(self.mj_data.qvel[:3].copy()).to(self.device)
                # root_lin_vel = self._world_to_root_frame(world_lin_vel, root_quat)
                world_ang_vel = torch.from_numpy(self.mj_data.qvel[3:6].copy()).to(self.device)

                root_body_id = self.mj_model.jnt_bodyid[0]
                com_local = torch.from_numpy(self.mj_model.body_ipos[root_body_id].copy()).to(self.device)
                com_world = quat_apply(root_quat, com_local)

                world_com_vel = world_lin_vel + torch.cross(world_ang_vel, com_world, dim=0)
                root_lin_vel = self._world_to_root_frame(world_com_vel, root_quat)

            if self._root_angular_velocity_sensor is not None:
                root_ang_vel = torch.from_numpy(self._root_angular_velocity_sensor.data.copy()).to(self.device)
            else:
                # Fallback: Convert world frame angular velocity to root frame.
                world_ang_vel = torch.from_numpy(self.mj_data.qvel[3:6].copy()).to(self.device)
                root_ang_vel = self._world_to_root_frame(world_ang_vel, root_quat)

        if self.fixed_base:
            joint_effort = torch.from_numpy(self.mj_data.qfrc_actuator.copy()).to(self.device)
        else:
            joint_effort = torch.from_numpy(self.mj_data.qfrc_actuator[6:].copy()).to(self.device)

        # Anchor body state (for motion tracking tasks).
        anchor_body_pos = None
        anchor_body_quat = None
        if self._anchor_body_id is not None:
            anchor_body_pos = torch.from_numpy(self.mj_data.xpos[self._anchor_body_id].copy()).to(self.device)
            anchor_body_quat = torch.from_numpy(self.mj_data.xquat[self._anchor_body_id].copy()).to(self.device)

        return SimState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_effort=joint_effort,
            anchor_body_pos=anchor_body_pos,
            anchor_body_quat=anchor_body_quat,
        )

    def step(self, joint_cmd: JointCommand):
        """
        Step the simulation with joint commands.

        Args:
            joint_cmd: Joint command with positions and PD gains.
        """
        if self.actuator_type == "position":
            # Position actuators: MuJoCo handles PD control internally.
            # Map joint order to actuator order (matching isaac-deploy).
            joint_positions_np = joint_cmd.position.cpu().numpy()
            joint_kp_np = joint_cmd.kp.cpu().numpy()
            joint_kd_np = joint_cmd.kd.cpu().numpy()

            # Set target positions in actuator order.
            self.mj_data.ctrl[:] = joint_positions_np[self._joint_to_ctrl_indices]

            # Update PD gains in actuator parameters.
            # See: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position
            self.mj_model.actuator_gainprm[:, 0] = joint_kp_np[self._joint_to_ctrl_indices]
            self.mj_model.actuator_biasprm[:, 1] = -joint_kp_np[self._joint_to_ctrl_indices]
            self.mj_model.actuator_biasprm[:, 2] = -joint_kd_np[self._joint_to_ctrl_indices]

        else:  # torque mode
            # Torque actuators: Compute torques manually.
            state = self.get_state()

            pos_error = joint_cmd.position - state.joint_pos
            vel_error = -state.joint_vel  # Assuming zero target velocity.

            torques = joint_cmd.kp * pos_error + joint_cmd.kd * vel_error

            # Store computed torques in state (matching isaac-deploy line 349).
            state.joint_effort = torques

            # Map joint order to actuator order (matching isaac-deploy line 350).
            self.mj_data.ctrl[:] = torques[self._joint_to_ctrl_indices].cpu().numpy()

        # Apply push force disturbance if active.
        if self._push_sign != 0.0 and _time.time() < self._push_expiry:
            xmat = self.mj_data.xmat[self._push_body_id].reshape(3, 3)
            direction = xmat[:, self._push_axis]  # body x or y axis in world frame
            self.mj_data.xfrc_applied[self._push_body_id, :3] = self._push_sign * self._push_force_magnitude * direction
        else:
            self.mj_data.xfrc_applied[self._push_body_id, :] = 0
            self._push_sign = 0.0

        # Step simulation.
        mujoco.mj_step(self.mj_model, self.mj_data)

    def close(self):
        """Close simulation and viewer."""
        if hasattr(self.viewer, "__exit__"):
            self.viewer.__exit__(None, None, None)

    def _key_callback(self, key: int):
        """
        Keyboard callback for simulation and command control.

        Args:
            key: GLFW key code.
        """
        import glfw

        # --- Pause / step controls (always active) ---
        if key == glfw.KEY_SPACE:
            self.paused = not self.paused
            print(f"{'⏸  PAUSED' if self.paused else '▶  RUNNING'}")
            return
        elif key == glfw.KEY_N:
            self.step_once = True
            return

        # --- Push force disturbance ---
        if key in (glfw.KEY_F, glfw.KEY_B, glfw.KEY_G, glfw.KEY_V):
            directions = {
                glfw.KEY_F: (1.0, 0),  # forward  (+x body axis)
                glfw.KEY_B: (-1.0, 0),  # backward (-x body axis)
                glfw.KEY_G: (1.0, 1),  # left     (+y body axis)
                glfw.KEY_V: (-1.0, 1),  # right    (-y body axis)
            }
            self._push_sign, self._push_axis = directions[key]
            self._push_expiry = _time.time() + 0.15
            return

        # --- Command manager controls (only when available) ---
        if self.command_manager is None:
            return

        # Forward/Backward (Arrow Up/Down or I/K)
        if key == glfw.KEY_UP or key == glfw.KEY_I:
            self.command_manager.update_linear_x(self.command_manager.vel_step)
            self.command_manager.print_status()
        elif key == glfw.KEY_DOWN or key == glfw.KEY_K:
            self.command_manager.update_linear_x(-self.command_manager.vel_step)
            self.command_manager.print_status()

        # Left/Right strafe (Arrow Left/Right or J/L)
        elif key == glfw.KEY_LEFT or key == glfw.KEY_J:
            self.command_manager.update_linear_y(self.command_manager.vel_step)
            self.command_manager.print_status()
        elif key == glfw.KEY_RIGHT or key == glfw.KEY_L:
            self.command_manager.update_linear_y(-self.command_manager.vel_step)
            self.command_manager.print_status()

        # Turn Left/Right (U/O)
        elif key == glfw.KEY_U:
            self.command_manager.update_angular_z(self.command_manager.ang_step)
            self.command_manager.print_status()
        elif key == glfw.KEY_O:
            self.command_manager.update_angular_z(-self.command_manager.ang_step)
            self.command_manager.print_status()

        # Height Up/Down (Page Up/Down or 9/0 on number row)
        elif key == glfw.KEY_PAGE_UP or key == glfw.KEY_9:
            self.command_manager.update_height(self.command_manager.height_step)
            self.command_manager.print_status()
        elif key == glfw.KEY_PAGE_DOWN or key == glfw.KEY_0:
            self.command_manager.update_height(-self.command_manager.height_step)
            self.command_manager.print_status()

        # STOP - Reset to defaults (H for "Home")
        elif key == glfw.KEY_H:
            self.command_manager.stop()

        # Print status (P)
        elif key == glfw.KEY_P:
            self.command_manager.print_status()

    def _print_keyboard_help(self):
        """Print keyboard control instructions."""
        print("\n" + "=" * 80)
        print("⌨️  KEYBOARD CONTROLS")
        print("=" * 80)
        print("  ⚠️  NOTE: Click on the MuJoCo viewer window to enable keyboard input!")
        print("=" * 80)
        print("  SPACE                    : Pause / Resume simulation")
        print("  N                        : Step one frame (while paused)")
        print("  F / B (hold)             : Push robot forward / backward (100 N)")
        print("  G / V (hold)             : Push robot left / right      (100 N)")
        if self.command_manager is not None:
            print("  " + "-" * 56)
            print("  ↑ / I     or  ↓ / K      : Forward / Backward    (±0.1 m/s,   [-0.5, 0.5])")
            print("  ← / J     or  → / L      : Left / Right strafe   (±0.1 m/s,   [-0.5, 0.5])")
            print("  U         or  O          : Turn Left / Right     (±0.2 rad/s, [-1.0, 1.0])")
            print("  PgUp / 9  or  PgDn / 0   : Height Up / Down      (±0.05 m,    [0.3, 0.8])")
            print("  H                        : STOP (reset commands to defaults)")
            print("  P                        : Print current status")
        print("=" * 80)
        print("  ⏸  Simulation starts PAUSED. Press SPACE to begin.")
        print("=" * 80 + "\n")
