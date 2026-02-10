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

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from agile.rl_env.assets import ASSET_DIR
from agile.rl_env.assets.robots.unitree_g1 import (
    ARM_JOINT_NAMES,
    G1_W_HANDS_AGILE_ACTION_SCALE,
    G1_W_HANDS_AGILE_CFG,
    HAND_JOINT_NAMES,
    WAIST_JOINT_NAMES,
)
from agile.rl_env.mdp.actions import JointPositionGUIActionCfg, ObjectPoseGUIActionCfg
from agile.rl_env.mdp.rewards import RewardVisualizerCfg
from agile.rl_env.tasks.pick_place.pick_place_domain_randomization import RecordRandomizationEventCfg
from agile.rl_env.tasks.pick_place.pick_place_tracking_env_cfg import ObservationsCfg, PickPlaceTrackingEnvCfg


@configclass
class G1PickPlaceTrackingEnvCfg(PickPlaceTrackingEnvCfg):
    """Configuration for the G1 pick-place tracking environment."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.scene.robot = G1_W_HANDS_AGILE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-1.5, -0.7, 0.8]
        self.scene.robot.init_state.rot = [0.7071068, 0, 0, 0.7071068]
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
            "left_shoulder_pitch_joint": 0.19,
            "left_shoulder_roll_joint": 0.4638,
            "left_shoulder_yaw_joint": -0.2448,
            "left_elbow_joint": 0.9777,
            "left_wrist_roll_joint": -0.0926,
            "left_wrist_pitch_joint": -0.0179,
            "left_wrist_yaw_joint": -0.0225,
            "left_hand_thumb_1_joint": 1.0,
            "left_hand_thumb_2_joint": 0.3,
        }

        self.scene.fixture_structure.init_state.pos = [0.0, 0.0, 0.0]
        self.scene.object.init_state.pos = [0.3872, 0.2480, 0.738]
        self.scene.object.init_state.rot = [0.7071068, 0.7071068, 0.0, 0.0]  # 90° rotation around X

        self.commands.tracking_command.file_path = (
            f"{ASSET_DIR}/motion_data/object_pick_and_place_retarget_motion_g1_3finger_hands.yaml"
        )

        self.actions.lower_body_joint_pos.policy_output_scale = G1_W_HANDS_AGILE_ACTION_SCALE
        # Only include right arm/hand joints in the scale dict (must match joint_names=RIGHT_HAND_ARM_JOINT_NAMES)
        self.actions.upper_body_joint_pos.scale = {
            k: 0.05
            for k in G1_W_HANDS_AGILE_ACTION_SCALE.keys()
            if k in (ARM_JOINT_NAMES + HAND_JOINT_NAMES + WAIST_JOINT_NAMES)
        }

        # rewards
        if hasattr(self.rewards, "lifting_object"):
            self.rewards.lifting_object.params["minimal_height"] = self.scene.object.init_state.pos[2]


@configclass
class G1PickPlaceTrackingEnvCfgRecord(G1PickPlaceTrackingEnvCfg):
    """Configuration for the G1 V2P environment with a camera and visual randomization."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        # Scene configuration overrides
        self.scene.num_envs = 4
        # Disable physics replication for visual texture randomization
        self.scene.replicate_physics = False
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/front_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            # Default pinhole camera configuration for OAK-D camera.
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=[
                    375.1500244140625,
                    0.0,
                    317.8176574707031,
                    0.0,
                    374.9283142089844,
                    248.43441772460938,
                    0.0,
                    0.0,
                    1.0,
                ],
                width=640,
                height=480,
                focal_length=24,
                focus_distance=400.0,
                clipping_range=(0.1, 20.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.45), rot=(-0.29, 0.64, -0.64, 0.29), convention="ros"),
        )

        # Add dome light for visual randomization
        self.scene.light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75),
                intensity=3000.0,
                texture_file="omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
            ),
        )

        # Set the object to be the can only
        self.scene.object.spawn.usd_path = [
            f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"
        ]

        # Add observations for recording
        self.observations.record = ObservationsCfg.RecordObservationsCfg()

        # Disable debug visualization of end-effector frame and tracking command
        self.scene.ee_frame.debug_vis = False
        self.commands.tracking_command.debug_vis = False

        # Use events with visual randomization (dome light + table material)
        self.events = RecordRandomizationEventCfg()
        self.events.reset_robot.params["trajectory_time_idx"] = (0, 5)


@configclass
class G1PickPlaceTrackingEnvCfgGr00tInference(G1PickPlaceTrackingEnvCfgRecord):
    """Configuration for the G1 V2P environment for GR00T inference."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        # Add background for GR00T inference
        self.scene.background = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Background",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0, 0, 0.01),  # Adjusted to avoid height competition issues.
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=None,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
            ),
        )
        # Set scene spacing for GR00T inference
        self.scene.env_spacing = 50

        # Disable termination terms that reference tracking_command
        self.terminations.bad_joint_pos = None
        self.terminations.bad_base_pose.params["base_pos_threshold"] = 0.5
        self.terminations.bad_base_rotation.params["base_ori_threshold"] = 0.5
        self.terminations.object_out_of_bound.params["in_bound_range"] = {"z": [0.3, 1.2]}

        # Disable domain randomization events
        self.events.randomize_light = None
        self.events.rand_ground_texture = None
        self.events.randomize_light_startup = None
        self.events.rand_ground_texture_startup = None


@configclass
class G1PickPlaceTrackingEnvCfgDebug(G1PickPlaceTrackingEnvCfg):
    """Debug configuration for G1 pick-place tracking with interactive GUI controls.

    Adds:
    - Joint position GUI for robot control
    - Object pose GUI for manipulating the object
    - Reward visualizer for monitoring reward terms
    """

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # Reduce envs for debugging
        self.scene.num_envs = 1

        # Increase episode length for interactive debugging
        self.episode_length_s = 3600.0

        # Fix robot root for easier joint debugging
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        # Using a position similar to the motion data starting frame for consistency
        self.scene.robot.init_state.pos = [0.15, -0.5, 0.78]
        # Quaternion [w, x, y, z] for 90° rotation around Z-axis (facing +Y direction)
        self.scene.robot.init_state.rot = [0.7071068, 0.0, 0.0, 0.7071068]

        # Disable original action terms to avoid conflicts with GUI control
        self.actions.upper_body_joint_pos = None  # type: ignore[assignment]
        self.actions.lower_body_joint_pos = None  # type: ignore[assignment]

        # Disable all observations (not needed for interactive debugging)
        self.observations.policy = None  # type: ignore[assignment]
        self.observations.agile_policy = None  # type: ignore[assignment]

        # Disable rewards that reference disabled action terms or tracking commands
        self.rewards.motion_global_anchor_pos = None  # type: ignore[assignment]
        self.rewards.motion_global_anchor_ori = None  # type: ignore[assignment]
        self.rewards.upper_body_joint_pos = None  # type: ignore[assignment]
        self.rewards.action_rate_l2 = None  # type: ignore[assignment]
        self.rewards.dof_vel_l2 = None  # type: ignore[assignment]
        self.rewards.joint_pos_limit = None  # type: ignore[assignment]
        self.rewards.object_pos_tracking = None  # type: ignore[assignment]
        # Only keep: termination_penalty, lifting_object, hand_object_tracking, etc.

        # Disable terminations that reference tracking_command
        self.terminations.bad_base_pose = None  # type: ignore[assignment]
        self.terminations.bad_base_rotation = None  # type: ignore[assignment]
        self.terminations.bad_joint_pos = None  # type: ignore[assignment]
        # Only keep: time_out

        # Disable events that reference tracking_command
        self.events.reset_robot = None  # type: ignore[assignment]

        # Disable commands
        self.commands.tracking_command.debug_vis = False

        # Add finger contact sensors for debugging.
        finger_tip_body_list = ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
                ),
            )

        # Enable contact force visualization on finger sensors
        for link_name in ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]:
            sensor = getattr(self.scene, f"{link_name}_object_s")
            sensor.debug_vis = True

        # Add joint position GUI control
        from agile.rl_env.assets.robots.unitree_g1 import RIGHT_HAND_ARM_JOINT_NAMES

        self.actions.joint_pos_gui = JointPositionGUIActionCfg(
            asset_name="robot",
            joint_names=RIGHT_HAND_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
            preserve_order=True,
            mirror_actions=False,
            robot_type="g1",
        )

        # Add object pose GUI control
        self.actions.object_pose_gui = ObjectPoseGUIActionCfg(
            asset_name="object",
            position_limits={
                "x": (-0.5, 1.0),
                "y": (-0.5, 0.5),
                "z": (0.3, 1.2),
            },
            rotation_limits={
                "roll": (-math.pi, math.pi),
                "pitch": (-math.pi, math.pi),
                "yaw": (-math.pi, math.pi),
            },
            disable_gravity=True,
            gui_window_title="Object Pose Controller",
        )

        # Add reward visualizer
        self.actions.reward_monitor = RewardVisualizerCfg(
            reward_terms=[
                "reaching_object",
                "lifting_object",
            ],  # Only show reaching_object reward
            exclude_terms=[],
            show_total_reward=True,
            show_weights=True,
            show_episode_sum=True,
            enable_history_plot=True,
            gui_window_title="V2P Reward Monitor",
        )

        self.viewer.eye = (2.5, 5.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.viewer.origin_type = "world"
