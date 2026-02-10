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

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

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
from agile.rl_env.tasks.pick_place.pick_place_tracking_env_cfg import PickPlaceTrackingEnvCfg


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
