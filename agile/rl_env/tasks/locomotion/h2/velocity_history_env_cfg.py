# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1-parity velocity-history environment for the Unitree H2."""

from isaaclab.utils.configclass import configclass

from agile.rl_env.assets.robots import unitree_h2
from agile.rl_env.tasks.locomotion.g1.velocity_history_env_cfg import G1LowerVelocityHistoryEnvCfg

_LEFT_FOOT_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/pelvis/left_hip_pitch_link/left_hip_roll_link/"
    "left_hip_yaw_link/left_knee_link/left_ankle_roll_link/left_ankle_pitch_link"
)
_RIGHT_FOOT_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/pelvis/right_hip_pitch_link/right_hip_roll_link/"
    "right_hip_yaw_link/right_knee_link/right_ankle_roll_link/right_ankle_pitch_link"
)


@configclass
class H2LowerVelocityHistoryEnvCfg(G1LowerVelocityHistoryEnvCfg):
    """G1 velocity-history training with H2 morphology overrides."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = unitree_h2.H2_DELAYED_DC_MOTOR.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_measurement_sensor.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/pelvis"
        self.scene.height_measurement_sensor_left_foot.prim_path = _LEFT_FOOT_PATH
        self.scene.height_measurement_sensor_right_foot.prim_path = _RIGHT_FOOT_PATH

        self.actions.random_pos.joint_names_exclude = unitree_h2.LEG_JOINT_NAMES + unitree_h2.WAIST_JOINT_NAMES

        feet = unitree_h2.FEET_LINK_NAMES
        self.rewards.base_height.params["target_height"] = unitree_h2.DEFAULT_PELVIS_HEIGHT
        self.rewards.feet_slip.params["sensor_cfg"].body_names = feet
        self.rewards.feet_slip.params["robot_cfg"].body_names = feet
        self.rewards.feet_roll.params["asset_cfg"].body_names = feet
        self.rewards.feet_yaw_diff.params["asset_cfg"].body_names = feet
        self.rewards.feet_yaw_mean.params["feet_asset_cfg"].body_names = feet
        self.rewards.feet_distance.params["asset_cfg"].body_names = feet
        self.rewards.jumping.params["sensor_cfg"].body_names = feet
        self.terminations.illegal_base_height.params["height_threshold"] = unitree_h2.DEFAULT_PELVIS_HEIGHT - 0.3

        ankle_pitch = [".*ankle_pitch_link.*"]
        extremities = self.events.apply_external_force_torque_extremities.params["asset_cfg"]
        extremities.body_names = [".*wrist_yaw_link.*", *ankle_pitch]
        self.events.reset_base.params["asset_cfg"].body_names = ankle_pitch
