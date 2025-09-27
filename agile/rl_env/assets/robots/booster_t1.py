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


import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from agile.rl_env.mdp.actuators import DelayedDCMotorCfg

T1_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd"

LEG_JOINT_NAMES = [
    ".*Hip.*",
    ".*Knee.*",
    ".*Ankle.*",
]
WAIST_JOINT_NAMES = [
    "Waist",
]
ARM_JOINT_NAMES = [
    ".*Shoulder.*",
    ".*Elbow.*",
    ".*Wrist.*",
]

HEAD_JOINT_NAMES = [
    ".*Head.*",
]

HAND_JOINT_NAMES = [
    ".*Hand.*",
]
FEET_LINK_NAMES = [
    ".*foot_link.*",
]
DEFAULT_TRUNK_HEIGHT = 0.65

p_gain_scale = 0.4
d_gain_scale = 1.2

UNDESIRED_CONTACTS_LINKS = [
    "Trunk",
    "H1",
    "H2",
    "AL.*",
    "AR.*",
    ".*hand.*",
    "Waist",
    "Hip.*",
]


MIN_DELAY_STEPS = 0
MAX_DELAY_STEPS = 8

"""Configuration for the Booster T1 robot with delayed DC motor actuators."""
T1_DELAYED_DC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=T1_USD_PATH,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    articulation_root_prim_path="/Trunk",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
            "Left_Hip_Pitch": -0.2,
            "Right_Hip_Pitch": -0.2,
            "Left_Knee_Pitch": 0.4,
            "Right_Knee_Pitch": 0.4,
            "Left_Ankle_Pitch": -0.2,
            "Right_Ankle_Pitch": -0.2,
            "Left_Elbow_Yaw": -0.3,
            "Right_Elbow_Yaw": 0.3,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all": DelayedDCMotorCfg(
            max_delay=MAX_DELAY_STEPS,
            min_delay=MIN_DELAY_STEPS,
            saturation_effort=130.0,
            joint_names_expr=[".*"],
            stiffness={
                ".*Head.*": 20.0,
                ".*Shoulder.*": 20.0,
                ".*Elbow.*": 20.0,
                ".*Waist.*": 100.0,
                ".*Hip.*": 100.0,
                ".*Knee.*": 100.0,
                ".*Ankle.*": 20.0,
            },
            damping={
                ".*Head.*": 0.2,
                ".*Shoulder.*": 0.5,
                ".*Elbow.*": 0.5,
                ".*Waist.*": 2.5,
                ".*Hip.*": 2.5,
                ".*Knee.*": 2.5,
                ".*Ankle.*": 1.0,
            },
            velocity_limit_sim={
                ".*Head.*": 30.0,
                ".*Shoulder.*": 30.0,
                ".*Elbow.*": 30.0,
                ".*Waist.*": 32.0,
                ".*Hip.*": 32.0,
                ".*Knee.*": 20.0,
                ".*Ankle.*": 32.0,
            },
            friction=0.01,
            armature=0.02,
            effort_limit_sim={
                "AAHead_yaw": 7,
                "Head_pitch": 7,
                "Left_Shoulder_Pitch": 18.0,
                "Left_Shoulder_Roll": 18.0,
                "Left_Elbow_Pitch": 18.0,
                "Left_Elbow_Yaw": 18.0,
                "Right_Shoulder_Pitch": 18.0,
                "Right_Shoulder_Roll": 18.0,
                "Right_Elbow_Pitch": 18.0,
                "Right_Elbow_Yaw": 18.0,
                "Waist": 30.0,
                "Left_Hip_Pitch": 45.0,
                "Left_Hip_Roll": 30.0,
                "Left_Hip_Yaw": 30.0,
                "Left_Knee_Pitch": 60.0,
                "Left_Ankle_Pitch": 24.0,
                "Left_Ankle_Roll": 15.0,
                "Right_Hip_Pitch": 45.0,
                "Right_Hip_Roll": 30.0,
                "Right_Hip_Yaw": 30.0,
                "Right_Knee_Pitch": 60.0,
                "Right_Ankle_Pitch": 24.0,
                "Right_Ankle_Roll": 15.0,
            },
        ),
    },  # type: ignore
)
