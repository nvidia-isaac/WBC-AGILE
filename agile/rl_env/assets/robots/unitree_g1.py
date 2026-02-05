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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from agile.rl_env.mdp.actuators.actuators_cfg import DelayedDCMotorCfg, DelayedImplicitActuatorCfg

MAX_DELAY_PHY_STEPS = 4
MIN_DELAY_PHY_STEPS = 0

G1_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"

LEG_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]
NO_LEG_JOINT_NAMES = [
    "waist_.*_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
    ".*_hand_.*_joint",
]
ANKLE_JOINT_NAMES = [
    ".*_ankle_.*_joint",
]
FEET_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
WAIST_JOINT_NAMES = [
    "waist_.*_joint",
]
ARM_JOINT_NAMES = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]
HAND_JOINT_NAMES = [
    ".*_hand_.*",
]
NO_HAND_JOINT_NAMES = [
    ".*_hip_.*_joint",
    "waist_.*_joint",
    ".*_knee_joint",
    ".*_shoulder_.*_joint",
    ".*_ankle_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]
RIGHT_HAND_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_.*_joint",
]
LEFT_HAND_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_.*_joint",
]

DEFAULT_PELVIS_HEIGHT = 0.72

# Using the delayed DC motor model.
G1_29DOF_DELAYED_DC_MOTOR = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        # Disable hands to accelerate training.
        variants={"Physics": "PhysX", "left_hand": "None", "right_hand": "None"},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.02,
                ".*_knee_joint": 0.02,
            },
            saturation_effort=180.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "feet": DelayedDCMotorCfg(
            joint_names_expr=ANKLE_JOINT_NAMES,
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit_sim={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.02,
            saturation_effort=80.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "waist": DelayedDCMotorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 300.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 300.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.02,
            saturation_effort=120.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "arms": DelayedDCMotorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 90.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 60.0,
                ".*_wrist_.*_joint": 4.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 0.4,
                ".*_elbow_joint": 1.0,
                ".*_wrist_.*_joint": 0.2,
            },
            armature={
                ".*_shoulder_.*": 0.02,
                ".*_elbow_.*": 0.02,
                ".*_wrist_.*_joint": 0.02,
            },
            saturation_effort=40.0,
            min_delay=0,
            max_delay=0,
        ),
    },
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


G1_29DOF = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        # Disable hands to accelerate training.
        variants={"Physics": "PhysX", "left_hand": "None", "right_hand": "None"},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "feet": DelayedImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            joint_names_expr=ANKLE_JOINT_NAMES,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 300.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 300.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.03,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 90.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 60.0,
                ".*_wrist_.*_joint": 4.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 0.4,
                ".*_elbow_joint": 1.0,
                ".*_wrist_.*_joint": 0.2,
            },
            armature={
                ".*_shoulder_.*": 0.03,
                ".*_elbow_.*": 0.03,
                ".*_wrist_.*_joint": 0.03,
            },
        ),
    },
)

G1_ACTION_SCALE_LOWER = {}
for actuator_name, actuator_cfg in G1_29DOF.actuators.items():
    if actuator_name != "legs" and actuator_name != "feet":
        continue
    e = actuator_cfg.effort_limit_sim
    s = actuator_cfg.stiffness
    names = actuator_cfg.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            G1_ACTION_SCALE_LOWER[n] = 0.25 * e[n] / s[n]


# =============================================================================
# G1 with Dex3-1 Hands Configuration
# =============================================================================

# Dex3-1 hand motor constants
# Manual: https://marketing.unitree.com/article/en/Dex3-1/User_Manual.html
ARMATURE_1515 = 0.00149
STIFFNESS_1515 = 2.0
DAMPING_1515 = 0.2
EFFORT_LIMIT_1515 = 0.76
VELOCITY_LIMIT_1515 = 23.0

G1_W_HANDS_AGILE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
            min_delay=0,
            max_delay=0,
        ),
        "feet": DelayedImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=ANKLE_JOINT_NAMES,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            min_delay=0,
            max_delay=0,
        ),
        "waist": DelayedImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 300.0,
                "waist_roll_joint": 300.0,
                "waist_pitch_joint": 300.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.03,
            min_delay=0,
            max_delay=0,
        ),
        "left_arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            effort_limit_sim={
                "left_shoulder_pitch_joint": 25.0,
                "left_shoulder_roll_joint": 25.0,
                "left_shoulder_yaw_joint": 25.0,
                "left_elbow_joint": 25.0,
                "left_wrist_roll_joint": 25.0,
                "left_wrist_pitch_joint": 5.0,
                "left_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                "left_shoulder_pitch_joint": 37.0,
                "left_shoulder_roll_joint": 37.0,
                "left_shoulder_yaw_joint": 37.0,
                "left_elbow_joint": 37.0,
                "left_wrist_roll_joint": 37.0,
                "left_wrist_pitch_joint": 22.0,
                "left_wrist_yaw_joint": 22.0,
            },
            stiffness={
                "left_shoulder_pitch_joint": 100.0,
                "left_shoulder_roll_joint": 100.0,
                "left_shoulder_yaw_joint": 100.0,
                "left_elbow_joint": 100.0,
                "left_wrist_roll_joint": 100.0,
                "left_wrist_pitch_joint": 100.0,
                "left_wrist_yaw_joint": 100.0,
            },
            damping={
                "left_shoulder_pitch_joint": 2.0,
                "left_shoulder_roll_joint": 1.0,
                "left_shoulder_yaw_joint": 0.4,
                "left_elbow_joint": 1.0,
                "left_wrist_roll_joint": 0.2,
                "left_wrist_pitch_joint": 0.2,
                "left_wrist_yaw_joint": 0.2,
            },
            armature={
                "left_shoulder_.*": 0.03,
                "left_elbow_.*": 0.03,
                "left_wrist_roll_joint": 0.03,
                "left_wrist_pitch_joint": 0.03,
                "left_wrist_yaw_joint": 0.03,
            },
            min_delay=0,
            max_delay=0,
        ),
        "right_arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            effort_limit_sim={
                "right_shoulder_pitch_joint": 25.0,
                "right_shoulder_roll_joint": 25.0,
                "right_shoulder_yaw_joint": 25.0,
                "right_elbow_joint": 25.0,
                "right_wrist_roll_joint": 25.0,
                "right_wrist_pitch_joint": 5.0,
                "right_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                "right_shoulder_pitch_joint": 37.0,
                "right_shoulder_roll_joint": 37.0,
                "right_shoulder_yaw_joint": 37.0,
                "right_elbow_joint": 37.0,
                "right_wrist_roll_joint": 37.0,
                "right_wrist_pitch_joint": 22.0,
                "right_wrist_yaw_joint": 22.0,
            },
            stiffness={
                "right_shoulder_pitch_joint": 100.0,
                "right_shoulder_roll_joint": 100.0,
                "right_shoulder_yaw_joint": 100.0,
                "right_elbow_joint": 100.0,
                "right_wrist_roll_joint": 100.0,
                "right_wrist_pitch_joint": 100.0,
                "right_wrist_yaw_joint": 100.0,
            },
            damping={
                "right_shoulder_pitch_joint": 2.0,
                "right_shoulder_roll_joint": 1.0,
                "right_shoulder_yaw_joint": 0.4,
                "right_elbow_joint": 1.0,
                "right_wrist_roll_joint": 0.2,
                "right_wrist_pitch_joint": 0.2,
                "right_wrist_yaw_joint": 0.2,
            },
            armature={
                "right_shoulder_.*": 0.03,
                "right_elbow_.*": 0.03,
                "right_wrist_roll_joint": 0.03,
                "right_wrist_pitch_joint": 0.03,
                "right_wrist_yaw_joint": 0.03,
            },
            min_delay=0,
            max_delay=0,
        ),
        "hands": DelayedImplicitActuatorCfg(
            effort_limit_sim=EFFORT_LIMIT_1515,
            velocity_limit_sim=VELOCITY_LIMIT_1515,
            joint_names_expr=HAND_JOINT_NAMES,
            stiffness=STIFFNESS_1515,
            damping=DAMPING_1515,
            armature=ARMATURE_1515,
            min_delay=0,
            max_delay=0,
        ),
    },
)

G1_W_HANDS_AGILE_ACTION_SCALE = {}
for _a in G1_W_HANDS_AGILE_CFG.actuators.values():
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    _names = _a.joint_names_expr
    if not isinstance(_e, dict):
        _e = dict.fromkeys(_names, _e)
    if not isinstance(_s, dict):
        _s = dict.fromkeys(_names, _s)
    for _n in _names:
        if _n in _e and _n in _s and _s[_n]:
            G1_W_HANDS_AGILE_ACTION_SCALE[_n] = 0.25 * _e[_n] / _s[_n]
