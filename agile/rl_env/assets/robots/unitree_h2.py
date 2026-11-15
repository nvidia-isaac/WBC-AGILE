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

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from agile.rl_env.mdp.actuators.actuators_cfg import DelayedDCMotorCfg

MAX_DELAY_PHY_STEPS = 4
MIN_DELAY_PHY_STEPS = 0

H2_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.1/Isaac/Robots/Unitree/H2plus/unitree_h2.usda"
)


@configclass
class NestedRigidBodyArticulationCfg(ArticulationCfg):
    """Articulation config that enables contact reports on nested rigid bodies."""

    def _post_spawn(self, stage) -> None:
        super()._post_spawn(stage)
        if self.spawn is None or not self.spawn.activate_contact_sensors:
            return

        author_path = self.spawn.spawn_path if self.spawn.spawn_path is not None else self.prim_path
        for root in sim_utils.find_matching_prims(author_path, stage=stage):
            rigid_bodies = sim_utils.get_all_matching_child_prims(
                root.GetPath(),
                predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
                stage=stage,
            )
            for rigid_body in rigid_bodies:
                sim_utils.activate_contact_sensors(
                    rigid_body.GetPath().pathString,
                    stage=stage,
                )


LEG_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]

ANKLE_JOINT_NAMES = [
    ".*_ankle_roll_joint",
    ".*_ankle_pitch_joint",
]

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

ARM_JOINT_NAMES = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]

HEAD_JOINT_NAMES = [
    "head_pitch_joint",
    "head_yaw_joint",
]

# Ankle pitch is the terminal rigid body in each H2 leg chain.
FEET_LINK_NAMES = [
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
]

# Full leg-chain Z extent from URDF (legs straight): ~0.956 m.
# G1 stands at 0.72 m with a 0.752 m fully-extended leg chain (ratio 0.957).
# Applying that ratio to H2 gives 0.915 m.
DEFAULT_PELVIS_HEIGHT = 0.915

H2_DELAYED_DC_MOTOR = NestedRigidBodyArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=H2_USD_PATH,
        variants={
            "Physics": "physx",
            "collision_profile": "feet_only",
            "hands": "none",
        },
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
        pos=(0.0, 0.0, DEFAULT_PELVIS_HEIGHT + 0.1),
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
                ".*_hip_yaw_joint": 360.0,
                ".*_hip_roll_joint": 360.0,
                ".*_hip_pitch_joint": 360.0,
                ".*_knee_joint": 360.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 8.0,
            },
            armature={
                ".*_hip_.*": 0.02,
                ".*_knee_joint": 0.02,
            },
            saturation_effort=360.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "feet": DelayedDCMotorCfg(
            joint_names_expr=ANKLE_JOINT_NAMES,
            effort_limit_sim={
                ".*_ankle_pitch_joint": 66.88,
                ".*_ankle_roll_joint": 19.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 28.61,
                ".*_ankle_roll_joint": 100.7,
            },
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 1.0,
                ".*_ankle_roll_joint": 0.5,
            },
            armature=0.02,
            saturation_effort=70.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "waist": DelayedDCMotorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim={
                "waist_yaw_joint": 120.0,
                "waist_roll_joint": 180.0,
                "waist_pitch_joint": 180.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 28.375,
                "waist_roll_joint": 28.375,
                "waist_pitch_joint": 28.375,
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
            saturation_effort=180.0,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "arms": DelayedDCMotorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 120.0,
                ".*_shoulder_roll_joint": 54.0,
                ".*_shoulder_yaw_joint": 54.0,
                ".*_elbow_joint": 54.0,
                ".*_wrist_roll_joint": 54.0,
                ".*_wrist_pitch_joint": 25.0,
                ".*_wrist_yaw_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 28.375,
                ".*_shoulder_roll_joint": 34.3,
                ".*_shoulder_yaw_joint": 34.3,
                ".*_elbow_joint": 34.3,
                ".*_wrist_roll_joint": 34.3,
                ".*_wrist_pitch_joint": 50.125,
                ".*_wrist_yaw_joint": 50.125,
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
            saturation_effort=120.0,
            min_delay=0,
            max_delay=0,
        ),
        "head": DelayedDCMotorCfg(
            joint_names_expr=HEAD_JOINT_NAMES,
            effort_limit_sim={
                "head_pitch_joint": 50.0,
                "head_yaw_joint": 50.0,
            },
            velocity_limit_sim={
                "head_pitch_joint": 10.0,
                "head_yaw_joint": 10.0,
            },
            stiffness={
                "head_pitch_joint": 100.0,
                "head_yaw_joint": 100.0,
            },
            damping={
                "head_pitch_joint": 5.0,
                "head_yaw_joint": 5.0,
            },
            armature=0.01,
            saturation_effort=50.0,
            min_delay=0,
            max_delay=0,
        ),
    },
)
