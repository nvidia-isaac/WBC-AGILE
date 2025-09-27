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


import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

from agile.rl_env import mdp
from agile.rl_env.assets.robots import unitree_g1
from agile.rl_env.mdp.terrains import LESS_ROUGH_TERRAIN_CFG, ROUGH_TERRAIN_CFG  # noqa: F401, F403

##
# Scene definition
##


@configclass
class PrivilegedVelocityPolicyCfg(ObsGroup):
    """Observations for policy group."""

    velocity_height_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class PrivilegedVelocityCriticCfg(ObsGroup):
    """Observations for critic group."""

    velocity_height_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
    )

    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
    actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
    base_height = ObsTerm(func=mdp.base_height_from_command, params={"command_name": "base_velocity"}, clip=(-10, 10))

    # privileged observations
    height_scan_feet = ObsTerm(
        func=mdp.height_scan_feet,
        params={
            "sensor_cfg_left": SceneEntityCfg("height_scanner_left_foot"),
            "sensor_cfg_right": SceneEntityCfg("height_scanner_right_foot"),
        },
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class StudentVelocityPolicyCfg(ObsGroup):
    """Observations for student policy group."""

    velocity_height_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class TeacherRecurrentStudentObservationsCfg:
    """Observation specifications for the MDP."""

    # the policy is the student policy
    policy: StudentVelocityPolicyCfg = StudentVelocityPolicyCfg()
    # the teacher gets is the privileged observations
    teacher: PrivilegedVelocityPolicyCfg = PrivilegedVelocityPolicyCfg()

    def __post_init__(self):
        self.policy.enable_corruption = True
        self.teacher.enable_corruption = True


@configclass
class TeacherHistoryStudentObservationsCfg:
    """Observation specifications for the MDP."""

    # the policy is the student policy
    policy: StudentVelocityPolicyCfg = StudentVelocityPolicyCfg()
    # the teacher gets is the privileged observations
    teacher: PrivilegedVelocityPolicyCfg = PrivilegedVelocityPolicyCfg()

    def __post_init__(self):
        self.policy.enable_corruption = True
        self.teacher.enable_corruption = True

        self.policy.history_length = 5
        self.policy.concatenate_terms = False
        self.policy.flatten_history_dim = False


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=LESS_ROUGH_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                f"TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot = unitree_g1.G1_29DOF.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_left_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.3, 0.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_right_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.3, 0.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityBaseHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.5, 10.0),
        rel_standing_envs=0.20,
        heading_command=False,
        debug_vis=True,
        default_height=unitree_g1.DEFAULT_PELVIS_HEIGHT,
        ema_smoothing_param=0.5,
        ranges=mdp.UniformVelocityBaseHeightCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
            base_height=(0.4, unitree_g1.DEFAULT_PELVIS_HEIGHT),
        ),
        min_walk_height=0.5,
        random_height_during_walking=False,
        height_sensor="height_measurement_sensor",
        root_name="pelvis",
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # For lower body policy.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=unitree_g1.LEG_JOINT_NAMES,
        scale=unitree_g1.G1_ACTION_SCALE_LOWER,
        use_default_offset=True,
        clip={".*": (-6.0, 6.0)},
    )

    # We don't want to randomize the roll and pitch joints of the waist.
    random_upper_body_pos = mdp.RandomActionCfg(
        asset_name="robot",
        joint_names_exclude=unitree_g1.LEG_JOINT_NAMES,
        joint_pos_limits={
            "waist_roll_joint": (-0.0, 0.0),
            "waist_pitch_joint": (-0.0, 0.0),
            "waist_yaw_joint": (-math.radians(15.0), math.radians(15.0)),
        },
        sample_range=(0.1, 2.0),
        velocity_profile_cfg=mdp.TrapezoidalVelocityProfileCfg(
            acceleration_range=(1.0, 10.0),  # Acceleration in rad/s²
            max_velocity_range=(5.0, 10.0),  # Max velocity in rad/s
            min_cruise_ratio=0.1,  # Minimum 10% of trajectory at cruise velocity
            synchronize_joints=True,  # All joints finish together
            time_scaling_method="max_time",  # Use slowest joint's time
            use_smooth_start=False,  # Don't inherit velocity from previous trajectory
            position_tolerance=0.001,
            velocity_tolerance=0.01,
            enable_position_limits=True,
            enable_velocity_limits=True,
        ),
        preserve_order=True,
        no_random_when_walking=True,
        command_name="base_velocity",
    )

    harness = mdp.HarnessActionCfg(
        asset_name="robot",
        root_name="pelvis",
        stiffness_torques=300.0,
        damping_torques=50.0,
        stiffness_forces=1000.0,
        damping_forces=100.0,
        force_limit=150.0,
        torque_limit=500.0,
        height_sensor="height_measurement_sensor",
        target_height=unitree_g1.DEFAULT_PELVIS_HEIGHT,
        command_name="base_velocity",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: PrivilegedVelocityPolicyCfg = PrivilegedVelocityPolicyCfg()
    critic: PrivilegedVelocityCriticCfg = PrivilegedVelocityCriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp_weighted,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_base_height_exp_smooth = RewTerm(
        func=mdp.track_base_height_exp_smooth,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    no_undersired_base_velocity_exp = RewTerm(
        func=mdp.no_undersired_base_velocity_exp_if_null_cmd,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1},
    )
    equal_foot_force_if_null_cmd = RewTerm(
        func=mdp.equal_foot_force_if_null_cmd,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )
    stand_with_both_feet_if_null_cmd = RewTerm(
        func=mdp.stand_with_both_feet_if_null_cmd,
        weight=1.0,
        params={
            "threshold": 1.0,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )

    # -- penalties
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.LEG_JOINT_NAMES)},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.LEG_JOINT_NAMES)},
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.LEG_JOINT_NAMES)},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.LEG_JOINT_NAMES)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.LEG_JOINT_NAMES)},
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate_rate = RewTerm(
        func=mdp.action_rate_rate_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    root_acc = RewTerm(
        func=mdp.root_acc_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ankle_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle_.*_joint")},
    )

    # Stylistic rewards
    relax_if_null_cmd = RewTerm(
        func=mdp.relax_if_null_cmd_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.LEG_JOINT_NAMES),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )

    feet_roll = RewTerm(
        func=mdp.feet_roll_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link.*")},
    )

    feet_yaw_diff = RewTerm(
        func=mdp.feet_yaw_diff_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link.*")},
    )

    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base,
        weight=-1.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link.*"),
            "base_body_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    jumping_penalty = RewTerm(
        func=mdp.jumping,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.1,
        },
    )

    impact_velocity = RewTerm(
        func=mdp.impact_velocity_l1,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "force_threshold": 10.0,
        },
    )

    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )

    # Penalize hip joint deviation.
    hip_pos_pen = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                ],
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "limit_angle": math.radians(40.0),
        },
    )

    illegal_contacts = DoneTerm(
        func=mdp.illegal_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                    "pelvis",
                    ".*_hip_.*_link",
                    ".*_knee_link",
                ],
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "threshold": 50.0,
            "min_height": 0.5,
        },
    )

    feet_distance = DoneTerm(
        func=mdp.link_distance,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "min_distance_threshold": 0.1,
            "max_distance_threshold": 0.5,
        },
    )

    knee_distance = DoneTerm(
        func=mdp.link_distance,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*knee_link"),
            "min_distance_threshold": 0.2,
        },
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, -25.0, 12.0)

    lookat: tuple[float, float, float] = (0.0, 0.0, 3.0)

    cam_prim_path: str = "/OmniverseKit_Persp"

    resolution: tuple[int, int] = (1280, 720)

    origin_type = "asset_root"
    """Available options are:

    * ``"world"``: The origin of the world.
    * ``"env"``: The origin of the environment defined by :attr:`env_index`.
    * ``"asset_root"``: The center of the asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    * ``"asset_body"``: The center of the body defined by :attr:`body_name` in asset defined by
                        :attr:`asset_name` in environment :attr:`env_index`.
    """

    asset_name: str = "robot"

    env_index: int = 0


@configclass
class LocomotionEventCfg:
    """Configuration for events."""

    # startup
    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    randomize_lower_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.LEG_JOINT_NAMES),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_upper_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=unitree_g1.ARM_JOINT_NAMES + unitree_g1.WAIST_JOINT_NAMES),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.0, 0.005),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "armature_distribution_params": (0.4, 1.6),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_bodies_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_bodies_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.05, 0.05), "z": (-0.15, 0.15)},
        },
    )

    # reset
    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-2.0, 2.0),
        },
    )

    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(2.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*wrist_yaw_link.*"]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-2.5, 2.5),
                "y": (-2.5, 2.5),
                "z": (-0.0, 0.0),
                "yaw": (-3.14, 3.14),
                "roll": (-math.radians(10), math.radians(10)),
                "pitch": (-math.radians(10), math.radians(10)),
            },
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 2.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            }
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_curriculum,
        params={
            "command_name": "base_velocity",
            "move_up_distance": 4.0,
            "move_down_distance": 2.0,
            "n_successes": 5,
            "n_failures": 10,
            "p_random_move_up": 0.0,
            "p_random_move_down": 0.0,
        },
    )

    remove_harness = CurrTerm(
        func=mdp.remove_harness,
        params={
            "harness_action_name": "harness",
            "start": 0,
            "num_steps": 50_000,
        },
    )

    increase_action_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate",
            "start_step": 75_000,
            "num_steps": 50_000,
            "terminal_weight": -1.0,
            "use_log_space": False,
        },
    )


@configclass
class G1LowerVelocityHeightEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 velocity tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: LocomotionEventCfg = LocomotionEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.controller_freq = 50.0
        self.physics_freq = 200.0
        self.episode_length_s = 30.0
        self.max_episode_length_offset_s = 0.0

        # simulation settings
        self.decimation = int(self.physics_freq / self.controller_freq)  # Should be 10
        self.sim.dt = 1.0 / self.physics_freq  # Should be 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.scene, "height_scanner", None) is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.scene, "height_scanner_left_foot", None) is not None:
            self.scene.height_scanner_left_foot.update_period = self.decimation * self.sim.dt

        if getattr(self.scene, "height_scanner_right_foot", None) is not None:
            self.scene.height_scanner_right_foot.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def eval(self):
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.viewer.eye = (-2.5, -5.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.viewer.origin_type = "world"
        self.rewards = None
        self.curriculum = None
        self.observations.policy.concatenate_terms = True
        self.observations.policy.flatten_history_dim = True
        self.observations.policy.enable_corruption = False
        self.observations.critic.concatenate_terms = True
        # Add evaluation observations
        self.observations.eval = mdp.EvaluationObservationsCfg()

        if hasattr(self.actions, "harness"):
            del self.actions.harness


@configclass
class G1VelocityHeightRecurrentStudentEnvCfg(G1LowerVelocityHeightEnvCfg):
    """Configuration for the student distillation locomotion velocity-tracking environment."""

    # Scene settings
    observations: TeacherRecurrentStudentObservationsCfg = TeacherRecurrentStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # disable harness immediately
        if hasattr(self.curriculum, "remove_harness"):
            del self.curriculum.remove_harness

        if hasattr(self.actions, "harness"):
            del self.actions.harness

        self.observations.policy.enable_corruption = True
        self.observations.policy.concatenate_terms = False
        self.observations.teacher.concatenate_terms = False

    def eval(self):
        # Add evaluation observations
        self.observations.eval = mdp.EvaluationObservationsCfg()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.viewer.eye = (-2.5, -5.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.viewer.origin_type = "world"
        self.events = None
        self.rewards = None
        self.curriculum = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.flatten_history_dim = True
        self.observations.policy.concatenate_terms = True
        self.observations.teacher.concatenate_terms = True


@configclass
class G1VelocityHeightHistoryStudentEnvCfg(G1VelocityHeightRecurrentStudentEnvCfg):
    """Configuration for the student distillation locomotion velocity-tracking environment."""

    # Scene settings
    observations: TeacherHistoryStudentObservationsCfg = TeacherHistoryStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Don't concatenate observations if mirror loss will be used
        # Mirror loss requires structured (dictionary) observations
        self.observations.policy.concatenate_terms = False
        self.observations.policy.flatten_history_dim = False
