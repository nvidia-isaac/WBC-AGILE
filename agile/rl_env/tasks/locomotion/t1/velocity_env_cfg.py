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
from agile.rl_env.assets.robots import booster_t1
from agile.rl_env.mdp.terrains import LESS_ROUGH_TERRAIN_CFG


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
            restitution=1.0,
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

    robot = booster_t1.T1_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )

    height_measurement_sensor_left_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.01, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )
    height_measurement_sensor_right_foot = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.01, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformNullVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformNullVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=booster_t1.LEG_JOINT_NAMES,
        scale=1.0,
        use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )

    random_pos = mdp.RandomActionCfg(
        asset_name="robot",
        joint_names_exclude=booster_t1.LEG_JOINT_NAMES + booster_t1.WAIST_JOINT_NAMES,
        sample_range=(0.1, 2.5),
        preserve_order=True,
        velocity_profile_cfg=mdp.TrapezoidalVelocityProfileCfg(
            acceleration_range=(1.0, 20.0),
            max_velocity_range=(10.0, 20.0),
            min_cruise_ratio=0.1,
            synchronize_joints=True,
            time_scaling_method="max_time",
            use_smooth_start=False,
            position_tolerance=0.001,
            velocity_tolerance=0.01,
            enable_position_limits=True,
            enable_velocity_limits=True,
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class HistoryPolicyCfg(ObsGroup):
        """Observations for policy group with history."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        controlled_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=booster_t1.LEG_JOINT_NAMES)},
        )
        controlled_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=booster_t1.LEG_JOINT_NAMES)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = False
            self.flatten_history_dim = False

    @configclass
    class PrivilegedVelocityCriticCfg(ObsGroup):
        """Observations for policy group."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: HistoryPolicyCfg = HistoryPolicyCfg()
    critic: PrivilegedVelocityCriticCfg = PrivilegedVelocityCriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )

    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Waist"]),
        },
    )

    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"])},
    )

    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=booster_t1.LEG_JOINT_NAMES)},
    )

    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Ankle.*")},
    )

    ankle_roll_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Ankle_Roll")},
    )

    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    action_rate_rate = RewTerm(
        func=mdp.action_rate_rate_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "soft_ratio": 0.9},
    )

    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Trunk"],
            ),
            "threshold": 1.0,
        },
    )

    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.1,
        params={
            "contact_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link.*"),
            "robot_cfg": SceneEntityCfg("robot", body_names=".*foot_link.*"),
        },
    )

    feet_roll = RewTerm(
        func=mdp.feet_roll_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link.*")},
    )

    feet_yaw_diff = RewTerm(
        func=mdp.feet_yaw_diff_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link.*")},
    )

    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base,
        weight=-4.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link.*"),
            "base_body_cfg": SceneEntityCfg("robot", body_names="Waist"),
        },
    )

    root_acc = RewTerm(
        func=mdp.root_acc_l2,  # type: ignore
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=booster_t1.FEET_LINK_NAMES),
            "ref_distance": 0.2,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "limit_angle": math.radians(30.0),
        },
    )

    illegal_contacts = DoneTerm(
        func=mdp.illegal_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Trunk", "Waist"],
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Trunk", "Waist"]),
            "threshold": 20.0,
            "min_height": 0.45,
        },
    )

    illegal_base_height = DoneTerm(
        func=mdp.illegal_base_height,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT - 0.3,
        },
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, -5.0, 2.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type = "asset_root"
    asset_name: str = "robot"
    env_index: int = 0


@configclass
class LocomotionEventCfg:
    """Configuration for events."""

    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 2.0),
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
            "armature_distribution_params": (0.0, 2.0),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.05, 0.05), "z": (-0.15, 0.15)},
        },
    )

    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*hand_link.*", ".*foot_link.*"]),
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
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot_link.*"]),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (-1.0, 1.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
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
            "n_successes": 4,
            "n_failures": 10,
            "p_random_move_up": 0.00,
            "p_random_move_down": 0.00,
        },
    )

    increase_action_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate",
            "start_step": 50_000,
            "num_steps": 100_000,
            "terminal_weight": -2.0,
            "use_log_space": False,
        },
    )

    increase_action_rate_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate_rate",
            "start_step": 60_000,
            "num_steps": 100_000,
            "terminal_weight": -1.0,
            "use_log_space": False,
        },
    )


@configclass
class T1LowerVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the T1 velocity tracking environment."""

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
        self.controller_freq = 50.0
        self.physics_freq = 200.0
        self.episode_length_s = 30.0
        self.max_episode_length_offset_s = 0.0

        self.decimation = int(self.physics_freq / self.controller_freq)
        self.sim.dt = 1.0 / self.physics_freq
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.solver_type = 1

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_measurement_sensor is not None:
            self.scene.height_measurement_sensor.update_period = self.sim.dt
        if self.scene.height_measurement_sensor_left_foot is not None:
            self.scene.height_measurement_sensor_left_foot.update_period = self.sim.dt
        if self.scene.height_measurement_sensor_right_foot is not None:
            self.scene.height_measurement_sensor_right_foot.update_period = self.sim.dt

        self.only_positive_rewards = False

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
