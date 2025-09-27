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
import pathlib

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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from agile.rl_env import mdp
from agile.rl_env.assets.robots import booster_t1
from agile.rl_env.mdp.terrains import STAND_UP_ROUGH_TERRAIN_CFG  # noqa: F401, F403

FILE_DIR = pathlib.Path(__file__).parent
REPO_DIR = FILE_DIR.parent.parent.parent

REST_DURATION_S = 2.0

from_scratch = 1.0
with_curriculum = 1.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAND_UP_ROUGH_TERRAIN_CFG,
        max_init_terrain_level=0,
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

    # robots
    robot = booster_t1.T1_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # lights
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


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # No commands for this task


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyObservationCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # The controlled joints are defined in the configs post init
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = False
            self.flatten_history_dim = False

    @configclass
    class CriticObservationsCfg(ObsGroup):
        """Observations for critic group."""

        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        is_env_inactive = ObsTerm(
            func=mdp.is_env_inactive,
            params={"rest_duration_s": REST_DURATION_S},
        )
        contact_forces = ObsTerm(
            func=mdp.contact_force_norm,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            },
            scale=5e-3,
            clip=(-25_000.0, 25_000.0),
        )
        base_height = ObsTerm(
            func=mdp.base_height_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("height_measurement_sensor")},
            clip=(-2, 2),
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = False
            self.flatten_history_dim = False

    policy: PolicyObservationCfg = PolicyObservationCfg()
    critic: CriticObservationsCfg = CriticObservationsCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.1,
        clip={".*": (-1.0, 1.0)},
        use_zero_offset=True,
        preserve_order=True,
    )

    lift = mdp.LiftActionCfg(
        asset_name="robot",
        link_to_lift="H2",  # Head
        stiffness_forces=5000.0,
        damping_forces=500.0,
        force_limit=300.0,
        height_sensor="height_measurement_sensor",
        target_height=booster_t1.DEFAULT_TRUNK_HEIGHT,
        start_lifting_time_s=3.0,
        lifting_duration_s=10.0,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Regularization:
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.01)
    joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.01, params={"soft_ratio": 0.8})
    action_rate = RewTerm(
        func=mdp.action_rate_l2_if_actor_active,
        weight=-0.01,
        params={"rest_duration_s": REST_DURATION_S},
    )
    action_rate_rate = RewTerm(
        func=mdp.action_rate_rate_l2_if_actor_is_active,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot"), "rest_duration_s": REST_DURATION_S},
    )
    action_l2 = RewTerm(func=mdp.action_l2_if_actor_active, weight=-0.05, params={"rest_duration_s": REST_DURATION_S})

    incoming_forces_penalty = RewTerm(
        func=mdp.max_incoming_forces_penalty,
        weight=-5e-7,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )

    # Task:
    base_height_rough = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_medium = RewTerm(
        func=mdp.base_height_exp,
        weight=8.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.25,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_fine = RewTerm(
        func=mdp.base_height_exp,
        weight=16.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_exp_if_standing,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "std": 0.1,
        },
    )

    orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"])},
    )

    not_moving = RewTerm(
        func=mdp.moving_if_standing,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "weight_lin": 1.0,
            "weight_ang": 1.0,
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    # Aesthetics
    equal_foot_force = RewTerm(
        func=mdp.equal_foot_force_if_standing,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "height_measurement_sensor": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    illegal_contacts = RewTerm(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=booster_t1.UNDESIRED_CONTACTS_LINKS),
            "threshold": 1.0,
        },
    )

    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref_if_standing,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=booster_t1.FEET_LINK_NAMES),
            "ref_distance": 0.2,  # 20cm lateral distance between feet
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
        },
    )

    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base_if_standing,
        weight=-5.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link.*"),
            "base_body_cfg": SceneEntityCfg("robot", body_names="Waist"),
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
        },
    )

    root_acc = RewTerm(
        func=mdp.root_acc_l2,  # type: ignore
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Termination
    stand_up_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=10.0,
        params={"term_keys": "standing"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    standing = DoneTerm(
        func=mdp.standing,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "duration_s": 5.0,
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
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

    # reset

    disable_robot_joint_actions = EventTerm(
        func=mdp.disable_joints,
        mode="pre_sim_step",
        params={"rest_duration_s": REST_DURATION_S},
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
        func=mdp.reset_root_state_uniform_some_standing,
        mode="reset",
        params={
            "standing_ratio": 0.1,
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "yaw": (-3.14, 3.14),
                "roll": (-math.radians(20), math.radians(20)),
                "pitch": (-math.radians(20), math.radians(20)),
            },
            "velocity_range": {
                "x": (-5.0, 5.0),
                "y": (-5.0, 5.0),
                "z": (-0.0, 0.0),
                "roll": (-5.0, 5.0),
                "pitch": (-5.0, 5.0),
                "yaw": (-5.0, 5.0),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot_link.*"]),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 2.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "roll": (-0.25, 0.25),
                "y": (-0.5, 0.5),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            }
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_successful_termination,
        params={
            "successful_termination_term": "standing",
            "n_successes": 5,
            "n_failures": 5,
        },
    )

    remove_lift = CurrTerm(
        func=mdp.remove_harness,
        params={
            "harness_action_name": "lift",
            "start": 5_000 * from_scratch,
            "num_steps": 200_000 * from_scratch,
            "linear": False,
        },
    )

    action_limit_successful_termination = CurrTerm(
        func=mdp.action_limit_successful_termination,
        params={
            "successful_termination_term": "standing",
            "activate_after_steps": 100_000 * from_scratch,
            "action_name": "joint_pos",
            "update_rate": 0.001,
            "move_up_ratio": 0.95,
            "move_down_ratio": 0.8,
            "max_action_limit": 1.0,
        },
    )

    # rewards
    increase_action_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_l2",
            "start_step": 70_000 * from_scratch,
            "num_steps": 150_000 * with_curriculum,
            "terminal_weight": -0.25,
            "use_log_space": True,
        },
    )

    increase_action_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate",
            "start_step": 100_000 * from_scratch,
            "num_steps": 150_000 * with_curriculum,
            "terminal_weight": -0.1,
            "use_log_space": True,
        },
    )

    increase_action_rate_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate_rate",
            "start_step": 150_000 * from_scratch,
            "num_steps": 150_000,
            "terminal_weight": -0.1,
            "use_log_space": False,
        },
    )

    increase_joint_deviation_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "joint_deviation_l1",
            "start_step": 100_000 * from_scratch,
            "num_steps": 150_000 * with_curriculum,
            "terminal_weight": 10.0,
            "use_log_space": False,
        },
    )

    increase_incoming_forces_penalty = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "incoming_forces_penalty",
            "start_step": 120_000 * from_scratch,
            "num_steps": 150_000 * with_curriculum,
            "terminal_weight": -1e-5,
            "use_log_space": True,
        },
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, -6.0, 3.0)

    lookat: tuple[float, float, float] = (0.0, 25.0, -5.0)

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
class T1StandUpEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 500
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt

        if self.scene.height_measurement_sensor is not None:
            self.scene.height_measurement_sensor.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
