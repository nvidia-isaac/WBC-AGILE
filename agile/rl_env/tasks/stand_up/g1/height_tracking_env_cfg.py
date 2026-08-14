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

from isaaclab_physx.physics import PhysxCfg

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from agile.rl_env import mdp
from agile.rl_env.assets.robots import unitree_g1
from agile.rl_env.mdp.terrains import STAND_UP_ROUGH_TERRAIN_G1_CFG

# Task-specific illegal contact links (not shared across robots/tasks)
ILLEGAL_CONTACTS_LINKS = [
    ".*wrist.*",
]


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAND_UP_ROUGH_TERRAIN_G1_CFG,
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
    robot = unitree_g1.G1_29DOF_HEIGHT_TRACKING.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
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

    height = mdp.SmoothHeightCommandCfg(
        asset_name="robot",
        body_name="torso_link",
        offset=mdp.SmoothHeightCommandCfg.OffsetCfg(pos=(0.0, 0.0, 0.20)),
        height_sensor="height_measurement_sensor",
        resampling_time_range=(1.0, 7.0),
        ranges=mdp.SmoothHeightCommandCfg.Ranges(height=(-0.5, unitree_g1.DEFAULT_PELVIS_HEIGHT + 0.2)),
        velocity_range=(1000.0, 1000.0),  # instant height jumps (no ramp)
        debug_vis=True,
        standing_ratio=0.3,
        flat_ratio=0.2,
        settle_time_s=2.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyObservationCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
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
        height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "height"},
        )

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
        height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "height"},
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

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=unitree_g1.G1_NO_HANDS_AGILE_ACTION_SCALE,
        use_default_offset=True,
        clip={".*": (-6.0, 6.0)},
    )

    lift = mdp.LiftActionCfg(
        asset_name="robot",
        link_to_lift="torso_link",
        stiffness_forces=5000.0,
        damping_forces=2500.0,
        force_limit_weight_fraction=0.9,
        damping_torques=100.0,
        torque_limit=250.0,
        height_sensor="height_measurement_sensor",
        target_height=unitree_g1.DEFAULT_PELVIS_HEIGHT,
        height_command="height",
        force_offset=(0.0, 0.0, 0.5),
        allow_push_down=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Regularization:
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.5e-4)
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)
    joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.01, params={"soft_ratio": 0.8})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    action_rate_rate = RewTerm(
        func=mdp.action_rate_rate_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    joint_tracking_error = RewTerm(func=mdp.joint_pos_tracking_error_l2, weight=-0.5)

    # Task: command-tracking height rewards (multi-scale for stable gradient signal)
    base_height_rough = RewTerm(
        func=mdp.track_height_command_exp,
        weight=4.0,
        params={"command_name": "height", "std": 0.5},
    )
    base_height_medium = RewTerm(
        func=mdp.track_height_command_exp,
        weight=8.0,
        params={"command_name": "height", "std": 0.3},
    )
    base_height_fine = RewTerm(
        func=mdp.track_height_command_exp,
        weight=16.0,
        params={"command_name": "height", "std": 0.2},
    )
    relaxation = RewTerm(
        func=mdp.relaxation_penalty,
        weight=-1.0,
        params={
            "command_name": "height",
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_weight": 1.0,
            "torque_weight": 1e-3,
        },
    )

    # Aesthetics:
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_if_standing,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "mode": "l1",
        },
    )
    joint_deviation_l1_arms = RewTerm(
        func=mdp.joint_deviation_if_standing,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=unitree_g1.ARM_JOINT_NAMES,
            ),
            "standing_height_threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "mode": "l1",
        },
    )
    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*ankle.*")},
    )
    # Penalize movement when tracking error is low (robot reached target)
    not_moving = RewTerm(
        func=mdp.moving_if_tracking,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "height",
            "error_threshold": 0.1,
            "weight_lin": 1.0,
            "weight_ang": 1.0,
        },
    )
    # Upright torso at moderate+ heights (not when lying down)
    torso_upright = RewTerm(
        func=mdp.upright_orientation_after_standing,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "standing_height_threshold": 0.4,
            "min_standing_duration_s": 1.0,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "norm": "l1",
        },
    )
    severely_tilted = RewTerm(
        func=mdp.severely_tilted_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
            "threshold_rad": math.radians(135),
        },
    )
    torso_roll = RewTerm(
        func=mdp.body_orientation_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "axis": "roll",
            "direction": "both",
            "kernel": "l1",
        },
    )
    forward_pitch = RewTerm(
        func=mdp.body_orientation_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "axis": "pitch",
            "direction": "forward",
            "kernel": "l2",
        },
    )
    illegal_contacts = RewTerm(
        func=mdp.illegal_contact,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_CONTACTS_LINKS),
            "threshold": 1.0,
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=unitree_g1.FEET_LINK_NAMES),
            "ref_distance": 0.2,
            "norm": "l2",
            "error_threshold": 0.2,
            "distance_mode": "absolute",
            "close_multiplier": 5.0,
            "episode_delay_s": 1.0,
            "episode_ramp_s": 2.0,
        },
    )
    feet_distance_standing = RewTerm(
        func=mdp.feet_distance_from_ref_if_standing,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=unitree_g1.FEET_LINK_NAMES),
            "ref_distance": 0.2,
            "norm": "l2",
            "error_threshold": 0.2,
            "distance_mode": "absolute",
            "close_multiplier": 5.0,
            "standing_height_threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "episode_delay_s": 1.0,
            "episode_ramp_s": 2.0,
        },
    )
    ground_unloaded = RewTerm(
        func=mdp.ground_unloaded,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=unitree_g1.FEET_LINK_NAMES),
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "height",
        },
    )
    flat_feet = RewTerm(
        func=mdp.foot_orientation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=unitree_g1.FEET_LINK_NAMES),
            "roll_weight": 1.0,
            "pitch_weight": 2.0,
            "yaw_weight": 0.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )

    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base,
        weight=-2.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=unitree_g1.FEET_LINK_NAMES),
            "base_body_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        },
    )
    # Severe binary penalty for being completely airborne (no body touching ground at all)
    completely_airborne = RewTerm(
        func=mdp.completely_airborne,
        weight=-50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold": 1.0,
        },
    )
    # Dense quasi-static motion penalty: all major body parts should move slowly
    body_velocity = RewTerm(
        func=mdp.bodies_lin_vel_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "pelvis",
                    "torso_link",
                    ".*_hip_pitch_link",
                    ".*_hip_roll_link",
                    ".*_hip_yaw_link",
                    ".*_knee_link",
                    ".*wrist.*",
                ],
            ),
            "threshold": 0.3,
        },
    )
    # Sparse impact penalty: penalize velocity at moment of ground contact (uses velocity history buffer)
    ground_slam = RewTerm(
        func=mdp.impact_velocity,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "pelvis",
                    "torso_link",
                    ".*_hip_pitch_link",
                    ".*_hip_roll_link",
                    ".*_hip_yaw_link",
                    ".*_knee_link",
                    ".*ankle_roll_link",
                ],
            ),
            "force_threshold": 10.0,
            "kernel": "l2",
        },
    )
    # Extra penalty for torso/pelvis impact — these should never slam the ground
    torso_slam = RewTerm(
        func=mdp.impact_velocity,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso_link"],
            ),
            "force_threshold": 10.0,
            "kernel": "l2",
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    invalid_state = DoneTerm(
        func=mdp.invalid_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_joint_vel": 100.0,
            "max_root_height": 5.0,
            "max_root_xy_distance": 200.0,
            "max_lin_vel": 20.0,
            "max_ang_vel": 50.0,
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
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.05, 0.05), "z": (-0.1, 0.1)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            }
        },
    )
    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "torso_link",
                ],
            ),
            "force_range": (-20.0, 20.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*wrist_yaw_link.*", ".*ankle_roll_link.*"]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_from_fallen_dataset,
        mode="reset",
        params={
            "standing_ratio": 0.5,
            "height_offset": 0.05,
            "random_fallen_ratio": 0.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    adaptive_lift = CurrTerm(
        func=mdp.adaptive_force_decay,
        params={
            "action_name": "lift",
            "metric_name": "height_error",
            "decay_when": "below",
            "threshold": 0.1,
            "command_name": "height",
            "ema_alpha": 0.05,
            "decay": 0.9999,
            "disable_threshold": 0.01,
        },
    )

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_tracking_at_timeout,
        params={
            "command_name": "height",
            "error_threshold": 0.1,
            "n_successes": 5,
            "n_failures": 5,
            "prerequisite_curriculum": "adaptive_lift",
            "prerequisite_threshold": 0.01,
            "prerequisite_direction": "below",
        },
    )

    # Polish phase: once terrain is mastered (level >= 4.0), ramp up penalties
    # ~1000 iterations = 24000 steps per curriculum for gradual adaptation
    polish_action_rate_rate = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={
            "reward_name": "action_rate_rate",
            "terminal_weight": -0.5,
            "prerequisite_curriculum": "terrain_levels",
            "prerequisite_threshold": 4.0,
            "delay_steps": 1750,
            "num_steps": 24000,
        },
    )
    polish_ground_slam = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={
            "reward_name": "ground_slam",
            "terminal_weight": -3.0,
            "prerequisite_curriculum": "terrain_levels",
            "prerequisite_threshold": 4.0,
            "delay_steps": 1500,
            "num_steps": 24000,
        },
    )
    polish_torso_slam = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={
            "reward_name": "torso_slam",
            "terminal_weight": -250.0,
            "prerequisite_curriculum": "terrain_levels",
            "prerequisite_threshold": 4.0,
            "delay_steps": 1250,
            "num_steps": 24000,
        },
    )
    polish_not_moving = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={
            "reward_name": "not_moving",
            "terminal_weight": -2.5,
            "prerequisite_curriculum": "terrain_levels",
            "prerequisite_threshold": 4.0,
            "delay_steps": 1750,
            "num_steps": 24000,
        },
    )
    polish_joint_vel_l2 = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={
            "reward_name": "joint_vel_l2",
            "terminal_weight": -0.5e-1,
            "prerequisite_curriculum": "terrain_levels",
            "prerequisite_threshold": 4.0,
            "delay_steps": 1500,
            "num_steps": 24000,
        },
    )

    # ~2000 iterations = 48000 steps for random fallen states ramp
    random_fallen_states = CurrTerm(
        func=mdp.update_event_param_after_curriculum,
        params={
            "event_term": "reset_base",
            "param_name": "random_fallen_ratio",
            "terminal_value": 0.5,
            "prerequisite_curriculum": "terrain_levels",
            "prerequisite_threshold": 3.0,
            "delay_steps": 2000,
            "num_steps": 48000,
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
    asset_name: str = "robot"
    env_index: int = 0


@configclass
class G1HeightTrackingEnvCfg(ManagerBasedRLEnvCfg):
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
        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt

        if self.sim.physics is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_max_rigid_patch_count = 2**20

        if self.scene.height_measurement_sensor is not None:
            self.scene.height_measurement_sensor.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
