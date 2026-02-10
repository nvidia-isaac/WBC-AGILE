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

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.lift.mdp import rewards as lift_rewards

from agile.rl_env import mdp
from agile.rl_env.assets.robots.unitree_g1 import (
    LEFT_HAND_ARM_JOINT_NAMES,
    LEG_JOINT_NAMES,
    NO_HAND_JOINT_NAMES,
    RIGHT_HAND_ARM_JOINT_NAMES,
    WAIST_JOINT_NAMES,
)
from agile.rl_env.mdp.actions.actions_cfg import (
    AgileLowerBodyActionCfg,
    DeltaJointPositionActionCfg,
)

#################################################
# Scene definition
#################################################


@configclass
class PickPlaceTrackingSceneCfg(InteractiveSceneCfg):
    """Configuration for the pick-place tracking scene with a legged robot."""

    terrain = terrain_gen.TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # robots
    robot: ArticulationCfg = MISSING

    # fix object
    fixture_structure = AssetBaseCfg(
        prim_path="/World/envs/env_.*/fixture_structure",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Mimic/exhaust_pipe_task/exhaust_pipe_assets/table.usd",
            scale=(0.8, 0.8, 0.92),
            activate_contact_sensors=False,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.5),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Makes the object fixed/immovable
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=MultiUsdFileCfg(
            usd_path=[
                f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
                f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            ],
            random_choice=True,  # Each environment gets a random USD from the list
            scale=(0.9, 0.9, 0.9),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
    )

    # End-effector frame transformer
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer").replace(
            markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.1, 0.1, 0.1))}
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand/right_hand_palm_link",
                name="right_hand_palm",
                offset=OffsetCfg(
                    pos=[0.05, 0.01, 0.0],
                    rot=[1.0, 0.0, 0.0, 0.0],
                ),
            ),
        ],
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


#################################################
# MDP settings
#################################################


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    tracking_command: mdp.TrackingCommandCfg = mdp.TrackingCommandCfg(
        joint_names=RIGHT_HAND_ARM_JOINT_NAMES + WAIST_JOINT_NAMES,
        file_path="",
        pos_offset=(0.0, 0.0, 0.06814659),
        object_name="object",
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_joint_pos = DeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_HAND_ARM_JOINT_NAMES + WAIST_JOINT_NAMES,
        steady_joint_names=LEFT_HAND_ARM_JOINT_NAMES,
        joint_limits={
            "waist_roll_joint": (-0.1, 0.1),
            "waist_pitch_joint": (-0.1, 0.1),
            "waist_yaw_joint": (-0.2, 0.2),
        },
    )

    lower_body_joint_pos = AgileLowerBodyActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        obs_group_name="agile_policy",  # need to be the same name as the on in ObservationCfg
        policy_path="agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt",
        clip={"vx": (-0.2, 0.5), "vy": (-0.4, 0.4), "wz": (-0.5, 0.5), "height": (0.65, 0.72)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "tracking_command"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "tracking_command"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        motion_joint_pos_delta = ObsTerm(
            func=mdp.motion_joint_pos_delta,
            params={"command_name": "tracking_command"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Object position relative to hand palm in body frame
        object_to_hand_pos = ObsTerm(
            func=mdp.object_to_hand_pos_b,
            params={"command_name": "tracking_command", "ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        object_pos_error = ObsTerm(
            func=mdp.object_pos_error,
            params={"command_name": "tracking_command"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        trajectory_progress = ObsTerm(
            func=mdp.trajectory_progress,
            params={"command_name": "tracking_command"},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            """Post initialization."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AgileTeacherPolicyObservationsCfg(ObsGroup):
        """Observation specifications for the Agile lower body policy.

        Note: This configuration defines only part of the observation input to the Agile lower body policy.
        The lower body command portion is appended to the observation tensor in the action term, as that
        is where the environment has access to those commands.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=NO_HAND_JOINT_NAMES,
                ),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=NO_HAND_JOINT_NAMES,
                ),
            },
        )

        actions = ObsTerm(
            func=mdp.last_action,
            params={
                "action_name": "lower_body_joint_pos",
            },
        )

        def __post_init__(self) -> None:
            """Post initialization."""
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    agile_policy: AgileTeacherPolicyObservationsCfg = AgileTeacherPolicyObservationsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot = EventTerm(
        func=mdp.reset_robot_to_trajectory,
        mode="reset",
        params={
            "trajectory_time_idx": (0, 150),
            "command_name": "tracking_command",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Reward being static (low velocities) toward the end of the trajectory
    static_at_goal = RewTerm(
        func=mdp.static_at_goal_exp,
        weight=1.0,
        params={
            "command_name": "tracking_command",
            "progress_threshold": 0.8,  # start rewarding staticness at 80% of trajectory
            "joint_vel_std": 0.3,  # smaller = stricter requirement for static joints
            "root_vel_std": 0.3,  # smaller = stricter requirement for static base
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=RIGHT_HAND_ARM_JOINT_NAMES + WAIST_JOINT_NAMES
            ),  # only upper body joints
        },
    )

    # Tracking rewards
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "tracking_command", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "tracking_command", "std": 0.3},
    )
    upper_body_joint_pos = RewTerm(
        func=mdp.motion_tracked_joint_pos_error_exp,
        weight=1.0,
        params={"command_name": "tracking_command", "std": 0.3},
    )

    # Object interaction rewards
    object_pos_tracking = RewTerm(
        func=mdp.motion_object_position_error_exp, weight=1.0, params={"command_name": "tracking_command", "std": 0.3}
    )

    # Auto-phased hand-object proximity reward:
    # - Before trajectory peak: Full proximity reward (approach/grasp/lift)
    # - After trajectory peak: Decaying reward over N steps (release phase)
    # Peak is auto-detected from reference trajectory - no manual tuning needed!
    hand_object_tracking = RewTerm(
        func=mdp.hand_object_distance_tracking_exp,
        weight=1.0,
        params={
            "command_name": "tracking_command",
            "std": 0.1,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "release_decay_steps": 50,  # Steps to decay proximity reward after peak
        },
    )

    lifting_object = RewTerm(func=lift_rewards.object_is_lifted, params={"minimal_height": 0.95}, weight=0.5)

    # Reward to encourage reaching the final frame's posture at the end of the trajectory
    nominal_posture_at_end = RewTerm(
        func=mdp.nominal_posture_at_end_exp,
        weight=1.0,
        params={
            "command_name": "tracking_command",
            "progress_threshold": 0.95,
            "std": 0.5,
        },
    )

    # To reduce upper body shaking
    root_acc = RewTerm(
        func=mdp.body_acc_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Penalize torso angular velocity to reduce shaking (orthogonal to tracking rewards)
    torso_ang_vel = RewTerm(
        func=mdp.body_ang_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_ARM_JOINT_NAMES + WAIST_JOINT_NAMES)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_HAND_ARM_JOINT_NAMES + WAIST_JOINT_NAMES)},
    )
    joint_pos_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_base_pose = DoneTerm(
        func=mdp.bad_base_pose,
        params={
            "command_name": "tracking_command",
            "base_pos_threshold": 0.7,
        },
    )
    bad_base_rotation = DoneTerm(
        func=mdp.bad_base_rotation,
        params={
            "command_name": "tracking_command",
            "base_ori_threshold": 0.7,
        },
    )
    bad_joint_pos = DoneTerm(
        func=mdp.bad_joint_pos,
        params={
            "command_name": "tracking_command",
            "joint_pos_threshold": 2.0,
        },
    )

    robot_fall = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "in_bound_range": {"z": [0.5, 1.5]},
        },
    )

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "in_bound_range": {"z": [0.6, 1.5]},
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    increase_object_pos_tracking = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "object_pos_tracking",
            "start_step": 50_000,
            "num_steps": 50_000,
            "terminal_weight": 5.0,
            "use_log_space": False,
        },
    )

    # Start with weight 0.5, decay to 0.0 over training
    lifting_object_curriculum = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "lifting_object",
            "start_step": 50_000,
            "num_steps": 50_000,
            "terminal_weight": 0.0,
        },
    )

    # Increase the action rate penalty
    increase_action_rate_penalty = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate_l2",
            "start_step": 100_000,
            "num_steps": 50_000,
            "terminal_weight": -0.01,
        },
    )


#################################################
# Environment configuration
#################################################


@configclass
class PickPlaceTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick-place tracking environment."""

    # Scene settings
    # NOTE: replicate_physics=False is required for MultiUsdFileCfg/MultiAssetSpawnerCfg
    # to spawn different objects per environment
    scene: PickPlaceTrackingSceneCfg = PickPlaceTrackingSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (-2.5, -5.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.viewer.origin_type = "world"
