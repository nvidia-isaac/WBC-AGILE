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
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from agile.rl_env import mdp
from agile.rl_env.assets.robots.booster_t1 import T1_DELAYED_DC_CFG
from agile.rl_env.assets.robots.unitree_g1 import G1_29DOF_DELAYED_DC_MOTOR
from agile.rl_env.mdp.terrains import ROUGH_TERRAIN_CFG  # noqa: F401, F403

FILE_DIR = pathlib.Path(__file__).parent
REPO_DIR = FILE_DIR.parent.parent.parent


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
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
    robot: ArticulationCfg = MISSING

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


@configclass
class CommandsCfg:
    """Commands to debug."""


@configclass
class ObservationsCfg:
    """Observations to debug."""

    @configclass
    class RealWorldObservationsCfg(ObservationGroupCfg):
        """Observations for policy group."""

        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: RealWorldObservationsCfg = RealWorldObservationsCfg()


@configclass
class ActionsCfg:
    """Gui Actions."""

    joint_pos = mdp.JointPositionGUIActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class RewardsCfg:
    """Rewards to debug."""

    dummy_reward = RewardTermCfg(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Terminations to debug."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Events to debug."""

    reset_root_state = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "z": (0.2, 0.2),
                "yaw": (math.pi / 2, math.pi / 2),
            },
            "velocity_range": {},
        },
    )


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    eye: tuple[float, float, float] = (0.0, 2.5, 1.2)

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.9)

    cam_prim_path: str = "/OmniverseKit_Persp"

    resolution: tuple[int, int] = (1280, 720)

    origin_type = "world"

    env_index: int = 0


@configclass
class DebugEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=2, env_spacing=1.25)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class G1DebugEnvCfg(DebugEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_29DOF_DELAYED_DC_MOTOR.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        # self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/pelvis/.*"
        self.actions.joint_pos.mirror_actions = True
        self.actions.joint_pos.robot_type = "g1"


@configclass
class T1DebugEnvCfg(DebugEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = T1_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
        self.actions.joint_pos.robot_type = "t1"
