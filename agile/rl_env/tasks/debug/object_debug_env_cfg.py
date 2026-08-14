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

"""Object interaction debug environment configuration.

This environment provides interactive GUI controls for both robot joints and object poses,
useful for debugging and testing robot-object interaction policies.

Usage Example: Adapting an Existing Task Config for GUI Debugging
-----------------------------------------------------------------

To use the GUI debug actions with an existing task config that has an object in the scene,
create a new env config that inherits from it and replace the action terms in __post_init__:

.. code-block:: python

    from isaaclab.utils.configclass import configclass
    # Note: ManipulationEnvCfg does not exist, this is just an example to show how to adapt an existing task config for GUI debugging.
    from agile.rl_env.tasks.manipulation.manipulation_env_cfg import ManipulationEnvCfg
    from agile.rl_env.mdp.actions.joint_pos_gui_action_cfg import JointPositionGUIActionCfg
    from agile.rl_env.mdp.actions.object_pose_gui_action_cfg import ObjectPoseGUIActionCfg
    from agile.rl_env.mdp.rewards.reward_visualizer_cfg import RewardVisualizerCfg

    @configclass
    class ManipulationDebugEnvCfg(ManipulationEnvCfg):
        '''Manipulation environment with GUI controls for debugging.'''

        def __post_init__(self):
            super().__post_init__()

            # 1. Replace action terms with GUI-controlled versions
            self.actions.arm_joint_pos = JointPositionGUIActionCfg(
                asset_name="robot",
                joint_names=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"],
            )
            # Disable other action terms (or replace with GUI control)
            self.actions.base_velocity = None

            # 2. Add object pose GUI control if needed
            self.actions.object_pose = ObjectPoseGUIActionCfg(
                asset_name="object",
                position_limits={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.3, 1.5)},
            )

            # 3. Add reward visualizer to monitor reward terms in real-time
            self.actions.reward_monitor = RewardVisualizerCfg(
                reward_terms=[],  # Empty list = show all registered reward terms
                exclude_terms=["action_rate"],  # Optionally exclude specific terms
                show_total_reward=True,
                show_weights=True,
                show_episode_sum=True,
                enable_history_plot=True,
                history_length=200,
                gui_window_title="Reward Monitor",
            )

            # 4. Disable events that interfere with manual control
            self.events.reset_robot_pose = None
            self.events.reset_object_pose = None

            # 5. Disable terminations for extended debugging
            self.terminations.bad_base_pose = None
            self.terminations.object_dropped = None
            self.terminations.time_out = None

            # 6. Extend episode length for debugging
            self.episode_length_s = 3600.0

This pattern allows you to reuse the scene, observations, and rewards from an existing
task while enabling manual GUI control for debugging specific behaviors.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from agile.rl_env.assets.robots.unitree_g1 import G1_29DOF_DELAYED_DC_MOTOR
from agile.rl_env.mdp.actions.object_pose_gui_action_cfg import ObjectPoseGUIActionCfg
from agile.rl_env.mdp.rewards.reward_visualizer_cfg import RewardVisualizerCfg
from agile.rl_env.tasks.debug.debug_env_cfg import ActionsCfg, DebugEnvCfg, SceneCfg


@configclass
class SceneCfg(SceneCfg):
    """Scene configuration with robot and interactive object."""

    # Rigid object for interaction
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.1, 0.1),
                metallic=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.8],
            rot=[0.0, 0.0, 0.0, 1.0],  # identity (x, y, z, w)
        ),
    )


@configclass
class ActionsCfg(ActionsCfg):
    """Actions configuration with joint, object pose GUI control, and reward visualization."""

    # Object pose GUI control
    object_pose = ObjectPoseGUIActionCfg(
        asset_name="object",
        position_limits={
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.3, 1.5),
        },
        rotation_limits={
            "roll": (-math.pi, math.pi),
            "pitch": (-math.pi, math.pi),
            "yaw": (-math.pi, math.pi),
        },
        disable_gravity=True,
        gui_window_title="Object Pose Controller",
    )

    # Reward visualization - shows all reward terms by default
    reward_monitor = RewardVisualizerCfg(
        reward_terms=[],  # Empty = show all registered reward terms
        exclude_terms=[],  # Optionally exclude specific terms
        show_total_reward=True,
        show_weights=True,
        show_episode_sum=True,
        enable_history_plot=True,
        history_length=200,
        gui_window_title="Reward Monitor",
    )


@configclass
class ObjectDebugEnvCfg(DebugEnvCfg):
    """Debug environment with robot and interactive object.

    This environment provides:
    - Joint position GUI control for the robot
    - Object pose GUI control for a rigid object

    Useful for debugging robot-object interaction policies by manually
    positioning both the robot and the object.
    """

    scene: SceneCfg = SceneCfg(num_envs=2, env_spacing=2.0)
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Increase episode length for debugging
        self.episode_length_s = 3600.0


@configclass
class G1ObjectDebugEnvCfg(ObjectDebugEnvCfg):
    """G1 robot with interactive object debug environment."""

    def __post_init__(self):
        super().__post_init__()
        # Configure G1 robot
        self.scene.robot = G1_29DOF_DELAYED_DC_MOTOR.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        # Configure joint GUI for G1
        self.actions.joint_pos.mirror_actions = True
        self.actions.joint_pos.robot_type = "g1"

        # Position object in front of robot
        self.scene.object.init_state.pos = [0.5, 0.0, 0.8]

        # Update object pose GUI limits for G1 workspace
        self.actions.object_pose.position_limits = {
            "x": (0.2, 1.0),  # In front of robot
            "y": (-0.5, 0.5),  # Side to side
            "z": (0.4, 1.2),  # Reachable height
        }
