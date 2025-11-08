# mypy: disable-error-code="attr-defined"

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

from isaaclab.envs import mdp as isaaclab_mdp  # To avoid circular import
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from . import observations_io


@configclass
class EvaluationObservationsCfg(ObsGroup):
    """Observation specifications for evaluation."""

    joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos)

    joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel)

    joint_acc = ObsTerm(func=observations_io.joint_acc)

    root_pos = ObsTerm(func=isaaclab_mdp.root_pos_w)

    root_rot = ObsTerm(func=isaaclab_mdp.root_quat_w)

    root_lin_vel = ObsTerm(func=isaaclab_mdp.root_lin_vel_w)

    root_ang_vel = ObsTerm(func=isaaclab_mdp.root_ang_vel_w)

    commands = ObsTerm(func=observations_io.velocity_height_command, params={"command_name": "base_velocity"})

    actions = ObsTerm(func=isaaclab_mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1
