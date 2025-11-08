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


from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


def pushing_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_vel: dict[str, float],
    min_vel: dict[str, float],
    fall_ratio_thresholds: tuple[float, float],
) -> torch.Tensor:
    """Curriculum based on how often the robot falls.

    This term is used to increase the mangitude of distr when the robot doesnt fall and decrease the
    magnitude of disturbance when the robot falls often.

    Args:
        env: The environment.
        env_ids: The environment ids.
        max_vel: The maximum velocity magnitude.
        fall_ratio_thresholds: The thresholds at which the push velocity is increased or decreased.

    Returns:
        The mean magnitude of disturbance for the given environment ids.
    """

    current_vel_range = env.event_manager.get_term_cfg("push_robot").params["velocity_range"]

    fallen_ratio = 1 - env.termination_manager.time_outs[env_ids].float().mean()

    if fallen_ratio < min(fall_ratio_thresholds):
        # increase the push velocity if ratio is below threshold
        current_vel_range = {
            k: ((max(v[0] * 1.0001, -max_vel[k]), min(v[1] * 1.0001, max_vel[k])) if k in max_vel else v)
            for k, v in current_vel_range.items()
        }
    elif fallen_ratio > max(fall_ratio_thresholds):
        # decrease the push velocity if ratio is above threshold
        current_vel_range = {
            k: ((min(v[0] * 0.9999, -min_vel[k]), max(v[1] * 0.9999, min_vel[k])) if k in max_vel else v)
            for k, v in current_vel_range.items()
        }

    env.event_manager.get_term_cfg("push_robot").params["velocity_range"] = current_vel_range

    return list(current_vel_range.values())[0][1]
