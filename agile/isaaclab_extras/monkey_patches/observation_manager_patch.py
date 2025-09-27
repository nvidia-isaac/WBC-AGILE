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


from tensordict.tensordict import TensorDict

from isaaclab.managers import ObservationManager

original_compute_group = ObservationManager.compute_group


def tensordict_compute_group_wrapper(self: ObservationManager, group_name: str, *args, **kwargs):
    group_obs = original_compute_group(self, group_name, *args, **kwargs)
    if not self._group_obs_concatenate[group_name]:
        return TensorDict(group_obs, batch_size=self._env.num_envs, device=self.device)
    return group_obs


ObservationManager.compute_group = tensordict_compute_group_wrapper
