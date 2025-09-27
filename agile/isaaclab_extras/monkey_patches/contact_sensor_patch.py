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

import carb
import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg

from .contact_sensor_data_patch import ContactSensorData


# update init data
def new_init(self: ContactSensor, cfg: ContactSensorCfg):
    """Initialize the contact sensor object.

    Args:
        cfg: The configuration parameters.
    """
    # initialize base class
    super(ContactSensor, self).__init__(cfg)

    # Enable contact processing
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/disableContactProcessing", False)

    # Create empty variables for storing output data
    self._data = ContactSensorData()  # type: ignore
    # initialize self._body_physx_view for running in extension mode
    self._body_physx_view = None


ContactSensor.__init__ = new_init


# update reset
original_reset = ContactSensor.reset


def reset_patch(self: ContactSensor, env_ids: Sequence[int] | None = None):
    original_reset(self, env_ids)
    self._data.velocities_w[env_ids] = 0.0  # type: ignore
    self._data.velocities_w_history[env_ids] = 0.0  # type: ignore


ContactSensor.reset = reset_patch


# update initialize impl
original_initialize_impl = ContactSensor._initialize_impl


def initialize_impl_patch(self: ContactSensor):
    original_initialize_impl(self)
    self._data.velocities_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)  # type: ignore
    self._data.velocities_w_history = torch.zeros_like(self._data.net_forces_w_history)  # type: ignore

    if self.cfg.history_length > 0:
        self._data.velocities_w_history = torch.zeros(  # type: ignore
            self._num_envs, self.cfg.history_length, self._num_bodies, 3, device=self._device
        )
    else:
        self._data.velocities_w_history = self._data.velocities_w.unsqueeze(1)  # type: ignore


ContactSensor._initialize_impl = initialize_impl_patch

# update update buffers impl
original_update_buffers_impl = ContactSensor._update_buffers_impl


def update_buffers_impl_patch(self: ContactSensor, env_ids: Sequence[int]):
    original_update_buffers_impl(self, env_ids)
    velocities_w = self.body_physx_view.get_velocities()[:, :3]
    self._data.velocities_w[env_ids, :, :] = velocities_w.view(-1, self._num_bodies, 3)[env_ids]  # type: ignore

    if self.cfg.history_length > 0:
        self._data.velocities_w_history[env_ids, 1:] = self._data.velocities_w_history[env_ids, :-1].clone()  # type: ignore
        self._data.velocities_w_history[env_ids, 0] = self._data.velocities_w[env_ids]  # type: ignore


ContactSensor._update_buffers_impl = update_buffers_impl_patch
