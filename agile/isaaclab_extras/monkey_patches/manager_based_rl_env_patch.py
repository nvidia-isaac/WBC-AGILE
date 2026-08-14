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


from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


def new_step(self: ManagerBasedRLEnv, action: torch.Tensor) -> VecEnvStepReturn:
    """``ManagerBasedRLEnv.step`` with two AGILE extensions: ``pre_sim_step`` events are applied
    before each simulation sub-step, and resets are skipped while ``_disable_terminations`` is set
    (e.g. during fallen-state dataset collection).
    """
    self.action_manager.process_action(action.to(self.device))
    self.recorder_manager.record_pre_step()
    is_rendering = self.sim.is_rendering

    if self._physics_handles_decimation:
        self._sim_step_counter += self.cfg.decimation
        self.action_manager.apply_action()
        self.scene.write_data_to_sim()
        if "pre_sim_step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="pre_sim_step", dt=self.step_dt)
        self.sim.step(render=False)
        self.recorder_manager.record_post_physics_decimation_step()
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render(skip_app_pumping=not self.render_enabled)
        self.scene.update(dt=self.step_dt)
    else:
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            if "pre_sim_step" in self.event_manager.available_modes:
                self.event_manager.apply(mode="pre_sim_step", dt=self.step_dt)
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.physics_dt)

    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    if getattr(self, "_disable_terminations", False):
        self.reset_buf = torch.zeros_like(self.reset_buf)
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    if len(self.recorder_manager.active_terms) > 0:
        self.obs_buf = self.observation_manager.compute()
        self.recorder_manager.record_post_step()

    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
    if len(reset_env_ids) > 0:
        self.recorder_manager.record_pre_reset(reset_env_ids)
        self._reset_idx(reset_env_ids)
        if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        self.recorder_manager.record_post_reset(reset_env_ids)

    self.command_manager.compute(dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
        self.event_manager.apply(mode="interval", dt=self.step_dt)
    self.obs_buf = self.observation_manager.compute(update_history=True)

    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


ManagerBasedRLEnv.step = new_step
