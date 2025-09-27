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
    """Monkey patch for the :meth:`ManagerBasedRLEnv.step` method.
    This patch adds pre-sim step events to the :meth:`ManagerBasedRLEnv.step` method.
    With this patch, events can have the mode "pre_sim_step". These events are applied before the simulation is stepped.

    The only difference is the addition of the following code:

    .. code-block:: python
        # apply pre-sim step events. This is why we need to patch
        if "pre_sim_step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="pre_sim_step", dt=self.step_dt)
    """
    # process actions
    self.action_manager.process_action(action.to(self.device))

    self.recorder_manager.record_pre_step()

    # check if we need to do rendering within the physics loop
    # note: checked here once to avoid multiple checks within the loop
    is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

    # perform physics stepping
    for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        # set actions into buffers
        self.action_manager.apply_action()
        # set actions into simulator
        self.scene.write_data_to_sim()
        # apply pre-sim step events. This is why we need to patch
        if "pre_sim_step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="pre_sim_step", dt=self.step_dt)
        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=self.physics_dt)

    # post-step:
    # -- update env counters (used for curriculum generation)
    self.episode_length_buf += 1  # step in current episode (per env)
    self.common_step_counter += 1  # total step (common for all envs)
    # -- check terminations
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    # -- reward computation
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

    if len(self.recorder_manager.active_terms) > 0:
        # update observations for recording if needed
        self.obs_buf = self.observation_manager.compute()
        self.recorder_manager.record_post_step()

    # -- reset envs that terminated/timed-out and log the episode information
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(reset_env_ids)

        self._reset_idx(reset_env_ids)
        # update articulation kinematics
        self.scene.write_data_to_sim()

        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(reset_env_ids)

    # -- update command
    self.command_manager.compute(dt=self.step_dt)
    # -- step interval events
    if "interval" in self.event_manager.available_modes:
        self.event_manager.apply(mode="interval", dt=self.step_dt)
    # -- compute observations
    # note: done after reset to get the correct observations for reset envs
    self.obs_buf = self.observation_manager.compute(update_history=True)

    # return observations, rewards, resets and extras
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


ManagerBasedRLEnv.step = new_step
