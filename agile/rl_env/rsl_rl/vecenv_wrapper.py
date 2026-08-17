# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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


import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from agile.rl_env.rsl_rl_observations import to_rsl_rl_observations


class RslRlVecEnvWrapper(VecEnv):
    """Wrap around Isaac Lab environment for RSL-RL library.

    RSL-RL 5.x consumes observations as a :class:`tensordict.TensorDict`. Observation-group
    selection is handled by the runner configuration through ``obs_groups``.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does
        not follow the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be
        modified to work with this wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initialize the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call
            reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or
                :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. "
                f"Environment type: {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # build termination handling map from DoneTermCfg metadata
        self._termination_handling: dict[str, tuple[str, float]] = {}
        self._init_termination_handling()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Return the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Return the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Return the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Return the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Return the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Return the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Return the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Return the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> TensorDict:
        """Return the current observations of the environment."""
        obs_dict = self._compute_observations()
        return to_rsl_rl_observations(obs_dict, self.num_envs)

    def get_observations_with_extras(self) -> tuple[TensorDict, dict]:
        """Return current observations and raw Isaac Lab observations for recording/eval."""
        obs_dict = self._compute_observations()
        return to_rsl_rl_observations(obs_dict, self.num_envs), {"observations": obs_dict}

    def _compute_observations(self) -> dict:
        """Compute raw Isaac Lab observations."""
        if hasattr(self.unwrapped, "observation_manager"):
            return self.unwrapped.observation_manager.compute()
        return self.unwrapped._get_observations()

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Return the episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        # return observations
        return to_rsl_rl_observations(obs_dict, self.num_envs), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for the RSL-RL VecEnv API
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # aggregate per-term termination sigmas for good/bad termination handling
        if self._termination_handling:
            self._compute_termination_sigmas(extras)

        # return the step information
        return to_rsl_rl_observations(obs_dict, self.num_envs), rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _init_termination_handling(self):
        """Scan termination configs for good/bad terms with sigma metadata.

        Builds a mapping of ``{term_name: (termination_type, sigma)}`` from
        :class:`DoneTermCfg` instances that have ``termination_type`` set to
        ``"good"`` or ``"bad"``.
        """
        if not isinstance(self.unwrapped, ManagerBasedRLEnv):
            return
        term_manager = self.unwrapped.termination_manager
        cfg = self.unwrapped.cfg.terminations
        for name in term_manager._term_names:
            term_cfg = getattr(cfg, name, None)
            if term_cfg is None:
                continue
            term_type = getattr(term_cfg, "termination_type", "neutral")
            if term_type in ("good", "bad"):
                sigma = getattr(term_cfg, "sigma", 5.0)
                self._termination_handling[name] = (term_type, sigma)

    def _compute_termination_sigmas(self, extras: dict):
        """Compute per-env aggregated sigma for good/bad terminations.

        For each environment, takes the max sigma across all fired good (or bad)
        termination terms. Adds ``bad_termination_sigma`` and/or
        ``good_termination_sigma`` tensors to extras.
        """
        term_manager = self.unwrapped.termination_manager
        bad_sigma = torch.zeros(self.num_envs, device=self.device)
        good_sigma = torch.zeros(self.num_envs, device=self.device)
        for name, (term_type, sigma) in self._termination_handling.items():
            term_mask = term_manager.get_term(name)
            if term_type == "bad":
                bad_sigma[term_mask] = torch.clamp_min(bad_sigma[term_mask], sigma)
            else:  # "good"
                good_sigma[term_mask] = torch.clamp_min(good_sigma[term_mask], sigma)
        if bad_sigma.any():
            extras["bad_termination_sigma"] = bad_sigma
        if good_sigma.any():
            extras["good_termination_sigma"] = good_sigma

    def _modify_action_space(self):
        """Modify the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change
        # it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
