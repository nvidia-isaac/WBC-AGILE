# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict
from torch.distributions import Normal
from typing import Literal

from rsl_rl.utils import flatten_dict, resolve_nn_activation


class SimpleMLP(nn.Module):
    """
    Simple MLP with linear layers and activation functions.
    Compatible with TensorDict observations.
    """

    def __init__(
        self,
        input_dim: int | TensorDict,
        hidden_dims: list[int],
        output_dim: int,
        activation: torch.nn.Module,
    ):
        super().__init__()

        # define layers
        if isinstance(input_dim, TensorDict):
            in_dim = torch.cat([v.flatten() for v in input_dim.values()]).shape[0]
        else:
            in_dim = input_dim

        layers = []
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)

        # define forward method
        if isinstance(input_dim, TensorDict):
            self.forward = self._forward_dict
        else:
            self.forward = self._forward_flat

    def _forward_flat(self, x: torch.Tensor):
        return self.layers(x)

    @torch.jit.ignore
    def _forward_dict(self, x: TensorDict):
        return self.layers(flatten_dict(x))


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int | TensorDict,
        num_critic_obs: int | TensorDict,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: Literal["scalar", "log", "pred"] = "scalar",
        log_std_range: tuple[float, float] = (-7.0, 2.0),
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Policy
        action_dim_scale = 2 if noise_std_type == "pred" else 1

        self.actor = SimpleMLP(num_actor_obs, actor_hidden_dims, num_actions * action_dim_scale, activation)

        # Value function
        self.critic = SimpleMLP(num_critic_obs, critic_hidden_dims, 1, activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif self.noise_std_type == "pred":
            self.num_actions = num_actions
            self.std_offset = math.log(init_noise_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.log_std_range = log_std_range

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = torch.clamp(self.std.expand_as(mean), min=1e-6)
        elif self.noise_std_type == "log":
            std = torch.exp(torch.clamp(self.log_std, min=self.log_std_range[0], max=self.log_std_range[1])).expand_as(
                mean
            )
        elif self.noise_std_type == "pred":
            mean, std_logits = mean.split(self.num_actions, dim=-1)
            std = torch.exp(
                torch.clamp(
                    std_logits + self.std_offset,
                    min=self.log_std_range[0],
                    max=self.log_std_range[1],
                )
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar', 'log' or 'pred'"
            )
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean[:, : self.num_actions] if self.noise_std_type == "pred" else actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
