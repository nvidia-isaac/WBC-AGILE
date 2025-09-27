# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict.tensordict import TensorDict
from rsl_rl.utils import resolve_nn_activation, flatten_dict


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


class StudentTrainedTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_path: str = None,
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        mlp_input_dim_s = num_student_obs

        # load teacher
        self.teacher = torch.jit.load(teacher_path)
        self.teacher.eval()
        self.loaded_teacher = True

        # student
        self.student = SimpleMLP(mlp_input_dim_s, student_hidden_dims, num_actions, activation)

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def actor(self):
        """Expose student network as actor for compatibility with export_policy_as_jit."""
        return self.student

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
        mean = self.student(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actions_mean = self.student(observations)
        return actions_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student network.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Always returns True since this only loads student parameters for distillation training.
        """
        # Load the student parameters
        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
