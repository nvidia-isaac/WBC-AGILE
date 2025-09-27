# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict.tensordict import TensorDict
from rsl_rl.utils import resolve_nn_activation, flatten_dict
from rsl_rl.networks import Memory
from rsl_rl.modules.student_trained_teacher import SimpleMLP


class StudentTrainedTeacherRecurrent(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs: int | TensorDict,
        num_teacher_obs: int | TensorDict,
        num_actions: int,
        student_hidden_dims: list[int] = [256, 256, 256],
        teacher_path: str = None,
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTrainedTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        # load teacher
        self.teacher = torch.jit.load(teacher_path)
        self.teacher.eval()
        self.loaded_teacher = True

        # Create RNN for student
        # Calculate input dimension for RNN
        if isinstance(num_student_obs, TensorDict):
            rnn_input_dim = torch.cat([v.flatten() for v in num_student_obs.values()]).shape[0]
        else:
            rnn_input_dim = num_student_obs

        self.memory_s = Memory(
            rnn_input_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # student MLP takes RNN output
        self.student = SimpleMLP(rnn_hidden_dim, student_hidden_dims, num_actions, activation)

        print(f"Student RNN: {self.memory_s}")
        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Store whether we expect TensorDict input
        self._use_tensordict = isinstance(num_student_obs, TensorDict)

        # Create alias for compatibility with exporter
        self.memory_a = self.memory_s

    def reset(self, dones=None, hidden_states=None):
        self.memory_s.reset(dones, hidden_states)

    def forward(self):
        raise NotImplementedError

    @property
    def actor(self):
        """Expose student network as actor for compatibility with export_policy_as_jit.

        The exporter will handle RNN separately via memory_a, so we just return the MLP.
        """
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
        # Handle TensorDict observations
        if isinstance(observations, TensorDict):
            observations = flatten_dict(observations)

        # Pass through RNN
        rnn_output = self.memory_s(observations)
        # Pass through student MLP
        mean = self.student(rnn_output.squeeze(0))
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        # Handle TensorDict observations
        if isinstance(observations, TensorDict):
            observations = flatten_dict(observations)

        # Pass through RNN
        rnn_output = self.memory_s(observations)
        # Pass through student MLP
        actions_mean = self.student(rnn_output.squeeze(0))
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
        return self.memory_s.hidden_states

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)

    def get_export_policy(self, normalizer=None):
        """Get a policy module suitable for TorchScript export.

        This returns a self-contained module that includes RNN, student MLP,
        and handles TensorDict observations if needed.
        """

        class ExportablePolicy(nn.Module):
            def __init__(self, rnn: nn.Module, student: nn.Module, use_tensordict: bool, normalizer):
                super().__init__()
                self.rnn = rnn
                self.student = student
                self.use_tensordict = use_tensordict
                self.normalizer = normalizer if normalizer is not None else nn.Identity()
                self.is_lstm = isinstance(self.rnn, nn.LSTM)

                # Initialize hidden state buffers as 2D for storage
                # Shape: (num_layers, hidden_size)
                if self.is_lstm:
                    self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, self.rnn.hidden_size))
                    self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, self.rnn.hidden_size))
                    # Assign the appropriate forward and reset methods
                    if use_tensordict:
                        self.forward = self._forward_lstm_tensordict
                    else:
                        self.forward = self._forward_lstm
                    self.reset = self._reset_lstm
                else:  # GRU
                    self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, self.rnn.hidden_size))
                    # Assign the appropriate forward and reset methods
                    if use_tensordict:
                        self.forward = self._forward_gru_tensordict
                    else:
                        self.forward = self._forward_gru
                    self.reset = self._reset_gru

            def _forward_lstm(self, x):
                # Normalize the observation
                x = self.normalizer(x)
                # Reshape input to (seq_len=1, batch_size=1, input_size)
                x = x.unsqueeze(0).unsqueeze(0)
                
                # Add batch dimension to stored hidden states before passing to RNN
                # Shape: (num_layers, 1, hidden_size)
                hidden = (self.hidden_state.unsqueeze(1), self.cell_state.unsqueeze(1))
                
                # Forward pass through LSTM
                x, (h, c) = self.rnn(x, hidden)
                
                # Update stored hidden states, removing the batch dimension
                self.hidden_state[:] = h.squeeze(1)
                self.cell_state[:] = c.squeeze(1)
                
                # Reshape output to (hidden_size) for the MLP
                x = x.squeeze(0).squeeze(0)
                return self.student(x)

            def _forward_lstm_tensordict(self, x):
                x = flatten_dict(x)
                return self._forward_lstm(x)

            def _forward_gru(self, x):
                # Normalize the observation
                x = self.normalizer(x)
                # Reshape input to (seq_len=1, batch_size=1, input_size)
                x = x.unsqueeze(0).unsqueeze(0)

                # Add batch dimension to stored hidden state
                # Shape: (num_layers, 1, hidden_size)
                hidden = self.hidden_state.unsqueeze(1)

                # Forward pass through GRU
                x, h = self.rnn(x, hidden)

                # Update stored hidden state, removing the batch dimension
                self.hidden_state[:] = h.squeeze(1)

                # Reshape output to (hidden_size) for the MLP
                x = x.squeeze(0).squeeze(0)
                return self.student(x)

            def _forward_gru_tensordict(self, x):
                x = flatten_dict(x)
                return self._forward_gru(x)

            @torch.jit.export
            def _reset_lstm(self):
                """Reset LSTM hidden states."""
                self.hidden_state.zero_()
                self.cell_state.zero_()

            @torch.jit.export
            def _reset_gru(self):
                """Reset GRU hidden states."""
                self.hidden_state.zero_()

        return ExportablePolicy(self.memory_s.rnn, self.student, self._use_tensordict, normalizer)
