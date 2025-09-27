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


class StudentFinetuneRecurrent(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs: int | TensorDict,
        num_critic_obs: int | TensorDict,
        num_actions: int,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentFinetuneRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # Handle TensorDict observations
        if isinstance(num_student_obs, TensorDict):
            rnn_input_dim_student = torch.cat([v.flatten() for v in num_student_obs.values()]).shape[0]
            self.preprocess_obs = flatten_dict
        else:
            rnn_input_dim_student = num_student_obs
            self.preprocess_obs = nn.Identity()

        if isinstance(num_critic_obs, TensorDict):
            rnn_input_dim_critic = torch.cat([v.flatten() for v in num_critic_obs.values()]).shape[0]
        else:
            rnn_input_dim_critic = num_critic_obs

        # Create RNN for student (actor)
        self.memory_a = Memory(
            rnn_input_dim_student,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # Create RNN for critic
        self.memory_c = Memory(
            rnn_input_dim_critic,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # student (actor) MLP takes RNN output
        self.actor = SimpleMLP(rnn_hidden_dim, actor_hidden_dims, num_actions, activation)

        # critic MLP takes RNN output
        self.critic = SimpleMLP(rnn_hidden_dim, critic_hidden_dims, 1, activation)

        print(f"Student (Actor) RNN: {self.memory_a}")
        print(f"Student (Actor) MLP: {self.actor}")
        print(f"Critic RNN: {self.memory_c}")
        print(f"Critic MLP: {self.critic}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Store whether we expect TensorDict input
        self._use_tensordict = isinstance(num_student_obs, TensorDict)

        # Create alias for compatibility with exporter
        self.memory_s = self.memory_a

    def reset(self, dones=None, hidden_states=None):
        """Reset both actor and critic RNN hidden states."""
        self.memory_a.reset(dones, hidden_states)
        self.memory_c.reset(dones, hidden_states)

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

    def update_distribution(self, observations, masks=None, hidden_states=None):
        # Handle TensorDict observations
        observations = self.preprocess_obs(observations)

        # Pass through actor RNN
        rnn_output = self.memory_a(observations, masks, hidden_states)
        # Pass through actor MLP
        mean = self.actor(rnn_output.squeeze(0))
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None):
        self.update_distribution(observations, masks, hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # Handle TensorDict observations
        observations = self.preprocess_obs(observations)

        # Pass through actor RNN
        rnn_output = self.memory_a(observations)
        # Pass through actor MLP
        actions_mean = self.actor(rnn_output.squeeze(0))
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate the critic network to get value estimates."""
        # Handle TensorDict observations
        critic_observations = self.preprocess_obs(critic_observations)
        
        # Pass through critic RNN
        rnn_output = self.memory_c(critic_observations, masks, hidden_states)
        # Pass through critic MLP
        value = self.critic(rnn_output.squeeze(0))
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load only the student (actor) parameters for finetuning.

        Args:
            state_dict (dict): State dictionary of the model (should contain student/actor parameters).
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Always returns True since this loads student parameters for finetuning.
        """
        # Filter state dict to only include student/actor parameters
        student_state_dict = {}
        student_keys_found = []
        
        for key, value in state_dict.items():
            # Include memory_s (student RNN), student MLP, and std parameters
            if key.startswith('memory_s.') or key.startswith('student.') or key.startswith('std'):
                student_keys_found.append(key)
                # Map memory_s to both memory_a and memory_s (since they're aliases in current model)
                # and student to actor for compatibility
                if key.startswith('memory_s.'):
                    # Load into memory_a
                    new_key_a = key.replace('memory_s.', 'memory_a.')
                    student_state_dict[new_key_a] = value
                    # Also load into memory_s (since it's an alias in the current model)
                    student_state_dict[key] = value
                elif key.startswith('student.'):
                    new_key = key.replace('student.', 'actor.')
                    student_state_dict[new_key] = value
                else:  # std parameter
                    student_state_dict[key] = value
        
        print(f"Found {len(student_keys_found)} student parameters to load:")
        for key in student_keys_found:
            print(f"  {key}")
        
        # Load only the student parameters, ignore missing critic parameters
        missing_keys, unexpected_keys = super().load_state_dict(student_state_dict, strict=False)
        
        # Check that all student parameters were loaded successfully
        student_missing = [k for k in missing_keys if not ('critic' in k or 'memory_c' in k)]
        if student_missing:
            print(f"WARNING: Some student parameters were not loaded: {student_missing}")
        else:
            print("✓ All student parameters loaded successfully!")
        
        critic_missing = [k for k in missing_keys if 'critic' in k or 'memory_c' in k]
        print(f"Critic parameters not loaded (expected): {len(critic_missing)} parameters")
        
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        return True

    def get_hidden_states(self):
        """Get hidden states from both actor and critic RNNs."""
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def detach_hidden_states(self, dones=None):
        """Detach hidden states for both actor and critic RNNs."""
        self.memory_a.detach_hidden_states(dones)
        self.memory_c.detach_hidden_states(dones)

    def get_export_policy(self, normalizer=None):
        """Get a policy module suitable for TorchScript export.

        This returns a self-contained module that includes actor RNN, actor MLP,
        and handles TensorDict observations if needed.
        """

        class ExportablePolicy(nn.Module):
            def __init__(self, rnn: nn.Module, actor: nn.Module, use_tensordict: bool, normalizer):
                super().__init__()
                self.rnn = rnn
                self.actor = actor
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
                return self.actor(x)

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
                return self.actor(x)

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

        return ExportablePolicy(self.memory_a.rnn, self.actor, self._use_tensordict, normalizer)
