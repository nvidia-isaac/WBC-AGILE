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


import copy
import os
from collections.abc import Callable

import torch


def export_policy_as_jit(actor_critic: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into a JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent

        # Copy student/actor policy
        # Simplified the logic to be more direct
        if hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        elif hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if hasattr(self.actor, "_forward_dict"):
                self.actor.forward = self.actor._forward_flat
        else:
            raise ValueError("Policy does not have a 'student' or 'actor' module.")

        # Set up recurrent network if it exists
        if self.is_recurrent:
            if hasattr(policy, "memory_s"):
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
            elif hasattr(policy, "memory_a"):
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            else:
                raise ValueError("Recurrent policy does not have a 'memory_s' or 'memory_a' module.")

            self.rnn.cpu()
            # <<< CHANGED >>>
            # Store hidden states as 2D. We will add/remove the batch dim dynamically.
            # Shape: (num_layers, hidden_size)
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, self.rnn.hidden_size))
            if isinstance(self.rnn, torch.nn.LSTM):
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = policy.reset
            else:  # GRU
                # No cell state for GRU
                self.forward = self.forward_gru
                self.reset = self.reset
        else:
            self.forward = self.forward_flat
            self.reset = self.reset_flat

        # Copy normalizer if it exists
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    # <<< CHANGED >>>: Correctly implemented LSTM forward pass for batch_size=1
    def forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        # Reshape input for (seq_len=1, batch_size=1, input_size)
        x = x.unsqueeze(0).unsqueeze(0)
        # Add batch dim to hidden states before passing to RNN
        hidden = (self.hidden_state.unsqueeze(1), self.cell_state.unsqueeze(1))
        # Forward pass
        x, (h, c) = self.rnn(x, hidden)
        # Update stored hidden states, removing the batch dimension
        self.hidden_state[:] = h.squeeze(1)
        self.cell_state[:] = c.squeeze(1)
        # Reshape output for the MLP
        x = x.squeeze(0).squeeze(0)
        return self.actor(x)

    # <<< NEW >>>: Added a separate GRU forward pass for clarity
    def forward_gru(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        x = x.unsqueeze(0).unsqueeze(0)
        hidden = self.hidden_state.unsqueeze(1)
        x, h = self.rnn(x, hidden)
        self.hidden_state[:] = h.squeeze(1)
        x = x.squeeze(0).squeeze(0)
        return self.actor(x)

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset_flat(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filepath)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        self.forward: Callable[..., torch.Tensor] | Callable[..., tuple[torch.Tensor, ...]]
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if hasattr(self.actor, "_forward_dict"):
                self.actor.forward = self.actor._forward_flat

            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                last_layer = self.actor.layers[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    num_actions = policy.num_actions
                    new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
                    new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
                    if last_layer.bias is not None:
                        new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
                    self.actor.layers[-1] = new_last_layer
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                last_layer = self.actor.layers[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    num_actions = policy.num_actions
                    new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
                    new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
                    if last_layer.bias is not None:
                        new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
                    self.actor.layers[-1] = new_last_layer
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        else:
            self.forward = self.forward_flat
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor.layers[0].in_features)
            torch.onnx.export(
                self,
                (obs,),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
