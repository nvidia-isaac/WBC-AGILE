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

"""Policy loading and inference."""

from pathlib import Path
from typing import Any

import torch


class PolicyWrapper:
    """Base wrapper for policy inference."""

    def __init__(self, model: Any, device: torch.device):
        """
        Initialize policy wrapper.

        Args:
            model: Loaded model (torch.jit or onnxruntime session).
            device: Device for inference.
        """
        self.model = model
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run policy inference.

        Args:
            obs: Observation tensor.

        Returns:
            Action tensor.
        """
        raise NotImplementedError

    def reset(self):
        """Reset policy state (for RNN policies)."""
        pass

    @classmethod
    def from_config(cls, checkpoint_path: Path, config: dict, device: torch.device) -> "PolicyWrapper":
        """
        Create policy wrapper from checkpoint and config.

        Args:
            checkpoint_path: Path to checkpoint file (.pt or .onnx).
            config: Configuration dictionary.
            device: Device for inference.

        Returns:
            Appropriate policy wrapper.
        """
        # Determine policy type from config.
        policy_config = config.get("policy", {"type": "MLP"})
        policy_type = policy_config.get("type", "MLP")

        # Detect file type.
        if checkpoint_path.suffix == ".onnx":
            return ONNXPolicyWrapper.load(checkpoint_path, device)
        else:
            # Load PyTorch JIT model.
            model = torch.jit.load(checkpoint_path, map_location=device)
            model.eval()

            if policy_type == "RNN":
                hidden_shape = policy_config.get("hidden_shape", [2, 1, 128])
                return RNNPolicyWrapper(model, hidden_shape, device)
            else:
                return MLPPolicyWrapper(model, device)


class MLPPolicyWrapper(PolicyWrapper):
    """Wrapper for MLP policies."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run MLP inference.

        Args:
            obs: Observation tensor, shape (obs_dim,).

        Returns:
            Action tensor, shape (action_dim,).
        """
        with torch.no_grad():
            return self.model(obs)


class RNNPolicyWrapper(PolicyWrapper):
    """Wrapper for RNN policies with hidden state management."""

    def __init__(self, model: Any, hidden_shape: list[int], device: torch.device):
        """
        Initialize RNN policy wrapper.

        Args:
            model: Loaded RNN model.
            hidden_shape: Shape of hidden state [num_layers, batch_size, hidden_dim].
            device: Device for inference.
        """
        super().__init__(model, device)
        self.hidden_shape = hidden_shape
        self.hidden = torch.zeros(*hidden_shape, device=device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run RNN inference with hidden state.

        Args:
            obs: Observation tensor, shape (obs_dim,).

        Returns:
            Action tensor, shape (action_dim,).
        """
        with torch.no_grad():
            # Add batch dimension.
            obs_batched = obs.unsqueeze(0)

            # Forward pass.
            output, self.hidden = self.model(obs_batched, self.hidden)

            # Remove batch dimension.
            return output.squeeze(0)

    def reset(self):
        """Reset hidden state to zeros."""
        self.hidden = torch.zeros(*self.hidden_shape, device=self.device)


class ONNXPolicyWrapper(PolicyWrapper):
    """Wrapper for ONNX policies."""

    def __init__(self, session: Any, device: torch.device):
        """
        Initialize ONNX policy wrapper.

        Args:
            session: ONNX runtime inference session.
            device: Device for inference (note: ONNX handles its own device placement).
        """
        super().__init__(session, device)
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run ONNX inference.

        Args:
            obs: Observation tensor, shape (obs_dim,).

        Returns:
            Action tensor, shape (action_dim,).
        """
        # Convert to numpy.
        obs_numpy = obs.cpu().numpy()

        # Add batch dimension if needed.
        if obs_numpy.ndim == 1:
            obs_numpy = obs_numpy[None, :]

        # Run inference.
        outputs = self.model.run([self.output_name], {self.input_name: obs_numpy})

        # Convert back to torch and remove batch dimension.
        action = torch.from_numpy(outputs[0]).to(self.device).squeeze(0)

        return action

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "ONNXPolicyWrapper":
        """
        Load ONNX model.

        Args:
            checkpoint_path: Path to .onnx file.
            device: Device for inference.

        Returns:
            ONNXPolicyWrapper instance.
        """
        import onnxruntime as ort

        # Set execution providers.
        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]

        # Create session.
        session = ort.InferenceSession(str(checkpoint_path), providers=providers)

        return cls(session, device)
