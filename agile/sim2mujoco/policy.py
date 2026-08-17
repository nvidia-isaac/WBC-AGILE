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

"""Policy loading and inference.

Supports three checkpoint formats:
  - TorchScript (.pt exported via JIT)
  - ONNX (.onnx)
  - Training checkpoints (.pt saved by RSL-RL OnPolicyRunner)
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Activation resolver (mirrors rsl_rl.utils.resolve_nn_activation)
# ---------------------------------------------------------------------------

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "crelu": nn.CELU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def _resolve_activation(name: str) -> nn.Module:
    cls = _ACTIVATION_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown activation '{name}'. Supported: {list(_ACTIVATION_MAP)}")
    return cls()


# ---------------------------------------------------------------------------
# Lightweight inference model reconstructed from a training checkpoint
# ---------------------------------------------------------------------------


def _infer_architecture(model_state: dict[str, torch.Tensor]) -> dict:
    """Infer actor architecture from training checkpoint state dict.

    Returns a dict with keys:
        actor_input_dim, actor_hidden_dims, actor_output_dim,
        num_actions, noise_std_type,
        is_recurrent, rnn_type, rnn_input_dim, rnn_hidden_dim, rnn_num_layers
    """
    # Collect linear layer weights in order.
    weight_keys = sorted(
        [k for k in model_state if k.startswith("mlp.") and k.endswith(".weight")],
        key=lambda k: int(k.replace("mlp.", "").split(".")[0]),
    )
    if not weight_keys:
        raise ValueError(f"No policy MLP weight keys found in checkpoint. Keys found: {list(model_state)[:10]}")

    actor_input_dim = model_state[weight_keys[0]].shape[1]
    actor_hidden_dims = [model_state[k].shape[0] for k in weight_keys[:-1]]
    actor_output_dim = model_state[weight_keys[-1]].shape[0]

    # Detect noise_std_type.
    if "distribution.std_param" in model_state:
        noise_std_type = "scalar"
        num_actions = model_state["distribution.std_param"].shape[0]
    elif "distribution.log_std_param" in model_state:
        noise_std_type = "log"
        num_actions = model_state["distribution.log_std_param"].shape[0]
    else:
        noise_std_type = "pred"
        num_actions = actor_output_dim // 2

    # Detect RNN.
    is_recurrent = any(k.startswith("rnn.rnn.") for k in model_state)
    rnn_type = None
    rnn_input_dim = 0
    rnn_hidden_dim = 0
    rnn_num_layers = 0

    if is_recurrent:
        wih = model_state.get("rnn.rnn.weight_ih_l0")
        whh = model_state.get("rnn.rnn.weight_hh_l0")
        if wih is None or whh is None:
            raise ValueError("Recurrent policy detected but missing rnn.rnn weights")

        rnn_input_dim = wih.shape[1]
        rnn_hidden_dim = whh.shape[1]
        gate_size = wih.shape[0]

        if gate_size == 3 * rnn_hidden_dim:
            rnn_type = "gru"
        elif gate_size == 4 * rnn_hidden_dim:
            rnn_type = "lstm"
        else:
            raise ValueError(f"Cannot determine RNN type from gate size {gate_size}")

        while f"rnn.rnn.weight_ih_l{rnn_num_layers}" in model_state:
            rnn_num_layers += 1

    return {
        "weight_keys": weight_keys,
        "actor_input_dim": actor_input_dim,
        "actor_hidden_dims": actor_hidden_dims,
        "actor_output_dim": actor_output_dim,
        "num_actions": num_actions,
        "noise_std_type": noise_std_type,
        "is_recurrent": is_recurrent,
        "rnn_type": rnn_type,
        "rnn_input_dim": rnn_input_dim,
        "rnn_hidden_dim": rnn_hidden_dim,
        "rnn_num_layers": rnn_num_layers,
    }


class _CheckpointInferenceModel(nn.Module):
    """Lightweight nn.Module for inference from a training checkpoint.

    Replicates the forward pass of the JIT exporter: normalizer -> (rnn) -> actor.
    """

    def __init__(self, arch: dict, activation: str):
        super().__init__()
        self.noise_std_type = arch["noise_std_type"]
        self.num_actions = arch["num_actions"]
        self.is_recurrent = arch["is_recurrent"]

        # Build actor MLP.
        layers = OrderedDict()
        in_dim = arch["actor_input_dim"]
        for layer_index, weight_key in enumerate(arch["weight_keys"]):
            module_name = weight_key.replace("mlp.", "").split(".")[0]
            if layer_index == len(arch["weight_keys"]) - 1:
                out_dim = arch["actor_output_dim"]
            else:
                out_dim = arch["actor_hidden_dims"][layer_index]
            layers[module_name] = nn.Linear(in_dim, out_dim)
            in_dim = out_dim
            if layer_index != len(arch["weight_keys"]) - 1:
                layers[f"{module_name}_activation"] = _resolve_activation(activation)
        self.actor = nn.Sequential(layers)

        # Normalizer (replaced after construction if checkpoint has one).
        self.normalizer: nn.Module = nn.Identity()

        # RNN (optional).
        if self.is_recurrent:
            rnn_cls = nn.LSTM if arch["rnn_type"] == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=arch["rnn_input_dim"],
                hidden_size=arch["rnn_hidden_dim"],
                num_layers=arch["rnn_num_layers"],
                batch_first=False,
            )
            self.register_buffer("hidden_state", torch.zeros(arch["rnn_num_layers"], arch["rnn_hidden_dim"]))
            if arch["rnn_type"] == "lstm":
                self.register_buffer("cell_state", torch.zeros(arch["rnn_num_layers"], arch["rnn_hidden_dim"]))
            self._rnn_type = arch["rnn_type"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)

        if self.is_recurrent:
            # (seq=1, batch=1, input_size)
            x = x.unsqueeze(0).unsqueeze(0)
            if self._rnn_type == "lstm":
                hidden = (self.hidden_state.unsqueeze(1), self.cell_state.unsqueeze(1))
                x, (h, c) = self.rnn(x, hidden)
                self.hidden_state[:] = h.squeeze(1)
                self.cell_state[:] = c.squeeze(1)
            else:
                hidden = self.hidden_state.unsqueeze(1)
                x, h = self.rnn(x, hidden)
                self.hidden_state[:] = h.squeeze(1)
            x = x.squeeze(0).squeeze(0)

        out = self.actor(x)

        # For pred noise type, output contains [mean, std_logits]; take only mean.
        if self.noise_std_type == "pred":
            out = out[: self.num_actions]

        return out

    def reset_hidden(self):
        if self.is_recurrent:
            self.hidden_state.zero_()
            if self._rnn_type == "lstm":
                self.cell_state.zero_()


class _SimpleNormalizer(nn.Module):
    """Minimal observation normalizer: (x - mean) / (std + eps)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        # Squeeze leading dim so shape is (obs_dim,) -- works with both 1D and batched inputs.
        self.register_buffer("_mean", mean.squeeze(0))
        self.register_buffer("_std", std.squeeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


# ---------------------------------------------------------------------------
# Policy wrappers
# ---------------------------------------------------------------------------


class PolicyWrapper:
    """Base wrapper for policy inference."""

    def __init__(self, model: Any, device: torch.device):
        self.model = model
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        """Reset policy state (for RNN policies)."""
        pass

    @classmethod
    def from_config(cls, checkpoint_path: Path, config: dict, device: torch.device) -> "PolicyWrapper":
        """Create policy wrapper from checkpoint and config.

        Supports TorchScript, ONNX, and raw training checkpoints.
        """
        if checkpoint_path.suffix == ".onnx":
            return ONNXPolicyWrapper.load(checkpoint_path, device)

        # Try TorchScript first; fall back to training checkpoint.
        try:
            model = torch.jit.load(checkpoint_path, map_location=device)
            model.eval()

            policy_config = config.get("policy", {"type": "MLP"})
            policy_type = policy_config.get("type", "MLP")

            if policy_type == "RNN":
                hidden_shape = policy_config.get("hidden_shape", [2, 1, 128])
                return RNNPolicyWrapper(model, hidden_shape, device)
            else:
                return MLPPolicyWrapper(model, device)

        except RuntimeError:
            # Not a TorchScript file -- try as a training checkpoint.
            return CheckpointPolicyWrapper.from_checkpoint(checkpoint_path, config, device)


class MLPPolicyWrapper(PolicyWrapper):
    """Wrapper for TorchScript MLP policies."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(obs)


class RNNPolicyWrapper(PolicyWrapper):
    """Wrapper for TorchScript RNN policies with hidden state management."""

    def __init__(self, model: Any, hidden_shape: list[int], device: torch.device):
        super().__init__(model, device)
        self.hidden_shape = hidden_shape
        self.hidden = torch.zeros(*hidden_shape, device=device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs_batched = obs.unsqueeze(0)
            output, self.hidden = self.model(obs_batched, self.hidden)
            return output.squeeze(0)

    def reset(self):
        self.hidden = torch.zeros(*self.hidden_shape, device=self.device)


class CheckpointPolicyWrapper(PolicyWrapper):
    """Wrapper for policies loaded directly from RSL-RL training checkpoints.

    Reconstructs the actor (and optional normalizer / RNN) from the saved
    state dict so that TorchScript export is not required.
    """

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(obs)

    def reset(self):
        self.model.reset_hidden()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, config: dict, device: torch.device) -> "CheckpointPolicyWrapper":
        """Load a training checkpoint and build an inference model.

        Args:
            checkpoint_path: Path to .pt checkpoint saved by RSL-RL.
            config: Policy YAML config (used for optional ``policy.activation``).
            device: Torch device.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "actor_state_dict" in checkpoint:
            model_state = checkpoint["actor_state_dict"]
        elif "student_state_dict" in checkpoint:
            model_state = checkpoint["student_state_dict"]
        else:
            raise ValueError(
                f"File is not a valid rsl_rl 5.x checkpoint. "
                f"Expected 'actor_state_dict' or 'student_state_dict'. Keys found: {list(checkpoint.keys())}"
            )
        arch = _infer_architecture(model_state)

        # Resolve activation (default: elu).
        policy_config = config.get("policy", {})
        act_name = policy_config.get("activation", "elu")

        # Build inference model.
        model = _CheckpointInferenceModel(arch, act_name)

        # Load actor weights.
        actor_state = {k.replace("mlp.", ""): v for k, v in model_state.items() if k.startswith("mlp.")}
        model.actor.load_state_dict(actor_state)

        # Load RNN weights.
        if arch["is_recurrent"]:
            rnn_state = {k.replace("rnn.rnn.", ""): v for k, v in model_state.items() if k.startswith("rnn.rnn.")}
            model.rnn.load_state_dict(rnn_state)

        # Load normalizer.
        if "obs_normalizer._mean" in model_state:
            model.normalizer = _SimpleNormalizer(
                mean=model_state["obs_normalizer._mean"],
                std=model_state["obs_normalizer._std"],
                eps=1e-2,
            )

        model.eval()
        model.to(device)

        # Diagnostic output.
        iteration = checkpoint.get("iter", "unknown")
        arch_type = f"{arch['rnn_type'].upper()}" if arch["is_recurrent"] else "MLP"
        has_norm = "obs_normalizer._mean" in model_state
        print(f"  Loaded training checkpoint (iteration {iteration})")
        print(f"  Architecture: {arch_type}, hidden dims: {arch['actor_hidden_dims']}")
        print(f"  Num actions: {arch['num_actions']} (noise_std_type: {arch['noise_std_type']})")
        if arch["is_recurrent"]:
            print(
                f"  RNN: {arch['rnn_type'].upper()} (hidden={arch['rnn_hidden_dim']}, layers={arch['rnn_num_layers']})"
            )
        print(f"  Observation normalizer: {'yes' if has_norm else 'no'}")

        return cls(model, device)


class ONNXPolicyWrapper(PolicyWrapper):
    """Wrapper for ONNX policies."""

    def __init__(self, session: Any, device: torch.device):
        super().__init__(session, device)
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        obs_numpy = obs.cpu().numpy()
        if obs_numpy.ndim == 1:
            obs_numpy = obs_numpy[None, :]
        outputs = self.model.run([self.output_name], {self.input_name: obs_numpy})
        return torch.from_numpy(outputs[0]).to(self.device).squeeze(0)

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "ONNXPolicyWrapper":
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(checkpoint_path), providers=providers)
        return cls(session, device)
