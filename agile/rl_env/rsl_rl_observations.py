# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation helpers shared by RSL-RL scripts and wrappers."""

from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase


def to_rsl_rl_observations(obs_dict: dict | TensorDictBase, num_envs: int) -> TensorDict:
    """Convert Isaac Lab grouped observations into rsl_rl 5.x observation tensors."""
    return TensorDict(
        {name: flatten_observation_group(group_obs) for name, group_obs in obs_dict.items()},
        batch_size=[num_envs],
    )


def flatten_observation_group(group_obs: torch.Tensor | dict | TensorDictBase) -> torch.Tensor:
    """Flatten one grouped observation into ``[num_envs, obs_dim]``."""
    if isinstance(group_obs, torch.Tensor):
        if group_obs.ndim == 1:
            return group_obs.unsqueeze(-1)
        return group_obs.reshape(group_obs.shape[0], -1)

    if isinstance(group_obs, dict | TensorDictBase):
        parts = [flatten_observation_group(obs) for obs in group_obs.values()]
        if not parts:
            raise ValueError("Observation groups must contain at least one tensor.")
        return torch.cat(parts, dim=-1)

    raise TypeError(f"Unsupported observation type: {type(group_obs)}")


def policy_observation(obs: torch.Tensor | dict | TensorDictBase, obs_group: str = "policy") -> torch.Tensor:
    """Return the flattened observation tensor consumed by exported policies."""
    if isinstance(obs, torch.Tensor):
        return flatten_observation_group(obs)

    if isinstance(obs, dict | TensorDictBase):
        if obs_group not in obs.keys():
            raise KeyError(f"Observation group '{obs_group}' not found in observations: {list(obs.keys())}")
        return flatten_observation_group(obs[obs_group])

    raise TypeError(f"Unsupported observation type: {type(obs)}")
