# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import torch
from tensordict import TensorDict

from agile.rl_env.rsl_rl_observations import policy_observation, to_rsl_rl_observations


class TestRslRlObservations(unittest.TestCase):
    def test_policy_observation_preserves_batch_dimension(self):
        obs = TensorDict(
            {
                "policy": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "critic": torch.zeros(2, 5),
            },
            batch_size=[2],
        )

        policy_obs = policy_observation(obs)

        self.assertEqual(policy_obs.shape, (2, 3))
        torch.testing.assert_close(policy_obs, obs["policy"])

    def test_grouped_isaac_lab_observations_are_flattened(self):
        obs = to_rsl_rl_observations(
            {
                "policy": {
                    "base": torch.ones(2, 3),
                    "history": torch.ones(2, 2, 4),
                },
                "critic": torch.ones(2, 5),
            },
            num_envs=2,
        )

        self.assertEqual(obs["policy"].shape, (2, 11))
        self.assertEqual(obs["critic"].shape, (2, 5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
