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


from isaaclab.utils import configclass

from agile.rl_env.mdp.symmetry import (
    lr_mirror_G1,
)  # noqa: F401
from agile.rl_env.rsl_rl import (  # noqa: F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class DummyPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "standing_g1"
    empirical_normalization = False
    enable_entropy_coef_annealing = False
    entropy_coef_annealing_start_progress = 0.2
    enable_entropy_coef_annealing_success_rate = 0.9
    enable_evaluation = True
    start_eval_iter = 300
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.0,
        schedule="",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=False,
            mirror_loss_coeff=1.0,
            data_augmentation_func=lr_mirror_G1,
        ),
    )
