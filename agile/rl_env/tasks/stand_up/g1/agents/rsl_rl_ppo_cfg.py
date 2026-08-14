# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from isaaclab.utils.configclass import configclass

from agile.rl_env.mdp.events import FallenStateDatasetCfg
from agile.rl_env.mdp.symmetry import lr_mirror_G1  # noqa: F401
from agile.rl_env.rsl_rl import (  # noqa: F401
    RslRlL2C2Cfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRewardNormalizationCfg,
    RslRlSymmetryCfg,
)


@configclass
class G1HeightTrackingPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "height_tracking_g1"
    run_name = "height_tracking_g1"
    wandb_project = "HeightTracking-G1"
    empirical_normalization = False
    enable_entropy_coef_annealing = False
    entropy_coef_annealing_start_progress = 0.2
    enable_entropy_coef_annealing_success_rate = 0.9
    fallen_state_dataset_cfg: FallenStateDatasetCfg | None = FallenStateDatasetCfg(
        spawn_orientation="on_back",
        spawn_pitch_range=(-1.7, -1.4),
        spawn_joint_mode="default",
        initial_lin_vel_range=0.0,
        initial_ang_vel_range=0.0,
    )
    fallen_state_dataset_secondary_cfg: FallenStateDatasetCfg | None = FallenStateDatasetCfg(
        spawn_orientation="random",
        spawn_joint_mode="random",
        initial_lin_vel_range=1.0,
        initial_ang_vel_range=1.0,
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_G1,
        ),
        l2c2_cfg=RslRlL2C2Cfg(
            lambda_actor=1.0,
            lambda_critic=0.1,
        ),
        reward_normalization_cfg=RslRlRewardNormalizationCfg(
            decay=0.999,  # ~693 steps half-life
            epsilon=1e-2,
            return_scale_decay=0.999,
        ),
    )
