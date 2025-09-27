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
    RslRlDistillationAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlStudentTrainedTeacherCfg,
    RslRlSymmetryCfg,
)


@configclass
class G1VelocityHeightPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 250
    experiment_name = "velocity_height_g1_lower"
    run_name = "velocity_height_g1_lower"
    wandb_project = "Velocity-Height-G1-Lower"
    empirical_normalization = False
    enable_entropy_coef_annealing = True
    entropy_coef_annealing_start_progress = 0.2
    enable_entropy_coef_annealing_success_rate = 0.8
    entropy_annealing_decay_rate = 0.9995
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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_G1,
        ),
    )


@configclass
class G1VelocityHeightDistillationRecurrentRunnerCfg(G1VelocityHeightPpoRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10_000
    save_interval = 100
    experiment_name = "velocity_height_g1_lower_distillation"
    run_name = "velocity_height_g1_lower_distillation"
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss_type="mse",
    )
    policy = RslRlStudentTrainedTeacherCfg(
        class_name="StudentTrainedTeacherRecurrent",  # "StudentTrainedTeacher",
        teacher_path="agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt",
        student_hidden_dims=[256, 256, 128],
        activation="elu",
    )


@configclass
class G1VelocityHeightDistillationHistoryRunnerCfg(G1VelocityHeightPpoRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10_000
    save_interval = 100
    experiment_name = "velocity_height_g1_lower_distillation"
    run_name = "velocity_height_g1_lower_distillation"
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss_type="mse",
        # The symmetry will be ignored for distilling recurrent policies.
        symmetry_cfg=RslRlSymmetryCfg(
            use_mirror_loss=True,
            data_augmentation_func=lr_mirror_G1,
            mirror_loss_coeff=0.1,  # Start with a small coefficient
        ),
    )
    policy = RslRlStudentTrainedTeacherCfg(
        class_name="StudentTrainedTeacher",  # "StudentTrainedTeacher",
        teacher_path="agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_teacher.pt",
        student_hidden_dims=[512, 256, 128],
        activation="elu",
    )
