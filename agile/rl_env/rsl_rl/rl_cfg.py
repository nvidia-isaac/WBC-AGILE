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


from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Literal

from isaaclab.utils.configclass import configclass

from .l2c2_cfg import RslRlL2C2Cfg
from .reward_normalization_cfg import RslRlRewardNormalizationCfg
from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the entire collected trajectories.
    Otherwise, the advantage is normalized over the mini-batches only.
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """

    critic_warmup_steps: int = 0
    """Number of steps to warmup the critic"""

    l2c2_cfg: RslRlL2C2Cfg | None = None
    """The configuration for L2C2 regularization. Default is None, in which case L2C2 is not used."""

    reward_normalization_cfg: RslRlRewardNormalizationCfg | None = None
    """Configuration for reward normalization. Default is None (disabled)."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.
    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    enable_entropy_coef_annealing: bool = False
    """Whether to enable entropy coefficient annealing. Default is False."""

    entropy_coef_annealing_start_progress: float = 0.1
    """The start progress of the entropy coefficient annealing. Default is 0.1."""

    enable_entropy_coef_annealing_success_rate: float = 0.8
    """The success rate for the entropy coefficient to start annealing. Default is 0.8."""

    entropy_annealing_decay_rate: float | None = None
    """The rate of the entropy coefficient annealing.
    If None, decay is linear with respect to the iteration, else exponential with respect to the progress."""

    min_entropy_coef: float = 0.001
    """The minimum value for the entropy coefficient. If the entropy coefficient decays too low, learning can become unstable.
    This threshold is in effect when `entropy_annealing_decay_rate` is set. Otherwise, it is ignored."""

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is
    not empty, then it is appended to the run directory's name, i.e. the logging directory's
    name will become ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

    enable_evaluation: bool = False
    """Whether to enable evaluation during training. Default is False."""

    start_eval_iter: int = 1000
    """The iteration to start evaluation. Default is 1000."""

    load_optimizer: bool = True
    """Whether to load the optimizer. Default is True."""

    def to_rsl_rl_dict(self) -> dict:
        """Return a native rsl_rl 5.x runner dictionary."""
        algorithm = _as_plain_dict(self.algorithm)
        policy = _as_plain_dict(self.policy)
        is_distillation = algorithm.get("class_name") == "Distillation"

        cfg = {
            "class_name": "DistillationRunner" if is_distillation else "OnPolicyRunner",
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "logger": self.logger,
            "neptune_project": self.neptune_project,
            "wandb_project": self.wandb_project,
            "resume": self.resume,
            "load_run": self.load_run,
            "load_checkpoint": self.load_checkpoint,
            "load_optimizer": self.load_optimizer,
            "obs_groups": {},
            "algorithm": algorithm,
            "enable_entropy_coef_annealing": self.enable_entropy_coef_annealing,
            "entropy_coef_annealing_start_progress": self.entropy_coef_annealing_start_progress,
            "enable_entropy_coef_annealing_success_rate": self.enable_entropy_coef_annealing_success_rate,
            "entropy_annealing_decay_rate": self.entropy_annealing_decay_rate,
            "min_entropy_coef": self.min_entropy_coef,
        }

        if is_distillation:
            cfg["student"] = _student_model_cfg(policy, self.empirical_normalization)
            cfg["teacher"] = _jit_teacher_model_cfg(policy)
        else:
            cfg["actor"] = _actor_model_cfg(policy, self.empirical_normalization)
            cfg["critic"] = _critic_model_cfg(policy, self.empirical_normalization)

        return cfg


def rsl_rl_cfg_to_dict(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict:
    """Return a native rsl_rl 5.x runner dictionary for an AGILE configclass."""
    return agent_cfg.to_rsl_rl_dict()


def _as_plain_dict(value):
    if isinstance(value, type(MISSING)):
        return value
    if is_dataclass(value):
        return {field.name: _as_plain_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _as_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_plain_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_as_plain_dict(item) for item in value)
    return value


def _distribution_cfg(policy: dict) -> dict:
    noise_std_type = policy.get("noise_std_type", "scalar")
    if noise_std_type == "pred":
        return {
            "class_name": "HeteroscedasticGaussianDistribution",
            "init_std": policy["init_noise_std"],
            "std_type": "log",
        }
    return {
        "class_name": "GaussianDistribution",
        "init_std": policy["init_noise_std"],
        "std_type": noise_std_type,
    }


def _actor_model_cfg(policy: dict, empirical_normalization: bool) -> dict:
    class_name = "RNNModel" if policy.get("class_name") == "ActorCriticRecurrent" else "MLPModel"
    cfg = {
        "class_name": class_name,
        "hidden_dims": policy["actor_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": empirical_normalization,
        "distribution_cfg": _distribution_cfg(policy),
    }
    if class_name == "RNNModel":
        cfg.update(
            {
                "rnn_type": policy["rnn_type"],
                "rnn_hidden_dim": policy["rnn_hidden_dim"],
                "rnn_num_layers": policy["rnn_num_layers"],
            }
        )
    return cfg


def _critic_model_cfg(policy: dict, empirical_normalization: bool) -> dict:
    class_name = "RNNModel" if policy.get("class_name") == "ActorCriticRecurrent" else "MLPModel"
    cfg = {
        "class_name": class_name,
        "hidden_dims": policy["critic_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": empirical_normalization,
    }
    if class_name == "RNNModel":
        cfg.update(
            {
                "rnn_type": policy["rnn_type"],
                "rnn_hidden_dim": policy["rnn_hidden_dim"],
                "rnn_num_layers": policy["rnn_num_layers"],
            }
        )
    return cfg


def _student_model_cfg(policy: dict, empirical_normalization: bool) -> dict:
    class_name = "RNNModel" if policy.get("class_name") == "StudentTrainedTeacherRecurrent" else "MLPModel"
    cfg = {
        "class_name": class_name,
        "hidden_dims": policy["student_hidden_dims"],
        "activation": policy.get("activation", "elu"),
        "obs_normalization": empirical_normalization,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": policy.get("init_noise_std", 0.1),
            "std_type": policy.get("noise_std_type", "scalar"),
        },
    }
    if class_name == "RNNModel":
        cfg.update(
            {
                "rnn_type": policy.get("rnn_type", "lstm"),
                "rnn_hidden_dim": policy.get("rnn_hidden_dim", 256),
                "rnn_num_layers": policy.get("rnn_num_layers", 1),
            }
        )
    if policy.get("dropout_rate", 0.0) != 0.0:
        cfg["dropout_rate"] = policy["dropout_rate"]
    return cfg


def _jit_teacher_model_cfg(policy: dict) -> dict:
    return {
        "class_name": "JitTeacherModel",
        "teacher_path": policy["teacher_path"],
    }
