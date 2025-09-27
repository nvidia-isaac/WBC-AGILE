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


"""Wrappers and utilities to configure an environment for RSL-RL library.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

from .exporter import export_policy_as_jit, export_policy_as_onnx
from .l2c2_cfg import RslRlL2C2Cfg
from .rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg
from .teacher_stundent_distillation_cfg import (
    RslRlDistillationAlgorithmCfg,
    RslRlStudentTrainedTeacherCfg,
)
from .vecenv_wrapper import RslRlVecEnvWrapper

__all__ = [
    "export_policy_as_jit",
    "export_policy_as_onnx",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoActorCriticCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlRndCfg",
    "RslRlSymmetryCfg",
    "RslRlL2C2Cfg",
    "RslRlDistillationAlgorithmCfg",
    "RslRlStudentTrainedTeacherCfg",
    "RslRlVecEnvWrapper",
]
