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

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable

from .rl_cfg import RslRlOnPolicyRunnerCfg, rsl_rl_cfg_to_dict


def make_rsl_rl_runner(
    env,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    log_dir: str | None = None,
    device: str | None = None,
) -> OnPolicyRunner:
    """Create the configured rsl_rl 5.x runner."""
    runner_cfg = rsl_rl_cfg_to_dict(agent_cfg)
    runner_class = resolve_callable(runner_cfg.pop("class_name", "OnPolicyRunner"))
    return runner_class(env, runner_cfg, log_dir=log_dir, device=device or agent_cfg.device)


def make_rsl_rl_inference_load_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict | None:
    """Load only the student from distillation checkpoints used for inference."""
    if agent_cfg.algorithm.class_name != "Distillation":
        return None

    return {
        "student": True,
        "teacher": False,
        "optimizer": False,
        "iteration": False,
    }


def make_rsl_rl_load_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict | None:
    """Return the rsl_rl 5.x load config for the requested optimizer policy."""
    if agent_cfg.load_optimizer:
        return None

    if agent_cfg.algorithm.class_name == "Distillation":
        return {
            "student": True,
            "teacher": True,
            "optimizer": False,
            "iteration": True,
        }

    return {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": True,
        "rnd": True,
    }
