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

# Import plotting and trajectory_logger directly (no Isaac Sim dependencies)
# Import evaluation scenario components (no Isaac Sim dependencies)
from .eval_config import EnvConfig, EvalConfig, ScheduleStep, SweepConfig
from .plotting import (
    load_all_episodes,
    load_episode,
    load_metadata,
    plot_all_episodes_summary,
    plot_joint_trajectories,
    plot_tracking_performance,
)
from .report_generator import TrajectoryReportGenerator
from .trajectory_logger import TrajectoryLogger


# Lazy import for PolicyEvaluator and VelocityHeightScheduler to avoid requiring Isaac Sim at import time
def __getattr__(name):
    """Lazy import to defer Isaac Sim dependencies until needed."""
    if name == "PolicyEvaluator":
        from .evaluator import PolicyEvaluator

        return PolicyEvaluator
    elif name == "VelocityHeightScheduler":
        from .velocity_height_scheduler import VelocityHeightScheduler

        return VelocityHeightScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PolicyEvaluator",
    "TrajectoryLogger",
    "TrajectoryReportGenerator",
    "VelocityHeightScheduler",
    "EvalConfig",
    "EnvConfig",
    "ScheduleStep",
    "SweepConfig",
    "load_episode",
    "load_all_episodes",
    "load_metadata",
    "plot_joint_trajectories",
    "plot_tracking_performance",
    "plot_all_episodes_summary",
]
