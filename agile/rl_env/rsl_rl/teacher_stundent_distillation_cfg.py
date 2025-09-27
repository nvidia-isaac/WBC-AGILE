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

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlStudentTrainedTeacherCfg:
    """Configuration for the student trained teacher networks."""

    class_name: str = "StudentTrainedTeacher"
    """The policy class name. Default is StudentTrainedTeacher."""

    teacher_path: str = MISSING
    """The path to the jit exported teacher model."""

    student_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the student network."""

    activation: str = "elu"
    """The activation function for the student network."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""

    class_name: str = "Distillation"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    gradient_length: int = MISSING
    """The length of the gradient."""

    learning_rate: float = MISSING
    """The learning rate for the distillation algorithm."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    loss_type: Literal["mse", "l1"] = "mse"
    """The type of loss function to use."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Configuration for symmetry-based training. Default is None.

    Note: Symmetry loss is only supported for non-recurrent policies.
    If a recurrent policy is detected, symmetry will be automatically disabled."""
