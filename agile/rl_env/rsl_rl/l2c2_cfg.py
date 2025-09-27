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


@configclass
class RslRlL2C2Cfg:
    """Configuration for the L2C2 regularization in the training. (https://arxiv.org/abs/2202.07152)

    When :meth:`use_l2c2` is True, the :meth:`lambda_actor` and :meth:`lambda_critic` are used to weight the L2C2 regularization loss. This loss is directly added to the agent's loss function.

    For more information, please check the paper: L2C2: Locally Lipschitz Continuous Constraint towards Stable and Smooth Reinforcement Learning
    """

    lambda_actor: float = 1.0
    """The weight for the L2C2 regularization loss for the actor. Default is 1.0."""

    lambda_critic: float = 0.1
    """The weight for the L2C2 regularization loss for the critic. Default is 0.1."""
