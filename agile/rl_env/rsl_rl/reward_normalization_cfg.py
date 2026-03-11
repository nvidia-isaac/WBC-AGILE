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


from isaaclab.utils import configclass


@configclass
class RslRlRewardNormalizationCfg:
    """Configuration for reward normalization in PPO.

    This normalizer scales rewards to achieve unit variance returns using the analytical
    formula based on the discount factor. For returns G = sum(gamma^t * r_t), if rewards
    have variance sigma^2, then Var(G) ~ sigma^2 / (1 - gamma^2). Dividing rewards by
    sigma / sqrt(1 - gamma^2) gives returns with approximately unit variance.

    Uses exponential moving average (EMA) for variance tracking, allowing adaptation to
    curriculum changes during training.
    """

    decay: float = 0.999
    """EMA decay factor. Higher = slower adaptation. 0.999 = ~693 steps half-life."""

    epsilon: float = 1e-2
    """Epsilon for numerical stability when dividing by standard deviation."""

    return_scale_decay: float | None = 0.999
    """EMA decay for the return-std correction factor that accounts for temporal
    reward correlation. Updated once per rollout (not per step).
    Set to None to disable (i.i.d. assumption only). Default: 0.999."""

    outlier_threshold: float | None = 10.0
    """Number of standard deviations to consider as outlier.
    Samples beyond this threshold are clipped when updating statistics,
    and the normalized output is also clipped to this range.
    Set to None to disable outlier filtering."""
