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

from __future__ import annotations

from typing import Literal

from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass


@configclass
class DoneTermCfg(TerminationTermCfg):
    """Extended termination term with RL training handling metadata.

    Extends Isaac Lab's ``TerminationTermCfg`` with metadata that tells the
    training algorithm how to handle each termination during learning:

    - **neutral** (default): No special handling. The episode ends and GAE
      bootstraps the value to 0. This is the standard behavior.
    - **bad**: The termination is undesirable (e.g., falling, crashing). The
      algorithm bootstraps the value (making it value-neutral, like timeouts)
      and then subtracts ``sigma`` as a penalty in normalized reward space.
    - **good**: The termination is desirable (e.g., reaching a goal). Same
      bootstrap, but ``sigma`` is added as a bonus.

    Since ``sigma`` operates post-normalization, it is scale-invariant and
    does not require manual tuning per reward scale.

    This class is fully backward-compatible with ``TerminationTermCfg`` —
    existing terms default to ``neutral`` with no special handling.
    """

    termination_type: Literal["neutral", "good", "bad"] = "neutral"
    """How the training algorithm should handle this termination.

    - ``"neutral"``: Default behavior, no bootstrap or penalty.
    - ``"bad"``: Bootstrap + penalty of ``sigma`` (termination is sigma worse than continuing).
    - ``"good"``: Bootstrap + bonus of ``sigma`` (termination is sigma better than continuing).
    """

    sigma: float = 5.0
    """Post-normalization penalty/bonus magnitude applied on good/bad terminations.

    Only used when ``termination_type`` is ``"good"`` or ``"bad"``.
    Typical values: 2.0-5.0. Since this operates in normalized reward space,
    it is scale-invariant.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.time_out and self.termination_type != "neutral":
            raise ValueError(
                f"Timeout terminations must use termination_type='neutral' (got "
                f"'{self.termination_type}'). Timeouts already bootstrap the value "
                f"estimate; applying good/bad handling would conflict with this."
            )
