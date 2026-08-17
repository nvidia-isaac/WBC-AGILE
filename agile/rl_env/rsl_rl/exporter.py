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


import os

import torch


def export_policy_as_jit(actor_critic: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The rsl_rl 5.x model to export.
        normalizer: Unused; kept only because Isaac Lab calls this helper with the same signature.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    if not hasattr(actor_critic, "as_jit"):
        raise TypeError("Expected an rsl_rl 5.x model with an as_jit() export method.")

    os.makedirs(path, exist_ok=True)
    jit_model = actor_critic.as_jit()
    jit_model.to("cpu")
    torch.jit.script(jit_model).save(os.path.join(path, filename))


def export_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The rsl_rl 5.x model to export.
        normalizer: Unused; kept only because Isaac Lab calls this helper with the same signature.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not hasattr(actor_critic, "as_onnx"):
        raise TypeError("Expected an rsl_rl 5.x model with an as_onnx() export method.")

    os.makedirs(path, exist_ok=True)
    onnx_model = actor_critic.as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
    )
