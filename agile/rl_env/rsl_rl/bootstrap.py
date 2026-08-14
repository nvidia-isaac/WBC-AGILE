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

"""Apply AGILE's RSL-RL compatibility patch for pure uv runs."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib
import shutil
import subprocess
import sys

EXPECTED_RSL_RL_VERSION = "5.4.1"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO_ROOT / "third_party" / "rsl_rl" / "patches" / "rsl_rl_5_4_1_agile.patch"
_PATCH_CHECKED = False


def ensure_rsl_rl_patch() -> None:
    """Ensure the public rsl-rl-lib wheel has AGILE's remaining additions."""
    global _PATCH_CHECKED

    if _PATCH_CHECKED or _is_patch_applied():
        _PATCH_CHECKED = True
        return

    if not _PATCH_FILE.is_file():
        raise RuntimeError(f"Missing AGILE RSL-RL patch: {_PATCH_FILE}")

    rsl_rl_parent = _locate_rsl_rl_parent()
    _remove_stale_jit_teacher_model(rsl_rl_parent)

    command = ["patch", "--forward", "--batch", "-p1", "-i", str(_PATCH_FILE)]
    try:
        subprocess.run(command, check=True, cwd=rsl_rl_parent)
    except FileNotFoundError as exc:
        raise RuntimeError("Applying AGILE's RSL-RL patch requires the `patch` command on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        _purge_rsl_rl_bytecode(rsl_rl_parent)
        if _is_patch_applied():
            _PATCH_CHECKED = True
            return
        raise RuntimeError(f"Failed to apply AGILE's RSL-RL patch: {_PATCH_FILE}") from exc

    _purge_rsl_rl_bytecode(rsl_rl_parent)
    if not _is_patch_applied():
        raise RuntimeError("AGILE RSL-RL patch did not apply cleanly")

    _PATCH_CHECKED = True


def _is_patch_applied() -> bool:
    try:
        if importlib.metadata.version("rsl-rl-lib") != EXPECTED_RSL_RL_VERSION:
            return False
        # Drop any cached rsl_rl modules so the imports below reflect the current
        # on-disk files. Otherwise a pre-patch check that imported the unpatched
        # package leaves it in sys.modules, and this post-patch re-check (and the
        # caller's own import) would keep seeing the stale, unpatched modules.
        _drop_cached_rsl_rl_modules()
        from rsl_rl.models import JitTeacherModel  # noqa: F401
        from rsl_rl.modules import ReturnVarianceNormalization  # noqa: F401
    except Exception:
        return False

    return True


def _drop_cached_rsl_rl_modules() -> None:
    for name in [n for n in sys.modules if n == "rsl_rl" or n.startswith("rsl_rl.")]:
        del sys.modules[name]
    importlib.invalidate_caches()


def _locate_rsl_rl_parent() -> pathlib.Path:
    spec = importlib.util.find_spec("rsl_rl")
    if spec is None or spec.origin is None:
        raise RuntimeError("rsl-rl-lib is not installed. Run this command through uv so dependencies are synced.")

    return pathlib.Path(spec.origin).resolve().parent.parent


def _remove_stale_jit_teacher_model(rsl_rl_parent: pathlib.Path) -> None:
    (rsl_rl_parent / "rsl_rl" / "models" / "jit_teacher_model.py").unlink(missing_ok=True)


def _purge_rsl_rl_bytecode(rsl_rl_parent: pathlib.Path) -> None:
    """Remove bytecode compiled before the in-place source patch was applied."""
    for cache_dir in (rsl_rl_parent / "rsl_rl").rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    importlib.invalidate_caches()
