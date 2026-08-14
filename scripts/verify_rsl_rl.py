#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the pinned public rsl-rl-lib wheel has AGILE's patch applied."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib
import sys

EXPECTED_VERSION = "5.4.1"


def _ensure_rsl_rl_patch() -> None:
    bootstrap_path = pathlib.Path(__file__).resolve().parents[1] / "agile" / "rl_env" / "rsl_rl" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("agile_rsl_rl_bootstrap", bootstrap_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load AGILE RSL-RL bootstrap from {bootstrap_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_rsl_rl_patch()


def check_rsl_rl() -> bool:
    """Check that rsl_rl resolves to 5.4.1 with AGILE additions."""
    try:
        _ensure_rsl_rl_patch()
        import rsl_rl
        from rsl_rl.models import JitTeacherModel
        from rsl_rl.modules import ReturnVarianceNormalization
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner
    except Exception as exc:
        print(f"FAIL: Failed to apply or import patched rsl_rl additions: {exc}")
        return False

    module_path = pathlib.Path(rsl_rl.__file__).resolve()
    print("OK: rsl_rl imported successfully")
    print(f"  Module location: {module_path}")

    version = importlib.metadata.version("rsl-rl-lib")
    print(f"  Installed rsl-rl-lib version: {version}")
    if version != EXPECTED_VERSION:
        print(f"FAIL: Expected rsl-rl-lib=={EXPECTED_VERSION}")
        return False

    if "agile/algorithms/rsl_rl" in module_path.as_posix():
        print("FAIL: rsl_rl is still imported from the removed vendored package")
        return False

    print(f"OK: Found AGILE JIT teacher model: {JitTeacherModel.__name__}")
    print(f"OK: Found AGILE reward normalizer: {ReturnVarianceNormalization.__name__}")
    print(f"OK: Found rsl_rl runners: {OnPolicyRunner.__name__}, {DistillationRunner.__name__}")
    return True


if __name__ == "__main__":
    if not check_rsl_rl():
        print("\nFAIL: Installation verification failed.")
        sys.exit(1)

    print("\nOK: Installation verified successfully.")
    sys.exit(0)
