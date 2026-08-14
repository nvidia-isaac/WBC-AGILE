#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit one OSMO training job per catalogued trainable task.

Builds the Docker image once (first task, ``--rebuild``) under a shared ``--image-key`` so every
subsequent task reuses it via ``--use-existing`` -- no repeated image builds. Each task becomes a
separate ``run.py train`` submission (and therefore a separate W&B run).

Defaults to a DRY RUN: it prints the exact ``run.py train`` commands without launching anything.
Pass ``--submit`` to actually submit.

Examples:
    # Show what would be submitted for every task in the table (no jobs launched):
    uv run scripts/submit_all_tasks.py

    # Actually submit a job per task (builds image once, then reuses it):
    uv run scripts/submit_all_tasks.py --submit

    # Only a subset, with custom iterations:
    uv run scripts/submit_all_tasks.py --only Velocity-G1-History-v0 Velocity-Height-G1-History-v0 --max_iterations 30000 --submit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
# The production catalog is the single source of truth for automated training and evaluation.
sys.path.insert(0, str(_REPO_ROOT))
from agile.evaluation.task_catalog import trainable_tasks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit", action="store_true", help="Actually submit. Without this it is a dry run.")
    parser.add_argument("--only", nargs="+", default=None, help="Restrict to these task ids.")
    parser.add_argument("--exclude", nargs="+", default=[], help="Skip these task ids.")
    parser.add_argument("--name-prefix", default="alltasks", help="Experiment-name prefix. Default: alltasks.")
    parser.add_argument(
        "--image-key",
        default="alltasks",
        help="Shared image key so all tasks reuse one image build. Default: alltasks.",
    )
    parser.add_argument("--max_iterations", "-m", type=int, default=50000, help="Iterations per task. Default 50000.")
    parser.add_argument("--seeds", "-s", default=None, help="Seeds passed through to run.py (e.g. '0,1,2').")
    parser.add_argument(
        "--priority",
        default="NORMAL",
        help=(
            "OSMO workflow priority (HIGH, NORMAL, LOW). Default NORMAL = non-preemptible, so jobs "
            "are not evicted mid-training (LOW is preemptible and gets restarted from scratch on "
            "eviction, never accumulating progress). Pass --priority LOW only to burst over the "
            "pool's non-preemptible quota when it is full."
        ),
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Use --use-existing for the first task too (assumes the image-key already exists).",
    )
    return parser.parse_args()


def _experiment_name(prefix: str, task_id: str) -> str:
    """OSMO-safe experiment name: lowercase, non-alphanumerics collapsed to single dashes."""
    slug = "".join(c if c.isalnum() else "-" for c in task_id.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"{prefix}-{slug.strip('-')}"


# Tasks whose scene uses replicate_physics=False (non-instanced geometry per env, required for the
# multi-asset object spawner). Rendering the full scene for the training preview video makes the RTX
# renderer build a per-env acceleration structure, which scales with env count and exhausts GPU
# memory on 48 GB cluster GPUs (OBSERVED: G1-PickPlace at 4096 envs OOMs with VkResult
# ERROR_OUT_OF_DEVICE_MEMORY in omni.rtx). Training itself fits in ~8 GB, and the eval pipeline still
# produces rollout videos, so disable the per-iteration training video for these tasks.
_NO_TRAINING_VIDEO = {"PickPlace-G1-v0"}


def _build_command(args: argparse.Namespace, task_id: str, *, first: bool) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--frozen",
        "run.py",
        "train",
        "--name",
        _experiment_name(args.name_prefix, task_id),
        "--task_name",
        task_id,
        "--image-key",
        args.image_key,
        "--max_iterations",
        str(args.max_iterations),
    ]
    # Build the image exactly once (on the first task) unless told the image-key already exists.
    cmd.append("--use-existing" if (args.no_build or not first) else "--rebuild")
    if args.seeds is not None:
        cmd += ["--seeds", args.seeds]
    if args.priority is not None:
        cmd += ["--priority", args.priority]
    if task_id in _NO_TRAINING_VIDEO:
        # Forwarded to the train workflow's `{% if video %}` guard so it omits --video.
        cmd += ["--set", "video=false"]
    return cmd


def main() -> int:
    args = _parse_args()

    task_ids = [entry.task_id for entry in trainable_tasks()]
    if args.only is not None:
        unknown = [t for t in args.only if t not in task_ids]
        if unknown:
            print(f"[ERROR] --only contains tasks not in the training catalog: {unknown}", file=sys.stderr)
            return 2
        task_ids = [t for t in task_ids if t in args.only]
    task_ids = [t for t in task_ids if t not in set(args.exclude)]

    if not task_ids:
        print("[ERROR] No tasks selected.", file=sys.stderr)
        return 2

    mode = "SUBMITTING" if args.submit else "DRY RUN (pass --submit to launch)"
    print(f"=== {mode}: {len(task_ids)} task(s), image-key='{args.image_key}' ===")
    for i, task_id in enumerate(task_ids):
        cmd = _build_command(args, task_id, first=(i == 0))
        print(f"\n[{i + 1}/{len(task_ids)}] {task_id}")
        print("  " + " ".join(cmd))
        if not args.submit:
            continue
        result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            # The first task builds the shared image; if it fails, later --use-existing tasks would
            # have no image to reuse, so abort. A later task failing is reported but does not block
            # the remaining independent submissions.
            if i == 0:
                print(
                    f"[ERROR] First submission (image build) failed (rc={result.returncode}). Aborting.",
                    file=sys.stderr,
                )
                return 1
            print(f"[WARN] Submission for {task_id} failed (rc={result.returncode}); continuing.", file=sys.stderr)

    if not args.submit:
        print("\n(Dry run only -- nothing was submitted. Re-run with --submit to launch.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
