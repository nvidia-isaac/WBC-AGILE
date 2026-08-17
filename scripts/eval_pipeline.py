#!/usr/bin/env python3
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

"""Run eval (+video) -> LEAPP export -> sim2sim (+video) -> report for ONE task.

Checkpoint source is explicit:
  --checkpoint <local .pt>    use directly
  --wandb_run <team/project/run> plus one exact checkpoint selector

Both eval.py and export_policy_leapp.py accept only --checkpoint (local path), so the W&B
source is resolved to a local path before any downstream command is invoked.

Writes <output-dir>/<task>/ with:
  eval/             Isaac Lab eval logs + metrics.json
  leapp/            LEAPP bundle
  checkpoint/videos/play/rl-video-*.mp4  Isaac Lab rollout (written by eval.py next to checkpoint)
  videos/sim2sim.mp4  MuJoCo rollout
  eval/reports/index.html  embedded-video HTML report
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agile.evaluation.evaluation_manifest import load_evaluation_spec, validate_run_label  # noqa: E402
from agile.evaluation.run_status import write_status  # noqa: E402
from agile.evaluation.task_catalog import TASK_CATALOG  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="Task ID, e.g. Velocity-Height-G1-History-v0")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="Local .pt checkpoint path")
    src.add_argument(
        "--wandb_run",
        type=str,
        metavar="TEAM/PROJECT/RUN",
        help="W&B run id; requires one exact checkpoint selector below.",
    )
    p.add_argument("--wandb-checkpoint-file", help="Exact checkpoint file in --wandb-run.")
    p.add_argument("--wandb-iteration", type=int, help="Exact model_<iteration>.pt in --wandb-run.")
    p.add_argument("--wandb-artifact-version", help="Immutable W&B artifact version.")
    p.add_argument("--mjcf", type=Path, required=True, help="MuJoCo MJCF scene path for sim2sim")
    p.add_argument("--output-dir", type=Path, required=True, help="Root output directory")
    p.add_argument("--run-label", required=True, help="Unique label for this evaluation run")
    p.add_argument("--eval-config", type=Path, default=None, help="Path to YAML eval scenario config")
    p.add_argument("--evaluation-spec", type=Path, default=None, help="Task evaluation specification override")
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Rollout duration in seconds; sets both the eval video length and the sim2sim duration.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = p.parse_args()
    selectors = [args.wandb_checkpoint_file, args.wandb_iteration, args.wandb_artifact_version]
    if args.wandb_run is not None and sum(value is not None for value in selectors) != 1:
        p.error(
            "--wandb_run requires exactly one exact checkpoint selector: --wandb-checkpoint-file, --wandb-iteration, or --wandb-artifact-version"
        )
    if args.checkpoint is not None and any(value is not None for value in selectors):
        p.error("W&B checkpoint selectors require --wandb_run")
    try:
        validate_run_label(args.run_label)
    except ValueError as exc:
        p.error(str(exc))
    return args


def _resolve_checkpoint(args: argparse.Namespace, checkpoint_dir: Path) -> Path:
    """Return a local checkpoint path, downloading from W&B if needed.

    In --dry-run mode the W&B path is NOT contacted; a labelled placeholder is returned instead.
    """
    if args.checkpoint is not None:
        return args.checkpoint

    # W&B source
    if args.dry_run:
        # Return a clearly-labelled placeholder so the printed commands are readable.
        return Path(f"<downloaded-from:{args.wandb_run}>")

    import wandb  # imported lazily to avoid overhead in dry-run / non-W&B cases

    print(f"[eval_pipeline] Downloading checkpoint from W&B run: {args.wandb_run}", flush=True)
    api = wandb.Api()
    run = api.run(args.wandb_run)

    if args.wandb_artifact_version is not None:
        artifact = api.artifact(args.wandb_artifact_version)
        artifact_root = Path(artifact.download(root=str(checkpoint_dir)))
        checkpoints = sorted(artifact_root.rglob("*.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError(
                f"W&B artifact {args.wandb_artifact_version!r} must contain exactly one checkpoint, found {len(checkpoints)}"
            )
        return checkpoints[0]
    expected_name = args.wandb_checkpoint_file or f"model_{args.wandb_iteration}.pt"
    matches = [f for f in run.files() if f.name == expected_name]
    if len(matches) != 1:
        raise RuntimeError(f"Exact checkpoint {expected_name!r} was not found in W&B run {args.wandb_run}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    matches[0].download(root=str(checkpoint_dir), replace=True)
    local_path = checkpoint_dir / matches[0].name
    print(f"[eval_pipeline] Downloaded: {local_path}", flush=True)
    return local_path


def _headless_mujoco_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    return env


def _run(cmd: list[str], dry: bool, run_dir: Path, stage: str, env: dict[str, str] | None = None) -> None:
    print("  " + " ".join(str(c) for c in cmd), flush=True)
    write_status(run_dir, stage, "running")
    if not dry:
        try:
            subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
        except subprocess.CalledProcessError as exc:
            write_status(run_dir, stage, "failed", str(exc))
            raise
    write_status(run_dir, stage, "succeeded")


def _required_videos(eval_video_dir: Path, sim2sim_video: Path) -> tuple[Path, Path]:
    """Return the two rollout videos required for a successful report."""
    isaac_videos = sorted(eval_video_dir.glob("*.mp4"))
    if not isaac_videos:
        raise RuntimeError(f"Isaac Lab rollout video was not produced under {eval_video_dir}")
    if not sim2sim_video.is_file():
        raise RuntimeError(f"MuJoCo rollout video was not produced at {sim2sim_video}")
    return isaac_videos[0], sim2sim_video


def _required_eval_video(eval_video_dir: Path) -> Path:
    """Return the Isaac Lab rollout video required for a report without Sim2MuJoCo."""
    isaac_videos = sorted(eval_video_dir.glob("*.mp4"))
    if not isaac_videos:
        raise RuntimeError(f"Isaac Lab rollout video was not produced under {eval_video_dir}")
    return isaac_videos[0]


def _metric_suite_args(metric_suite: str | None) -> list[str]:
    """Map each supported metric suite to its evaluator invocation."""
    if metric_suite == "motion_tracking":
        return ["--run_evaluation", "--save_trajectories"]
    raise ValueError(f"unsupported metric_suite: {metric_suite}")


def _validate_leapp_bundle(bundle_yaml: Path) -> None:
    """Fail early if LEAPP export did not produce a runnable bundle."""
    if not bundle_yaml.is_file():
        raise RuntimeError(f"LEAPP YAML was not produced: {bundle_yaml}")

    desc = yaml.safe_load(bundle_yaml.read_text())
    if not isinstance(desc, dict):
        raise RuntimeError(f"LEAPP YAML is not a mapping: {bundle_yaml}")

    try:
        frequency = float(desc["pipeline"]["configs"]["frequency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"LEAPP bundle is missing positive pipeline.configs.frequency: {bundle_yaml}") from exc
    if frequency <= 0.0:
        raise RuntimeError(f"LEAPP bundle is missing positive pipeline.configs.frequency: {bundle_yaml}")

    models = desc.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError(f"LEAPP bundle has no models: {bundle_yaml}")
    for model_name, model_desc in models.items():
        params = model_desc.get("parameters", {}) if isinstance(model_desc, dict) else {}
        model_path = params.get("model_path")
        if not model_path:
            raise RuntimeError(f"LEAPP model {model_name!r} has no model_path: {bundle_yaml}")
        resolved_model_path = bundle_yaml.parent / model_path
        if not resolved_model_path.is_file():
            raise RuntimeError(f"LEAPP model file does not exist: {resolved_model_path}")

    initial_values = desc.get("pipeline", {}).get("initial_values")
    if initial_values is not None and not (bundle_yaml.parent / initial_values).is_file():
        raise RuntimeError(f"LEAPP initial values file does not exist: {bundle_yaml.parent / initial_values}")


def main() -> int:
    a = _parse_args()
    catalog_entry = next((entry for entry in TASK_CATALOG if entry.task_id == a.task), None)
    spec_path = a.evaluation_spec or (catalog_entry.evaluation_spec if catalog_entry else None)
    if spec_path is None:
        raise SystemExit(f"[eval_pipeline] task has no evaluation specification: {a.task}")
    spec = load_evaluation_spec(Path(spec_path))
    scenario = a.eval_config or spec.scenario
    sim2mujoco_scenario = spec.sim2mujoco_scenario or scenario

    task_dir = a.output_dir / a.task / a.run_label
    checkpoint_dir = task_dir / "checkpoint"
    eval_log = task_dir / "eval"
    bundle_dir = task_dir / "leapp"
    videos_dir = task_dir / "videos"
    sim2sim_video = videos_dir / "sim2sim.mp4"
    py = sys.executable

    print(f"=== eval_pipeline: {a.task} ===", flush=True)

    # Resolve checkpoint to a local path once; both eval and export require --checkpoint.
    local_ckpt = _resolve_checkpoint(a, checkpoint_dir)

    # Stage 1: Isaac Lab eval with video + trajectory save
    print("\n[Stage 1/4] Isaac Lab eval", flush=True)
    eval_cmd = [
        py,
        "scripts/eval.py",
        "--task",
        a.task,
        "--checkpoint",
        str(local_ckpt),
        "--video",
        "--video_length_s",
        str(a.duration),
        "--headless",
    ]
    if not spec.video_only:
        eval_cmd += [*_metric_suite_args(spec.metric_suite), "--metrics_file", str(eval_log / "metrics.json")]
    if spec.fail_on_non_timeout_dones:
        eval_cmd += ["--fail_on_non_timeout_dones"]
    if spec.non_timeout_done_warmup_steps:
        eval_cmd += ["--non_timeout_done_warmup_steps", str(spec.non_timeout_done_warmup_steps)]
    if scenario is not None:
        eval_cmd += ["--eval_config", str(scenario)]
    _run(eval_cmd, a.dry_run, task_dir, "isaac_eval")
    # Hard-fail if eval produced no metrics. The evaluation can fail internally (e.g. no
    # observations) without eval.py necessarily surfacing a non-zero exit, and continuing would
    # produce a report with no eval data + no eval video.
    if not a.dry_run and not spec.video_only and not (eval_log / "metrics.json").is_file():
        write_status(task_dir, "isaac_eval", "failed", "requested metrics file was not produced")
        raise SystemExit(
            f"[eval_pipeline] Stage 1 (eval) produced no metrics at {eval_log / 'metrics.json'} — "
            "evaluation failed; aborting before export/sim2sim."
        )

    # Stage 2: LEAPP export (always needs a local checkpoint path)
    print("\n[Stage 2/4] LEAPP export", flush=True)
    _run(
        [
            py,
            "scripts/export_policy_leapp.py",
            "--task",
            a.task,
            "--checkpoint",
            str(local_ckpt),
            "--export_save_path",
            str(bundle_dir),
            # Headless/automated: never pop the interactive graph window (it blocks on window close).
            "--disable_graph_visualization",
            *([] if not spec.video_only else ["--validation_steps", "0"]),
        ],
        a.dry_run,
        task_dir,
        "leapp_export",
    )

    # Stage 3: sim2sim evaluation with video
    bundle_yaml = bundle_dir / a.task / f"{a.task}.yaml"
    if not a.dry_run:
        try:
            _validate_leapp_bundle(bundle_yaml)
        except RuntimeError as exc:
            write_status(task_dir, "leapp_export", "failed", str(exc))
            raise SystemExit(f"[eval_pipeline] {exc}") from exc

    if spec.sim2mujoco:
        print("\n[Stage 3/4] Sim2MuJoCo eval", flush=True)
        sim_cmd = [
            py,
            "scripts/sim2mujoco_eval.py",
            "--leapp-yaml",
            str(bundle_yaml),
            "--mjcf",
            str(a.mjcf),
            "--no-viewer",
            "--device",
            "cpu",
            "--no-real-time",
            "--video",
            str(sim2sim_video),
        ]
        if spec.sim2mujoco_scenario is None:
            sim_cmd += ["--duration", str(a.duration)]
        if sim2mujoco_scenario is not None:
            # MuJoCo has one environment. Use the first declared Isaac evaluation scenario for its
            # representative sim2sim rollout, rather than silently ignoring the multi-env config.
            sim_cmd += ["--eval-config", str(sim2mujoco_scenario), "--eval-env-id", "0"]
        _run(sim_cmd, a.dry_run, task_dir, "sim2sim", env=_headless_mujoco_env())
    else:
        print("\n[Stage 3/4] Sim2MuJoCo eval skipped by evaluation spec", flush=True)
        write_status(task_dir, "sim2sim", "skipped", "disabled by evaluation spec")

    # Stage 4: report with both videos embedded.
    # eval.py writes its recording to <checkpoint_parent>/videos/play/rl-video-*.mp4,
    # so we resolve from local_ckpt rather than using a hardcoded path.
    print("\n[Stage 4/4] Generate report", flush=True)
    eval_video_dir = Path(local_ckpt).parent / "videos" / "play"
    report_cmd = [
        py,
        "agile/algorithms/evaluation/generate_report.py",
        "--log_dir",
        str(eval_log),
        "--output-dir",
        str(eval_log / "reports"),
        "--no-browser",
        "--task-name",
        str(a.task),
    ]
    if a.dry_run:
        # Stage 1 didn't run, so no files exist to glob; show a readable placeholder.
        report_cmd += ["--eval_video", str(eval_video_dir / "eval-video.mp4")]
    else:
        try:
            if spec.sim2mujoco:
                eval_video, _ = _required_videos(eval_video_dir, sim2sim_video)
            else:
                eval_video = _required_eval_video(eval_video_dir)
        except RuntimeError as exc:
            write_status(task_dir, "report", "failed", str(exc))
            raise SystemExit(f"[eval_pipeline] {exc}") from exc
        report_cmd += ["--eval_video", str(eval_video)]
    if spec.sim2mujoco:
        report_cmd += ["--sim2sim_video", str(sim2sim_video)]
    _run(report_cmd, a.dry_run, task_dir, "report")

    print(f"\nreport: {eval_log / 'reports' / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
