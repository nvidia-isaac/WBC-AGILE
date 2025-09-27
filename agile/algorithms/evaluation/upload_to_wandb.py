#!/usr/bin/env python3
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

"""Upload evaluation results to Weights & Biases.

This script uploads:
1. Tracking performance plots as inline images
2. HTML evaluation reports as artifacts
3. Summary metrics as wandb.summary

Usage:
    python agile/algorithms/evaluation/upload_to_wandb.py \\
        --log_dir logs/evaluation/task_datetime \\
        --wandb_project my_project \\
        --wandb_run_name eval_iter5000 \\
        --checkpoint_iteration 5000
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# Import plotting utilities for reusing visualization functions
try:
    from agile.algorithms.evaluation.plotting import (
        calculate_velocity_height_tracking_errors,
        plot_tracking_performance,
    )
except ImportError:
    # Fallback for standalone usage
    from plotting import calculate_velocity_height_tracking_errors, plot_tracking_performance


def upload_velocity_height_to_wandb(
    log_dir: Path,
    wandb_project: str,
    wandb_run_name: str,
    checkpoint_iteration: int = None,
    training_wandb_run: str = None,
):
    """Upload evaluation results to wandb for velocity+height tasks.

    NOTE: This function is specifically designed for velocity+height command environments
    (tasks with lin_vel_x, lin_vel_y, ang_vel_z, base_height commands). It expects
    trajectory data with these specific command fields.

    Args:
        log_dir: Evaluation log directory containing trajectories/ and reports/
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name for this evaluation
        checkpoint_iteration: Checkpoint iteration number
        training_wandb_run: Wandb run path of the training run being evaluated (for tracking)
    """
    import wandb

    # Initialize wandb run
    tags = ["evaluation"]
    if checkpoint_iteration is not None:
        tags.append(f"iter_{checkpoint_iteration}")
    if training_wandb_run:
        # Add training run ID as a tag for easy filtering
        training_run_id = training_wandb_run.split("/")[-1]
        tags.append(f"train_run_{training_run_id}")

    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        job_type="evaluation",
        tags=tags,
    )

    print(f"Uploading to wandb: {wandb_project}/{wandb_run_name}")

    # Load metrics
    metrics_file = log_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Log summary metrics including training run reference (in summary)
        wandb.summary["checkpoint_iteration"] = checkpoint_iteration
        wandb.summary["num_episodes"] = metrics.get("num_environments", 0)
        if training_wandb_run:
            wandb.summary["training_wandb_run"] = training_wandb_run
            print(f"  Linked to training run: {training_wandb_run}")

        # Log success rate in overall group for comparison across runs
        success_rate = metrics.get("success_rate", 0.0)
        wandb.log({"overall/success_rate": success_rate})

        print(f"  ✓ Logged metrics: success_rate={success_rate:.2%}")

    # Generate and log tracking plots for all episodes using plotting utilities
    trajectories_dir = log_dir / "trajectories"
    if trajectories_dir.exists():
        parquet_files = sorted(trajectories_dir.glob("episode_*.parquet"))

        if parquet_files:
            import matplotlib.pyplot as plt
            import numpy as np

            # Collect tracking errors for all episodes
            errors_by_env = {}

            # Plot tracking performance for each episode (each env)
            for ep_file in parquet_files:
                df = pd.read_parquet(ep_file)
                ep_id = df["episode_id"].iloc[0] if "episode_id" in df.columns else 0
                env_id = df["env_id"].iloc[0] if "env_id" in df.columns else 0

                # Calculate tracking errors for this episode
                errors = calculate_velocity_height_tracking_errors(df)
                if errors:
                    errors_by_env[env_id] = errors

                # Generate tracking plot using plotting utility (imported at top)
                fig, _axis = plot_tracking_performance(df)

                if fig is not None:
                    # Log as wandb image in plots group
                    wandb.log(
                        {
                            f"plots/episode_{ep_id}_env_{env_id}": wandb.Image(fig),
                        }
                    )
                    plt.close(fig)

            # Log per-environment tracking error metrics grouped by velocity term
            for env_id, errors in errors_by_env.items():
                for quantity in ["lin_vel_x", "lin_vel_y", "ang_vel_z", "height"]:
                    if quantity in errors:
                        wandb.log(
                            {
                                f"{quantity}/env_{env_id}_mean_error": errors[quantity]["mean"],
                                f"{quantity}/env_{env_id}_std_error": errors[quantity]["std"],
                                f"{quantity}/env_{env_id}_max_error": errors[quantity]["max"],
                                f"{quantity}/env_{env_id}_rms_error": errors[quantity]["rms"],
                            }
                        )

            # Log aggregate metrics (average across all envs) in separate overall group
            if errors_by_env:
                overall_metrics = {}
                for quantity in ["lin_vel_x", "lin_vel_y", "ang_vel_z", "height"]:
                    # Collect mean errors from all envs for this quantity
                    mean_errors = [
                        errors_by_env[env][quantity]["mean"] for env in errors_by_env if quantity in errors_by_env[env]
                    ]
                    if mean_errors:
                        overall_metrics.update(
                            {
                                f"overall/{quantity}_mean_error": np.mean(mean_errors),
                                f"overall/{quantity}_std_across_envs": np.std(mean_errors),
                                f"overall/{quantity}_max_error": np.max(mean_errors),
                            }
                        )

                # Log all overall metrics together
                if overall_metrics:
                    wandb.log(overall_metrics)

            print(f"  ✓ Logged tracking plots and metrics for {len(parquet_files)} episodes (all envs)")

    # Upload HTML report as artifact (zip first for efficiency)
    report_dir = log_dir / "reports"
    if report_dir.exists() and (report_dir / "index.html").exists():
        import shutil

        # Create zip file of the report directory
        zip_path = log_dir / f"evaluation_report_iter{checkpoint_iteration}"
        shutil.make_archive(str(zip_path), "zip", str(report_dir))
        zip_file = Path(f"{zip_path}.zip")

        # Create artifact with the zip file
        artifact = wandb.Artifact(
            name=f"evaluation_report_iter{checkpoint_iteration}",
            type="evaluation_report",
            description=f"Interactive HTML evaluation report for checkpoint iteration {checkpoint_iteration}",
        )

        # Add the zip file
        artifact.add_file(str(zip_file), name="report.zip")

        # Log artifact
        run.log_artifact(artifact)

        # Clean up zip file
        zip_file.unlink()

        print("  ✓ Uploaded HTML report as artifact (zipped)")
        print(f"    View at: {run.url}")

    # Finish run
    run.finish()
    print("  ✓ Wandb upload complete!")


def main():
    parser = argparse.ArgumentParser(description="Upload evaluation results to Weights & Biases")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to evaluation log directory",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=True,
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        required=True,
        help="Wandb run name for this evaluation",
    )
    parser.add_argument(
        "--checkpoint_iteration",
        type=int,
        default=None,
        help="Checkpoint iteration number (optional for local checkpoints)",
    )
    parser.add_argument(
        "--training_wandb_run",
        type=str,
        default=None,
        help="Wandb run path of the training run (for tracking)",
    )

    args = parser.parse_args()

    upload_velocity_height_to_wandb(
        log_dir=Path(args.log_dir),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_iteration=args.checkpoint_iteration,
        training_wandb_run=args.training_wandb_run,
    )


if __name__ == "__main__":
    main()
