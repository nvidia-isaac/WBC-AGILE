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

"""Generate interactive HTML evaluation reports from trajectory data.

This script is standalone and works without Isaac Sim.

Usage:
    # Generate report for all episodes
    python agile/algorithms/evaluation/generate_report.py \
        --log_dir logs/evaluation/Velocity-Height-G1-Dev-v0_20251010_214925

    # Generate for specific episodes
    python agile/algorithms/evaluation/generate_report.py \
        --log_dir logs/evaluation/task_datetime \
        --episodes 0,3,5,7

    # Generate for failed episodes only
    python agile/algorithms/evaluation/generate_report.py \
        --log_dir logs/evaluation/task_datetime \
        --episodes failed

    # Don't open browser automatically
    python agile/algorithms/evaluation/generate_report.py \
        --log_dir logs/evaluation/task_datetime \
        --no-browser
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for direct imports
sys.path.insert(0, str(Path(__file__).parent))

from report_generator import TrajectoryReportGenerator


def parse_episode_arg(episodes_str: str) -> list[int] | str:
    """Parse episode argument."""
    if episodes_str in ["all", "success", "failed"]:
        return episodes_str

    # Parse comma-separated list
    try:
        return [int(x.strip()) for x in episodes_str.split(",")]
    except ValueError:
        raise ValueError(
            f"Invalid episodes argument: {episodes_str}. Must be 'all', 'success', 'failed', or comma-separated IDs."
        ) from None


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML reports from trajectory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to evaluation directory containing trajectories/",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="all",
        help="Which episodes to include: 'all', 'success', 'failed', or comma-separated IDs (e.g., '0,3,5')",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open report in browser",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for reports (default: log_dir/reports)",
    )

    args = parser.parse_args()

    # Parse episode specification
    episode_ids = parse_episode_arg(args.episodes)

    # Create report generator
    print("Initializing report generator...")
    print(f"  Trajectory directory: {args.log_dir}")

    generator = TrajectoryReportGenerator(
        trajectory_dir=args.log_dir,
        output_dir=args.output_dir,
        plot_backend="plotly",
    )

    # Generate report
    print(f"\nGenerating reports for episodes: {episode_ids}")
    index_path = generator.generate_full_report(
        episode_ids=episode_ids,
        include_all_joints=True,
        open_browser=not args.no_browser,
    )

    print("\n✓ Done! Report available at:")
    print(f"  {index_path.absolute()}")

    if not args.no_browser:
        print("\n→ Report should open in your default browser")
    else:
        print("\nTo view the report, open the file in your browser:")
        print(f"  file://{index_path.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nReport generation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error generating report: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
