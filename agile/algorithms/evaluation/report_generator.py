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


"""Interactive HTML report generator for trajectory analysis.

This module is standalone and works without Isaac Sim.
Only requires: pandas, plotly, jinja2
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from jinja2 import Template

# Add evaluation directory to path to import plotting.py directly
sys.path.insert(0, str(Path(__file__).parent))

# Import plotting functions directly (no relative imports for standalone usage)
try:
    # Try absolute import first (when used as module)
    from agile.algorithms.evaluation.plotting import (
        calculate_tracking_errors,
        load_episode,
        load_metadata,
    )
except ImportError:
    # Fall back to direct import (when run standalone)
    from plotting import calculate_tracking_errors, load_episode, load_metadata


class TrajectoryReportGenerator:
    """Generate interactive HTML reports from trajectory data.

    Completely standalone - no Isaac Sim dependencies.
    Works with saved parquet files and metadata.
    """

    def __init__(
        self,
        trajectory_dir: str | Path,
        output_dir: str | Path | None = None,
        plot_backend: str = "plotly",
    ):
        """Initialize report generator.

        Args:
            trajectory_dir: Directory containing trajectory parquet files and metadata.json
            output_dir: Where to save reports. Defaults to trajectory_dir/../reports
            plot_backend: 'plotly' for interactive or 'matplotlib' for static (plotly recommended)
        """
        self.trajectory_dir = Path(trajectory_dir)

        # Handle both direct trajectories/ dir and parent dir
        if (self.trajectory_dir / "trajectories").exists():
            self.trajectory_dir = self.trajectory_dir / "trajectories"

        # Set output directory
        if output_dir is None:
            self.output_dir = self.trajectory_dir.parent / "reports"
        else:
            self.output_dir = Path(output_dir)

        self.plot_backend = plot_backend

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "episodes").mkdir(exist_ok=True)

        # Load metadata
        self.metadata = self._load_metadata()

        # Get list of available episodes
        self.available_episodes = self._get_available_episodes()

    def _load_metadata(self) -> dict:
        """Load metadata.json file."""
        try:
            # Pass parent directory since load_metadata handles trajectories/ subfolder
            metadata = load_metadata(self.trajectory_dir.parent)
            if not metadata:
                print("Warning: Could not load metadata. Some features may be limited.")
                return {}
            return metadata
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
            return {}

    def _get_available_episodes(self) -> list[int]:
        """Get list of available episode IDs."""
        parquet_files = sorted(self.trajectory_dir.glob("episode_*.parquet"))
        episode_ids = []
        for f in parquet_files:
            # Extract episode ID from filename
            try:
                ep_id = int(f.stem.split("_")[1])
                episode_ids.append(ep_id)
            except (IndexError, ValueError):
                continue
        return sorted(episode_ids)

    def generate_full_report(
        self,
        episode_ids: list[int] | str = "all",
        include_all_joints: bool = True,
        open_browser: bool = True,
    ) -> Path:
        """Generate complete HTML report with index and episode pages.

        Args:
            episode_ids: Which episodes to include. Options:
                        - 'all': All episodes
                        - 'success': Only successful episodes
                        - 'failed': Only failed episodes
                        - list of ints: Specific episodes
            include_all_joints: Whether to include all joints or just key joints
            open_browser: Whether to automatically open the report in browser

        Returns:
            Path to the generated index.html file

        Example:
            >>> generator = TrajectoryReportGenerator("logs/evaluation/task_datetime")
            >>> index_path = generator.generate_full_report(episode_ids='all')
        """
        # Determine which episodes to process
        episodes_to_process = self._resolve_episode_list(episode_ids)

        print(f"Generating report for {len(episodes_to_process)} episodes...")

        # Generate individual episode pages
        episode_summaries = []
        for ep_id in episodes_to_process:
            print(f"  Processing episode {ep_id}...")
            ep_summary = self.generate_episode_page(ep_id, include_all_joints=include_all_joints)
            episode_summaries.append(ep_summary)

        # Generate index page
        print("Generating index page...")
        index_path = self.generate_index_page(episode_summaries)

        print("\n✓ Report generated successfully!")
        print(f"  Location: {index_path}")
        print(f"  Episodes: {len(episodes_to_process)}")

        # Open in browser
        if open_browser:
            webbrowser.open(f"file://{index_path.absolute()}")
            print("\n→ Opening report in browser...")

        return index_path

    def _resolve_episode_list(self, episode_ids: list[int] | str) -> list[int]:
        """Convert episode specification to list of episode IDs."""
        if isinstance(episode_ids, list):
            return [ep for ep in episode_ids if ep in self.available_episodes]

        if episode_ids == "all":
            return self.available_episodes

        # Filter by success/failure
        filtered = []
        for ep_id in self.available_episodes:
            try:
                df = load_episode(self.trajectory_dir, ep_id)
                is_success = df["is_success"].iloc[0]

                if episode_ids == "success" and is_success:
                    filtered.append(ep_id)
                elif episode_ids == "failed" and not is_success:
                    filtered.append(ep_id)
            except Exception:
                continue

        return filtered

    def generate_index_page(self, episode_summaries: list[dict]) -> Path:
        """Generate main summary index.html page.

        Args:
            episode_summaries: List of episode summary dictionaries

        Returns:
            Path to generated index.html
        """
        # Calculate overall statistics
        total_episodes = len(episode_summaries)
        successful = sum(1 for ep in episode_summaries if ep["is_success"])
        success_rate = (successful / total_episodes * 100) if total_episodes > 0 else 0

        # Generate summary plot (tracking errors across all episodes)
        summary_plot_html = self._generate_summary_tracking_plot(episode_summaries)

        # Render HTML
        html_content = self._render_index_template(
            task_name=self.metadata.get("task_name", "Unknown Task"),
            total_episodes=total_episodes,
            successful_episodes=successful,
            success_rate=success_rate,
            episodes=episode_summaries,
            summary_plot=summary_plot_html,
        )

        # Save to file
        index_path = self.output_dir / "index.html"
        with open(index_path, "w") as f:
            f.write(html_content)

        return index_path

    def generate_episode_page(self, episode_id: int, include_all_joints: bool = True) -> dict:
        """Generate detailed HTML page for one episode.

        Args:
            episode_id: Episode ID to generate page for
            include_all_joints: Whether to include all joints or just key joints

        Returns:
            Dictionary with episode summary info for index page
        """
        # Load episode data
        df = load_episode(self.trajectory_dir, episode_id)

        # Extract summary info
        is_success = df["is_success"].iloc[0]
        env_id = df["env_id"].iloc[0]
        duration = df["timestep"].max()
        num_frames = len(df)

        # Calculate tracking errors if available
        tracking_errors = self._calculate_tracking_errors(df)

        # Generate plots
        tracking_plot_html = self._generate_tracking_plot(df, episode_id)
        joint_plots_html = self._generate_joint_plots(df, episode_id, include_all_joints)

        # Render episode page
        html_content = self._render_episode_template(
            episode_id=episode_id,
            env_id=env_id,
            is_success=is_success,
            duration=duration,
            num_frames=num_frames,
            tracking_errors=tracking_errors,
            tracking_plot=tracking_plot_html,
            joint_plots=joint_plots_html,
            task_name=self.metadata.get("task_name", "Unknown"),
        )

        # Save to file
        episode_path = self.output_dir / "episodes" / f"episode_{episode_id:03d}.html"
        with open(episode_path, "w") as f:
            f.write(html_content)

        # Return summary for index page
        return {
            "id": episode_id,
            "env_id": env_id,
            "is_success": is_success,
            "duration": duration,
            "num_frames": num_frames,
            "tracking_errors": tracking_errors,
            "status_class": "success" if is_success else "failed",
            "status_icon": "✓" if is_success else "✗",
        }

    def _calculate_tracking_errors(self, df: pd.DataFrame) -> dict:
        """Calculate tracking error statistics."""
        # Reuse utility function from plotting module
        return calculate_tracking_errors(df)

    def _generate_tracking_plot(self, df: pd.DataFrame, episode_id: int) -> str:
        """Generate plotly tracking performance plot (dynamic for variable command structures)."""
        command_prefix = (
            "commands" if "commands_0" in df.columns else ("command" if "command_0" in df.columns else None)
        )

        if not command_prefix:
            return "<p>Command data not available for this episode.</p>"

        timestep = df["timestep"]

        # Dynamically detect available command fields and their corresponding state variables
        quantities = []
        cmd_idx = 0

        # Common command field mappings (in order)
        # Format: (cmd_idx, actual_col, fallback_col, unit, label)
        potential_mappings = [
            (0, "root_lin_vel_robot_0", "root_lin_vel_0", "m/s", "Linear Velocity X"),
            (1, "root_lin_vel_robot_1", "root_lin_vel_1", "m/s", "Linear Velocity Y"),
            (2, "root_ang_vel_2", None, "rad/s", "Angular Velocity Z"),
            (3, "root_pos_2", None, "m", "Height"),
        ]

        for cmd_idx, actual_col, fallback_col, unit, label in potential_mappings:
            cmd_col = f"{command_prefix}_{cmd_idx}"

            # Check if this command field exists
            if cmd_col not in df.columns:
                break  # No more command fields

            # Determine actual state column to use
            state_col = None
            if actual_col and actual_col in df.columns:
                state_col = actual_col
            elif fallback_col and fallback_col in df.columns:
                state_col = fallback_col

            if state_col:
                quantities.append((state_col, cmd_col, unit, label))

        if not quantities:
            return "<p>No trackable command-state pairs found in this episode.</p>"

        # Create subplots with dynamic number of rows
        subplot_titles = tuple(q[3] for q in quantities)  # Extract labels
        fig = sp.make_subplots(
            rows=len(quantities),
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
        )

        # Add traces for each tracking quantity
        for idx, (actual_col, cmd_col, unit, _label) in enumerate(quantities):
            row = idx + 1  # plotly rows are 1-indexed

            if actual_col in df.columns and cmd_col in df.columns:
                # Actual
                fig.add_trace(
                    go.Scatter(
                        x=timestep,
                        y=df[actual_col],
                        name="Actual",
                        line={"color": "blue", "width": 2},
                        showlegend=(row == 1),
                    ),
                    row=row,
                    col=1,
                )
                # Commanded
                fig.add_trace(
                    go.Scatter(
                        x=timestep,
                        y=df[cmd_col],
                        name="Commanded",
                        line={"color": "red", "width": 2, "dash": "dash"},
                        showlegend=(row == 1),
                    ),
                    row=row,
                    col=1,
                )

                fig.update_yaxes(title_text=unit, row=row, col=1)

        # Add x-axis label to bottom row (dynamic)
        fig.update_xaxes(title_text="Time (s)", row=len(quantities), col=1)

        # Dynamic height based on number of command fields
        plot_height = max(600, len(quantities) * 250)
        fig.update_layout(
            height=plot_height, title_text=f"Tracking Performance - Episode {episode_id}", hovermode="x unified"
        )

        return fig.to_html(include_plotlyjs=False)

    def _generate_joint_plots(self, df: pd.DataFrame, episode_id: int, include_all: bool) -> dict:
        """Generate joint plots organized by groups.

        Returns:
            Dictionary mapping group names to HTML strings.
            Groups are determined from metadata or default to single group.
        """
        joint_names = self.metadata.get("joint_names", [])
        if not joint_names:
            return {"default": "<p>No joint metadata available</p>"}

        # Get joint groups (from metadata or fallback to default)
        joint_groups = self._get_joint_groups()

        if not joint_groups:
            return {"default": "<p>No joint groups available</p>"}

        # Generate plots for each group
        plots_html = {}
        for group_name, indices in joint_groups.items():
            display_name = group_name.replace("_", " ").title()
            plots_html[group_name] = self._generate_body_part_plots(df, indices, display_name, episode_id)

        return plots_html

    def _get_joint_groups(self) -> dict[str, list[int]]:
        """Get joint groups from metadata or create default grouping.

        Returns:
            Dict mapping group names to lists of joint indices.
            If metadata has joint_groups, uses those.
            Otherwise, creates "default" group with all joints.
        """
        # Try to get joint groups from metadata
        joint_groups_meta = self.metadata.get("joint_groups", None)

        if joint_groups_meta:
            # Extract indices from metadata structure
            joint_groups = {}
            for group_name, group_data in joint_groups_meta.items():
                if isinstance(group_data, dict) and "indices" in group_data:
                    joint_groups[group_name] = group_data["indices"]
                elif isinstance(group_data, list):
                    joint_groups[group_name] = group_data

            return joint_groups

        # Fallback: Create default group with all joints
        num_joints = self.metadata.get("num_joints", 0)
        if num_joints > 0:
            return {"default": list(range(num_joints))}

        return {}

    def _generate_body_part_plots(
        self, df: pd.DataFrame, joint_indices: list[int], body_part: str, episode_id: int
    ) -> str:
        """Generate plots for a group of joints."""
        if not joint_indices:
            return f"<p>No {body_part} joints found</p>"

        joint_names = self.metadata.get("joint_names", [])
        joint_pos_limits = self.metadata.get("joint_pos_limits", [])
        joint_vel_limits = self.metadata.get("joint_vel_limits", [])

        # Create grid of subplots: N rows (joints) × 3 columns (pos, vel, acc)
        num_joints = len(joint_indices)

        # Create subplots - avoid using both row_titles and column_titles to prevent plotly bugs
        fig = sp.make_subplots(
            rows=num_joints,
            cols=3,
            subplot_titles=None,  # Set titles manually later to avoid plotly state issues
            vertical_spacing=0.02,
            horizontal_spacing=0.08,
        )

        # Manually add column titles as annotations
        fig.add_annotation(
            text="Position (rad)", xref="paper", yref="paper", x=0.15, y=1.02, showarrow=False, font={"size": 14}
        )
        fig.add_annotation(
            text="Velocity (rad/s)", xref="paper", yref="paper", x=0.5, y=1.02, showarrow=False, font={"size": 14}
        )
        fig.add_annotation(
            text="Acceleration (rad/s²)", xref="paper", yref="paper", x=0.85, y=1.02, showarrow=False, font={"size": 14}
        )

        timestep = df["timestep"]

        # Plot each joint as a row with 3 columns (pos, vel, acc)
        for row_idx, joint_idx in enumerate(joint_indices):
            joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"Joint {joint_idx}"
            row_num = row_idx + 1  # plotly rows are 1-indexed

            # Position (column 1)
            pos_col = f"joint_pos_{joint_idx}"
            if pos_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=timestep, y=df[pos_col], name=joint_name, showlegend=False, line={"width": 1.5}),
                    row=row_num,
                    col=1,
                )

                # Add position limits as scatter traces (avoids plotly recursion bug with add_hline)
                if joint_idx < len(joint_pos_limits):
                    limits = joint_pos_limits[joint_idx]
                    x_vals = [timestep.iloc[0], timestep.iloc[-1]]
                    # Upper limit
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=[limits[1], limits[1]],
                            mode="lines",
                            line={"color": "red", "dash": "dash", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_num,
                        col=1,
                    )
                    # Lower limit
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=[limits[0], limits[0]],
                            mode="lines",
                            line={"color": "red", "dash": "dash", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_num,
                        col=1,
                    )

            # Velocity (column 2)
            vel_col = f"joint_vel_{joint_idx}"
            if vel_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=timestep, y=df[vel_col], name=joint_name, showlegend=False, line={"width": 1.5}),
                    row=row_num,
                    col=2,
                )

                # Add velocity limits as scatter traces (avoids plotly recursion bug with add_hline)
                if joint_idx < len(joint_vel_limits):
                    vel_limit = joint_vel_limits[joint_idx]
                    x_vals = [timestep.iloc[0], timestep.iloc[-1]]
                    # Positive limit
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=[vel_limit, vel_limit],
                            mode="lines",
                            line={"color": "red", "dash": "dash", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_num,
                        col=2,
                    )
                    # Negative limit
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=[-vel_limit, -vel_limit],
                            mode="lines",
                            line={"color": "red", "dash": "dash", "width": 1},
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_num,
                        col=2,
                    )

            # Acceleration (column 3)
            acc_col = f"joint_acc_{joint_idx}"
            if acc_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=timestep, y=df[acc_col], name=joint_name, showlegend=False, line={"width": 1.5}),
                    row=row_num,
                    col=3,
                )

        # Update layout
        height = max(400, num_joints * 200)  # Scale height based on number of joints
        width = 1400  # Fixed width for 3 columns
        fig.update_layout(
            height=height, width=width, title_text=f"{body_part} Joints - Episode {episode_id}", hovermode="x unified"
        )

        # Add joint names as y-axis labels for first column
        for row_idx, joint_idx in enumerate(joint_indices):
            joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"Joint {joint_idx}"
            fig.update_yaxes(title_text=joint_name, row=row_idx + 1, col=1)

        # Add x-axis labels to bottom row only
        fig.update_xaxes(title_text="Time (s)", row=num_joints, col=1)
        fig.update_xaxes(title_text="Time (s)", row=num_joints, col=2)
        fig.update_xaxes(title_text="Time (s)", row=num_joints, col=3)

        return fig.to_html(include_plotlyjs=False)

    def _generate_summary_tracking_plot(self, episode_summaries: list[dict]) -> str:
        """Generate summary plot showing tracking errors across all episodes (dynamic)."""
        # Dynamically determine which error fields are available
        all_error_keys = set()
        for ep in episode_summaries:
            if ep.get("tracking_errors"):
                all_error_keys.update(ep["tracking_errors"].keys())

        if not all_error_keys:
            return "<p>No tracking data available</p>"

        # Create error data structure for available fields only
        errors_data = {key: [] for key in all_error_keys}
        errors_data["status"] = []

        for ep in episode_summaries:
            if ep.get("tracking_errors"):
                for key in all_error_keys:
                    if key in ep["tracking_errors"]:
                        errors_data[key].append(ep["tracking_errors"][key]["mean"])
                errors_data["status"].append("Success" if ep["is_success"] else "Failed")

        # Create box plot comparing errors
        fig = go.Figure()

        # Define display labels for common fields
        field_labels = {
            "lin_vel_x": "Lin Vel X (m/s)",
            "lin_vel_y": "Lin Vel Y (m/s)",
            "ang_vel_z": "Ang Vel Z (rad/s)",
            "height": "Height (m)",
        }

        for key in sorted(all_error_keys):
            if errors_data[key]:
                label = field_labels.get(key, key.replace("_", " ").title())
                fig.add_trace(go.Box(y=errors_data[key], name=label))

        fig.update_layout(
            title="Mean Tracking Errors Across All Episodes",
            yaxis_title="Absolute Error",
            height=400,
            showlegend=True,
        )

        return fig.to_html(include_plotlyjs="cdn")

    def _render_index_template(self, **kwargs) -> str:
        """Render index.html using template."""
        template = Template(INDEX_HTML_TEMPLATE)
        return template.render(**kwargs)

    def _render_episode_template(self, **kwargs) -> str:
        """Render episode HTML using template."""
        template = Template(EPISODE_HTML_TEMPLATE)
        return template.render(**kwargs)


# ============================================================================
# HTML Templates
# ============================================================================

INDEX_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{task_name}} - Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        table {
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }
        thead {
            background: #667eea;
            color: white;
        }
        th, td {
            padding: 12px;
            text-align: left;
        }
        th {
            cursor: pointer;
            user-select: none;
        }
        th:hover {
            background: #5568d3;
        }
        tr:nth-child(even) {
            background: #f9f9f9;
        }
        tr:hover {
            background: #f0f0f0;
        }
        .success {
            color: #28a745;
            font-weight: bold;
        }
        .failed {
            color: #dc3545;
            font-weight: bold;
        }
        a {
            color: #667eea;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .search-box {
            margin: 20px 0;
            padding: 10px;
            width: 300px;
            border: 2px solid #667eea;
            border-radius: 5px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Evaluation Report: {{task_name}}</h1>
        <p>Trajectory Analysis Dashboard</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{{total_episodes}}</div>
            <div class="metric-label">Total Episodes</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {{'success' if success_rate > 50 else 'failed'}}">{{successful_episodes}}</div>
            <div class="metric-label">Successful</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{{success_rate|round(1)}}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
    </div>

    <h2>Episode Summary</h2>
    <input type="text" id="searchBox" class="search-box" onkeyup="filterTable()" placeholder="Search episodes...">

    <table id="episodeTable">
        <thead>
            <tr>
                <th onclick="sortTable(0)">Episode ID ▼</th>
                <th onclick="sortTable(1)">Env ID</th>
                <th onclick="sortTable(2)">Status</th>
                <th onclick="sortTable(3)">Duration (s)</th>
                <th onclick="sortTable(4)">Frames</th>
                <th onclick="sortTable(5)">Mean Error</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
            {% for ep in episodes %}
            <tr>
                <td>{{ep.id}}</td>
                <td>{{ep.env_id}}</td>
                <td class="{{ep.status_class}}">{{ep.status_icon}}</td>
                <td>{{ep.duration|round(2)}}</td>
                <td>{{ep.num_frames}}</td>
                <td>
                    {% if ep.tracking_errors %}
                        {{ep.tracking_errors.get('lin_vel_x', {}).get('mean', 0)|round(4)}}
                    {% else %}
                        N/A
                    {% endif %}
                </td>
                <td><a href="episodes/episode_{{"%03d"|format(ep.id)}}.html">View →</a></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <h2>Tracking Performance Summary</h2>
    {{summary_plot|safe}}

    <script>
        // Table sorting
        function sortTable(columnIndex) {
            var table = document.getElementById("episodeTable");
            var switching = true;
            var dir = "asc";
            var switchcount = 0;

            while (switching) {
                switching = false;
                var rows = table.rows;

                for (var i = 1; i < (rows.length - 1); i++) {
                    var shouldSwitch = false;
                    var x = rows[i].getElementsByTagName("TD")[columnIndex];
                    var y = rows[i + 1].getElementsByTagName("TD")[columnIndex];

                    var xContent = isNaN(x.innerHTML) ? x.innerHTML.toLowerCase() : parseFloat(x.innerHTML);
                    var yContent = isNaN(y.innerHTML) ? y.innerHTML.toLowerCase() : parseFloat(y.innerHTML);

                    if (dir == "asc" && xContent > yContent) {
                        shouldSwitch = true;
                        break;
                    } else if (dir == "desc" && xContent < yContent) {
                        shouldSwitch = true;
                        break;
                    }
                }

                if (shouldSwitch) {
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount++;
                } else if (switchcount == 0 && dir == "asc") {
                    dir = "desc";
                    switching = true;
                }
            }
        }

        // Table search/filter
        function filterTable() {
            var input = document.getElementById("searchBox");
            var filter = input.value.toUpperCase();
            var table = document.getElementById("episodeTable");
            var tr = table.getElementsByTagName("tr");

            for (var i = 1; i < tr.length; i++) {
                var td = tr[i].getElementsByTagName("td");
                var found = false;

                for (var j = 0; j < td.length; j++) {
                    if (td[j]) {
                        var txtValue = td[j].textContent || td[j].innerText;
                        if (txtValue.toUpperCase().indexOf(filter) > -1) {
                            found = true;
                            break;
                        }
                    }
                }

                tr[i].style.display = found ? "" : "none";
            }
        }
    </script>
</body>
</html>
"""

EPISODE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Episode {{episode_id}} - {{task_name}}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .nav {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav a {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .status-badge.success {
            background: #d4edda;
            color: #155724;
        }
        .status-badge.failed {
            background: #f8d7da;
            color: #721c24;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 0.85em;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }
        section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        details {
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }
        summary {
            background: #f8f9fa;
            padding: 15px;
            cursor: pointer;
            font-weight: 600;
            user-select: none;
        }
        summary:hover {
            background: #e9ecef;
        }
        details[open] summary {
            background: #667eea;
            color: white;
        }
        .plot-container {
            margin: 20px 0;
            overflow-x: auto;
            overflow-y: visible;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 10px;
            background: #fafafa;
        }
    </style>
</head>
<body>
    <div class="nav">
        <a href="../index.html">← Back to Summary</a>
    </div>

    <div class="header">
        <h1>Episode {{episode_id}}
            <span class="status-badge {{status_class}}">
                {{'✓ Success' if is_success else '✗ Failed'}}
            </span>
        </h1>
        <p>Environment ID: {{env_id}} | Duration: {{duration|round(2)}}s | Frames: {{num_frames}}</p>
    </div>

    {% if tracking_errors %}
    <div class="stats-grid">
        {% for name, values in tracking_errors.items() %}
        <div class="stat-item">
            <div class="stat-label">{{name.replace('_', ' ').title()}} Error</div>
            <div class="stat-value">{{values.mean|round(4)}}</div>
            <div class="stat-label">Max: {{values.max|round(4)}}</div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <section>
        <h2>Tracking Performance</h2>
        <div class="plot-container">
            {{tracking_plot|safe}}
        </div>
    </section>

    <section>
        <h2>Joint Analysis</h2>
        <p style="color: #666; font-style: italic; margin-bottom: 15px;">
            💡 Tip: Plots are horizontally scrollable. Use mouse or trackpad to scroll right/left to see all joints.
        </p>

        {% for group_name, plot_html in joint_plots.items() %}
        <details {{'open' if loop.first else ''}}>
            <summary>▶ {{group_name.replace('_', ' ').title()}} Joints</summary>
            <div class="plot-container">
                {{plot_html|safe}}
            </div>
        </details>
        {% endfor %}
    </section>

    <div class="nav" style="text-align: center;">
        <a href="../index.html">← Back to Summary</a>
    </div>
</body>
</html>
"""
