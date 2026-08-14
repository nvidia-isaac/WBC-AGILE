# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The report embeds videos as <video> tags."""

import json

from agile.algorithms.evaluation.report_generator import TrajectoryReportGenerator, _video_embed_html


def test_video_embed_html_has_video_tag_and_src():
    html = _video_embed_html("videos/eval.mp4", "Evaluation")
    assert "<video" in html
    assert 'src="videos/eval.mp4"' in html
    assert "Evaluation" in html
    assert "controls" in html


def test_report_uses_explicit_task_name_without_metadata(tmp_path):
    report_dir = tmp_path / "reports"

    generator = TrajectoryReportGenerator(
        tmp_path / "eval",
        output_dir=report_dir,
        task_name="Velocity-G1-History-v0",
    )
    index = generator.generate_full_report(open_browser=False)

    assert "Evaluation Report: Velocity-G1-History-v0" in index.read_text()


def test_report_handles_null_provenance_metadata(tmp_path):
    (tmp_path / "trajectories").mkdir()
    (tmp_path / "trajectories" / "metadata.json").write_text(json.dumps({"provenance": None}))

    generator = TrajectoryReportGenerator(tmp_path / "trajectories", output_dir=tmp_path / "reports")
    index = generator.generate_full_report(open_browser=False)

    assert "Evaluation Report: Unknown Task" in index.read_text()
