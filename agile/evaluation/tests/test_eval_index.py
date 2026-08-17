import json
from pathlib import Path

from scripts.build_eval_index import build_index


def test_batch_index_lists_success_and_failure(tmp_path: Path) -> None:
    for label, state in (("ok", "succeeded"), ("bad", "failed")):
        run_dir = tmp_path / "Velocity-G1-History-v0" / label
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps({"state": state, "stage": "sim2sim"}))
        (run_dir / "eval" / "reports").mkdir(parents=True)
        (run_dir / "eval" / "reports" / "index.html").write_text("report")
    index = build_index(tmp_path)
    text = index.read_text()
    assert "succeeded" in text and "failed" in text
    assert "Velocity-G1-History-v0/ok/eval/reports/index.html" in text
