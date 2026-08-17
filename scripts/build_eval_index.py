#!/usr/bin/env python3
"""Build a small navigational HTML index from downloaded evaluation artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def build_index(batch_dir: Path) -> Path:
    """Write ``index.html`` listing each per-run report and recorded status."""
    rows: list[str] = []
    for status_path in sorted(batch_dir.glob("*/*/status.json")):
        run_dir = status_path.parent
        data = json.loads(status_path.read_text())
        rel_report = run_dir.relative_to(batch_dir) / "eval" / "reports" / "index.html"
        state = html.escape(str(data.get("state", "unknown")))
        stage = html.escape(str(data.get("stage", "unknown")))
        label = html.escape(str(run_dir.relative_to(batch_dir)))
        rows.append(
            f'<tr><td>{label}</td><td>{state}</td><td>{stage}</td><td><a href="{rel_report}">report</a></td></tr>'
        )
    output = batch_dir / "index.html"
    output.write_text(
        "<!doctype html><title>Evaluation batch</title><h1>Evaluation batch</h1>"
        "<table><tr><th>run</th><th>status</th><th>stage</th><th>report</th></tr>" + "".join(rows) + "</table>"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    print(build_index(parser.parse_args().batch_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
