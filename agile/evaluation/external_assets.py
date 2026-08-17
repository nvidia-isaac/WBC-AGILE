# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download public robot assets used by AGILE evaluation workflows."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ASSET_ROOT = Path("external_assets")


@dataclass(frozen=True)
class SparseAssetRepo:
    """A pinned Git repository with the sparse paths required by AGILE."""

    name: str
    url: str
    revision: str
    sparse_paths: tuple[str, ...]
    required_files: tuple[Path, ...]


UNITREE_MUJOCO = SparseAssetRepo(
    name="unitree_mujoco",
    url="https://github.com/unitreerobotics/unitree_mujoco.git",
    revision="ae6a8403e272733e9996ef59990880330496177f",
    sparse_paths=("unitree_robots/g1",),
    required_files=(Path("unitree_robots/g1/scene_29dof.xml"),),
)
BOOSTER_ASSETS = SparseAssetRepo(
    name="booster_assets",
    url="https://github.com/BoosterRobotics/booster_assets.git",
    revision="508cbee6ca9ae6fbc8c0b38dd58785a6f3fc61a2",
    sparse_paths=("robots/T1",),
    required_files=(Path("robots/T1/T1_23dof.xml"),),
)

G1_MJCF = ASSET_ROOT / UNITREE_MUJOCO.name / "unitree_robots/g1/scene_29dof.xml"
T1_MJCF = ASSET_ROOT / BOOSTER_ASSETS.name / "robots/T1/T1_23dof.xml"

_REPOS = (UNITREE_MUJOCO, BOOSTER_ASSETS)
_REVISION_MARKER = ".agile_asset_revision"


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _has_origin(destination: Path) -> bool:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=destination,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _repo_is_current(repo: SparseAssetRepo, destination: Path) -> bool:
    marker = destination / _REVISION_MARKER
    return (
        marker.is_file()
        and marker.read_text().strip() == repo.revision
        and all((destination / required_file).is_file() for required_file in repo.required_files)
    )


def _prepare_destination(repo: SparseAssetRepo, destination: Path, *, force: bool) -> None:
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").is_dir():
        _run(["git", "init"], cwd=destination)
    if _has_origin(destination):
        _run(["git", "remote", "set-url", "origin", repo.url], cwd=destination)
    else:
        _run(["git", "remote", "add", "origin", repo.url], cwd=destination)


def download_repo(repo: SparseAssetRepo, *, asset_root: Path = ASSET_ROOT, force: bool = False) -> Path:
    """Download one pinned sparse repository if it is not already present."""

    destination = asset_root / repo.name
    if not force and _repo_is_current(repo, destination):
        print(f"{repo.name}: already at {repo.revision}")
        return destination

    _prepare_destination(repo, destination, force=force)
    _run(["git", "sparse-checkout", "init", "--cone"], cwd=destination)
    _run(["git", "sparse-checkout", "set", *repo.sparse_paths], cwd=destination)
    _run(["git", "fetch", "--depth", "1", "origin", repo.revision], cwd=destination)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)
    (destination / _REVISION_MARKER).write_text(f"{repo.revision}\n")

    missing = [required_file for required_file in repo.required_files if not (destination / required_file).is_file()]
    if missing:
        missing_paths = ", ".join(str(destination / path) for path in missing)
        raise FileNotFoundError(f"{repo.name} did not provide required file(s): {missing_paths}")

    print(f"{repo.name}: downloaded {repo.revision}")
    return destination


def download_external_assets(*, asset_root: Path = ASSET_ROOT, force: bool = False) -> None:
    """Download all public external assets used by automated evaluation."""

    for repo in _REPOS:
        download_repo(repo, asset_root=asset_root, force=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT, help="Directory for downloaded assets.")
    parser.add_argument(
        "--force", action="store_true", help="Redownload assets even if the pinned revision is present."
    )
    args = parser.parse_args(argv)

    download_external_assets(asset_root=args.asset_root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
