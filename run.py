#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import abc
import argparse
import json
import os
import pathlib
import subprocess
import time
import uuid
from dataclasses import dataclass
from urllib.parse import quote, urljoin

import requests
import yaml
from bs4 import BeautifulSoup

# Current file directory
SCRIPT_DIR = pathlib.Path(__file__).parent

# File to store image mappings
IMAGE_MAPPINGS_FILE = SCRIPT_DIR / "workflows/image_mappings.json"

CHECKPOINT_FORMAT_DESCRIPTION = (
    "For local checkpoints the string should be formatted like "
    "`file://path/to/checkpoint.pt` or `path/to/checkpoint.pt`. For OSMO checkpoints the string "
    "should be formatted like `osmo://<workflow_name>/<iteration_number>` or "
    "`<workflow_name>/<iteration_number>`."
)


@dataclass
class RunConfig:
    """Configuration for remote training/eval workflows."""

    image_name: str
    dockerfile: pathlib.Path
    osmo_pools: dict[str, str]
    omni_server_url: str
    wandb_team_name: str
    osmo_storage_url: str
    download_base_dir: pathlib.Path
    train_workflow: pathlib.Path
    eval_workflow: pathlib.Path
    sweep_workflow: pathlib.Path

    @classmethod
    def load_from_path(cls, path: pathlib.Path) -> "RunConfig":
        """Load run config from YAML file and flatten nested structure."""
        data = yaml.safe_load(path.read_text())

        # Flatten nested structure
        flattened = {
            "image_name": data["container_registry"]["image_name"],
            "dockerfile": data["container_registry"]["dockerfile"],
            "osmo_pools": data["osmo_pools"],
            "omni_server_url": data["omniverse"]["server_url"],
            "wandb_team_name": data["wandb"]["team_name"],
            "osmo_storage_url": data["storage"]["osmo_storage_url"],
            "download_base_dir": data["storage"]["download_base_dir"],
            "train_workflow": data["paths"]["train_workflow"],
            "eval_workflow": data["paths"]["eval_workflow"],
            "sweep_workflow": data["paths"]["sweep_workflow"],
        }

        config = cls(**flattened)

        # Make paths absolute relative to run_config.yaml location
        config.dockerfile = path.parent / config.dockerfile
        config.download_base_dir = path.parent / config.download_base_dir
        config.train_workflow = path.parent / config.train_workflow
        config.eval_workflow = path.parent / config.eval_workflow
        config.sweep_workflow = path.parent / config.sweep_workflow

        return config


def run(command: list[str], cwd: pathlib.Path | None = None):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=cwd)


class FileApi(abc.ABC):
    @abc.abstractmethod
    def list_directory(self, remote_path: pathlib.Path) -> list[pathlib.Path]:
        pass

    @abc.abstractmethod
    def download_file(self, remote_path: pathlib.Path, local_path: pathlib.Path) -> None:
        pass


class HttpFileApi(FileApi):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/") + "/"

    def list_directory(self, remote_path: pathlib.Path) -> list[pathlib.Path]:
        url = urljoin(self.base_url, str(remote_path))
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        items: list[pathlib.Path] = []

        for row in soup.select("table#listing tr.item"):
            # Skip parent directory link
            if row.get("id") == "parent":
                continue

            link = row.select_one("td.colname a")
            if not link:
                continue

            href = link.get("href")
            if href == "../":
                continue

            # Determine if it's a directory
            if isinstance(href, str):
                items.append(pathlib.Path(href))

        return items

    def download_file(self, remote_path: pathlib.Path, local_path: pathlib.Path) -> None:
        print(f"Downloading {remote_path} to {local_path}.")
        url = urljoin(self.base_url, str(remote_path))
        response = requests.get(url)
        response.raise_for_status()
        local_path.write_bytes(response.content)


class FileBrowserFileApi(FileApi):
    def __init__(self, server_url: str, username: str, password: str, workflow: str):
        self.server_url = server_url.rstrip("/")
        self.username = username
        self.password = password
        self._port_process = start_port_forwarding(workflow)
        for _ in range(10):
            try:
                time.sleep(1)
                self._token = self._connect()
                break
            except requests.exceptions.ConnectionError:
                pass
        if not hasattr(self, "_token"):
            raise RuntimeError("Failed to connect to running OSMO workflow.")

    def __del__(self):
        self._port_process.terminate()
        self._port_process.wait()

    def _connect(self) -> str:
        print(f"Trying to connect to {self.server_url} as {self.username}")
        login_url = f"{self.server_url}/api/login"
        response = requests.post(login_url, json={"username": self.username, "password": self.password})
        response.raise_for_status()
        print("Connected.")
        return response.text

    def list_directory(self, remote_path: pathlib.Path) -> list[pathlib.Path]:
        encoded_path = quote(str(remote_path))
        url = f"{self.server_url}/api/resources/{encoded_path}"
        response = requests.get(url, headers={"X-Auth": self._token})
        response.raise_for_status()
        return [pathlib.Path(item["path"]) for item in response.json().get("items", [])]

    def download_file(self, remote_path: pathlib.Path, local_path: pathlib.Path) -> None:
        print(f"Downloading {remote_path} to {local_path}.")
        encoded_path = quote(str(remote_path))
        url = f"{self.server_url}/api/raw/{encoded_path}"
        response = requests.get(url, headers={"X-Auth": self._token})
        response.raise_for_status()
        local_path.write_bytes(response.content)


def get_workflow_status(workflow: str) -> str:
    result = subprocess.run(
        ["osmo", "workflow", "query", workflow, "--format-type", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query workflow status: {result.stderr}")
    return json.loads(result.stdout).get("status", "UNKNOWN")


def start_port_forwarding(workflow: str) -> subprocess.Popen:
    process = subprocess.Popen(
        ["osmo", "workflow", "port-forward", workflow, "train", "--port", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def download_checkpoints(workflow_name: str, iterations: list[int], run_config: RunConfig) -> list[pathlib.Path]:
    """Download checkpoints from OSMO."""
    workflow_status = get_workflow_status(workflow_name)
    print(f"Workflow status: {workflow_status}")

    if workflow_status == "RUNNING":
        file_api: FileApi = FileBrowserFileApi(
            server_url="http://localhost:8080",
            username="admin",
            password="admin",
            workflow=workflow_name,
        )
        output_dir = file_api.list_directory(pathlib.Path("osmo/data/output"))[0]
    else:
        file_api = HttpFileApi(run_config.osmo_storage_url)
        output_dir = pathlib.Path(workflow_name) / "train"

    download_dir = run_config.download_base_dir / workflow_name
    download_dir.mkdir(parents=True, exist_ok=True)

    files = file_api.list_directory(output_dir)
    tensorboard_files = [f.name for f in files if f.name.startswith("events.out.tfevents")]

    for config_file in ["config.json", "env_config.json"] + tensorboard_files:
        remote_path = output_dir / config_file
        local_path = download_dir / config_file
        file_api.download_file(remote_path, local_path)

    checkpoint_paths = []
    for iteration in iterations:
        checkpoint_filename = f"model_{iteration}.pt"
        remote_path = output_dir / checkpoint_filename
        local_path = download_dir / checkpoint_filename
        file_api.download_file(remote_path, local_path)
        checkpoint_paths.append(local_path)

    return checkpoint_paths


def get_checkpoints(checkpoints: list[str], run_config: RunConfig) -> list[pathlib.Path]:
    """
    Get checkpoints from local paths or from OSMO.

    The checkpoints can be specified
    - as a file URI, ie. `file://path/to/checkpoint.pt` or `path/to/checkpoint.pt`
    - as a custom OSMO URI, ie. `osmo://<workflow_name>/<iteration_number>` or
      `<workflow_name>/<iteration_number>`

    Normally the `file://` or `osmo://` can be omitted, but it can be used when the checkpoint is
    ambiguous.
    """
    local_checkpoints = []
    for checkpoint in checkpoints:
        print(f"Checking checkpoint: {checkpoint}")
        if checkpoint.startswith("file://"):
            path = pathlib.Path(checkpoint.replace("file://", ""))
            assert path.exists(), f"Checkpoint {path} does not exist."
            local_checkpoints.append(path)
        elif pathlib.Path(checkpoint).exists():
            local_checkpoints.append(pathlib.Path(checkpoint))
        else:
            workflow, iterations_str = checkpoint.replace("osmo://", "").split("/")
            iterations = [int(iteration) for iteration in iterations_str.split(",")]
            local_checkpoints.extend(download_checkpoints(workflow, iterations, run_config))
    return local_checkpoints


def build_docker_image(
    run_config: RunConfig,
    resume_checkpoint_path: pathlib.Path | None = None,
) -> str:
    image_name = f"{run_config.image_name}:{uuid.uuid4()}"
    image_name_latest = f"{run_config.image_name}:latest"

    # Extract git information for reproducibility before building Docker image
    print("Extracting git information for reproducibility...")
    extract_git_script = SCRIPT_DIR / "scripts" / "extract_git_info.sh"
    if extract_git_script.exists():
        try:
            run([str(extract_git_script)])
            print("✅ Git information extracted successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to extract git information: {e}")
            print("Proceeding with Docker build without git info...")
    else:
        print("⚠️  Warning: extract_git_info.sh not found, proceeding without git info...")

    print(f"Building docker container: {image_name}")
    print(f"Resume checkpoint path: {resume_checkpoint_path}")
    command = [
        "docker",
        "build",
        "-f",
        str(run_config.dockerfile),
        "-t",
        image_name,
        "-t",
        image_name_latest,
        ".",
    ]

    if resume_checkpoint_path is not None:
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume-checkpoint file {resume_checkpoint_path} does not exist.")

        # Always copy checkpoint to a consistent location for Docker bundling
        import shutil

        resolved_checkpoint = resume_checkpoint_path.resolve()

        # Use agile/data/policy/eval for eval checkpoints (already in .gitignore via agile/data/policy/)
        eval_checkpoint_dir = SCRIPT_DIR / "agile" / "data" / "policy" / "eval"
        eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        eval_checkpoint = eval_checkpoint_dir / resolved_checkpoint.name
        shutil.copy2(resolved_checkpoint, eval_checkpoint)
        print(f"Bundling checkpoint: {resolved_checkpoint.name}")

        # Calculate paths relative to workspace for Docker
        resume_path = eval_checkpoint.parent.relative_to(SCRIPT_DIR)
        resume_checkpoint = eval_checkpoint.name
        resume_path_str = f"{resume_path}/"

        command.extend(["--build-arg", "RESUME_STAGE=yes"])
        command.extend(["--build-arg", f"RESUME_PATH={resume_path_str}"])
        command.extend(["--build-arg", f"RESUME_CHECKPOINT={resume_checkpoint}"])

    run(command, cwd=SCRIPT_DIR)
    run(["docker", "push", image_name])
    return image_name


def submit_osmo_workflow(workflow_file: pathlib.Path, set_args: list[str], pool: str):
    command = [
        "osmo",
        "workflow",
        "submit",
        str(workflow_file),
        f"--pool={pool}",
        "--set",
    ] + set_args
    run(command)


def load_image_mappings() -> dict[str, str]:
    """Load experiment name to image name mappings from JSON file."""
    if not IMAGE_MAPPINGS_FILE.exists():
        return {}

    try:
        with open(IMAGE_MAPPINGS_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load image mappings from {IMAGE_MAPPINGS_FILE}: {e}")
        return {}


def save_image_mappings(mappings: dict[str, str]) -> None:
    """Save experiment name to image name mappings to JSON file."""
    try:
        with open(IMAGE_MAPPINGS_FILE, "w") as f:
            json.dump(mappings, f, indent=2)
    except OSError as e:
        print(f"Warning: Could not save image mappings to {IMAGE_MAPPINGS_FILE}: {e}")


def get_existing_image(experiment_name: str) -> str | None:
    """Get the existing image name for an experiment, if any."""
    mappings = load_image_mappings()
    return mappings.get(experiment_name)


def store_image_mapping(experiment_name: str, image_name: str) -> None:
    """Store the mapping between experiment name and image name."""
    mappings = load_image_mappings()
    mappings[experiment_name] = image_name
    save_image_mappings(mappings)


def handle_train(
    name: str,
    run_config: RunConfig,
    task_name: str,
    project_name: str,  # This should always have a value (either user-provided or defaulted to task_name)
    resume_checkpoint: str | None,
    use_existing: bool = False,
    rebuild: bool = False,
    logger: str = "wandb",
    seeds: list[int] | None = None,
    max_iterations: int = 30000,
    set_args: list[str] = None,
    image_lookup_name: str | None = None,
):
    if seeds:
        for seed in seeds:
            seed_run_name = f"{name}_seed{seed}"
            # Create a copy of set_args to avoid modifying the original list
            seed_set_args = set_args.copy()
            seed_set_args.append(f"seed={seed}")
            handle_train_single(
                name=seed_run_name,
                run_config=run_config,
                task_name=task_name,
                project_name=project_name,
                resume_checkpoint=resume_checkpoint,
                use_existing=use_existing,
                rebuild=rebuild,
                logger=logger,
                max_iterations=max_iterations,
                set_args=seed_set_args,
                image_lookup_name=image_lookup_name or name,  # Use provided key or original name
            )
    else:
        handle_train_single(
            name=name,
            run_config=run_config,
            task_name=task_name,
            project_name=project_name,
            resume_checkpoint=resume_checkpoint,
            use_existing=use_existing,
            rebuild=rebuild,
            logger=logger,
            max_iterations=max_iterations,
            set_args=set_args,
            image_lookup_name=image_lookup_name,
        )


def handle_train_single(
    name: str,
    run_config: RunConfig,
    task_name: str,
    project_name: str,  # This should always have a value (either user-provided or defaulted to task_name)
    resume_checkpoint: str | None,
    use_existing: bool = False,
    rebuild: bool = False,
    logger: str = "wandb",
    max_iterations: int = 30000,
    set_args: list[str] = None,
    image_lookup_name: str | None = None,
):
    full_name = f"agile_{name}"
    resume_checkpoint_path: pathlib.Path | None = None
    # Use image_lookup_name for image lookup/storage, fallback to name
    image_key = image_lookup_name or name

    # Collect resume checkpoint.
    if resume_checkpoint:
        print(f"Resume checkpoint: {resume_checkpoint}")
        checkpoints = get_checkpoints([resume_checkpoint], run_config)
        assert len(checkpoints) == 1
        resume_checkpoint_path = checkpoints[0]

    # Handle image selection/building
    image_name = None

    if use_existing and not rebuild:
        # Try to use existing image
        existing_image = get_existing_image(image_key)
        if existing_image:
            print(f"Using existing image: {existing_image}")
            image_name = existing_image
        else:
            print(f"No existing image found for experiment '{image_key}', building new image...")

    if image_name is None or rebuild:
        # Build new image
        image_name = build_docker_image(
            resume_checkpoint_path=resume_checkpoint_path,
            run_config=run_config,
        )
        # Store the mapping
        store_image_mapping(image_key, image_name)
        print(f"Stored image mapping: {image_key} -> {image_name}")

    submit_osmo_workflow(
        run_config.train_workflow,
        [
            f"workflow_name={full_name}",
            f"image={image_name}",
            f"omni_server={run_config.omni_server_url}",
            f"image_default={run_config.image_name}:latest",
            f"resume={bool(resume_checkpoint)}",
            f"task_name={task_name}",
            f"project_name={project_name}",
            f"run_name={name}",
            f"logger={logger}",
            f"max_iterations={max_iterations}",
        ]
        + set_args,
        pool=run_config.osmo_pools["train"],
    )


def handle_sweep(
    name: str,
    run_config: RunConfig,
    sweep_name: str,
    use_existing: bool = False,
    rebuild: bool = False,
    set_args: list[str] = None,
):
    full_name = f"agile_{name}"
    image_name = None

    if use_existing and not rebuild:
        existing_image = get_existing_image(name)
        if existing_image:
            print(f"Using existing image: {existing_image}")
            image_name = existing_image
        else:
            print(f"No existing image found for experiment '{name}', building new image...")

    if image_name is None or rebuild:
        image_name = build_docker_image(run_config=run_config)
        store_image_mapping(name, image_name)
        print(f"Stored image mapping: {name} -> {image_name}")

    submit_osmo_workflow(
        run_config.sweep_workflow,
        [
            f"workflow_name={full_name}",
            f"image={image_name}",
            f"omni_server={run_config.omni_server_url}",
            f"image_default={run_config.image_name}:latest",
            f"sweep_name={sweep_name}",
        ]
        + set_args,
        pool=run_config.osmo_pools["sweep"],
    )


def add_wandb_args(set_args: list[str], wandb_team_name: str):
    set_args.extend(
        [
            f"wandb_pass={os.environ.get('WANDB_API_KEY', '')}",
            f"wandb_username={wandb_team_name}",
        ]
    )
    return set_args


def add_huggingface_token(set_args: list[str]):
    set_args.extend(
        [
            f"hf_token={os.environ.get('HF_TOKEN', '')}",
        ]
    )
    return set_args


def handle_eval(
    name: str,
    wandb_run_path: str | None,
    checkpoint_path: str | None,
    checkpoints_spec: str,
    eval_config_path: str | None,
    task_name: str,
    project_name: str | None,
    use_existing: bool,
    rebuild: bool,
    set_args: list[str],
    run_config: RunConfig,
):
    """Handle evaluation of trained checkpoints from wandb or local paths.

    Downloads checkpoints from wandb inside OSMO containers or bundles local checkpoints
    into Docker image. Runs deterministic evaluation scenarios. Each checkpoint gets its
    own parallel OSMO job. Results are uploaded to wandb with tracking plots and HTML reports.

    Args:
        name: Name of the evaluation experiment
        wandb_run_path: Full wandb run path (e.g., 'nvidia-isaac/project/run_id') or None
        checkpoint_path: Local checkpoint path or None
        checkpoints_spec: Checkpoint specification ("latest" or "1000,5000,10000")
        eval_config_path: Path to evaluation config YAML (or None for default)
        task_name: Task name (required - must match training task)
        project_name: Wandb project name for eval results (defaults to {task_name}_eval)
        use_existing: Whether to reuse existing Docker image
        rebuild: Whether to force rebuild Docker image
        set_args: Additional arguments to pass to workflow
        config: RunConfig with image/workflow paths

    Example:
        # Evaluate latest checkpoint from a wandb training run
        python run.py eval --name eval_test \\
            --wandb_run nvidia-isaac/Locomotion-G1-29DoF-v0/runs/hk87x2ms?nw=nwuserhuihuaz \\
            --task_name Velocity-Height-G1-Dev-v0

        # Evaluate specific checkpoints
        python run.py eval --name multi_checkpoint \\
            --wandb_run nvidia-isaac/Locomotion-G1-29DoF-v0/runs/hk87x2ms \\
            --task_name Velocity-Height-G1-Dev-v0 \\
            --checkpoints 5000,10000,15000

        # Use custom evaluation scenario
        python run.py eval --name custom_test \\
            --wandb_run nvidia-isaac/Locomotion-G1-29DoF-v0/runs/hk87x2ms \\
            --task_name Velocity-Height-G1-Dev-v0 \\
            --eval_config agile/algorithms/evaluation/configs/examples/multi_env_capability_test.yaml

        # Evaluate local checkpoint
        python run.py eval --name eval_local \\
            --checkpoint_path /path/to/model_5000.pt \\
            --task_name Velocity-Height-G1-Dev-v0
    """
    # Validate input arguments
    if wandb_run_path is None and checkpoint_path is None:
        raise ValueError("Either --wandb_run or --checkpoint_path must be provided")
    if wandb_run_path is not None and checkpoint_path is not None:
        raise ValueError("Cannot use both --wandb_run and --checkpoint_path simultaneously")

    # Determine if using local checkpoint
    use_local_checkpoint = checkpoint_path is not None

    if use_local_checkpoint:
        print(f"Setting up evaluation for local checkpoint: {checkpoint_path}")
    else:
        print(f"Setting up evaluation for wandb run: {wandb_run_path}")
        # Validate wandb run path format
        if "/" not in wandb_run_path:
            raise ValueError(
                f"Invalid wandb run path: '{wandb_run_path}'. "
                "Please provide the full path in format: 'team/project/run_id' or 'project/run_id'"
            )

    print(f"Task: {task_name}")

    # Parse checkpoint specification (only for wandb runs)
    if use_local_checkpoint:
        # For local checkpoints, we'll evaluate just the one provided
        iterations = [0]  # Use 0 as a placeholder
        full_wandb_path = None
        print(f"Will evaluate local checkpoint: {checkpoint_path}")
    else:
        if checkpoints_spec == "latest":
            iterations = [0]  # Use 0 to indicate "latest" in the workflow
            print("Will evaluate latest checkpoint from wandb")
        else:
            # Parse comma-separated list
            iterations = [int(x.strip()) for x in checkpoints_spec.split(",")]
            print(f"Will evaluate {len(iterations)} checkpoint(s): {iterations}")

        # Clean wandb path (remove query parameters and /runs/ prefix)
        # Format: nvidia-isaac/Locomotion-G1-29DoF-v0/runs/hk87x2ms?nw=...
        # -> nvidia-isaac/Locomotion-G1-29DoF-v0/hk87x2ms
        clean_path = wandb_run_path.split("?")[0]  # Remove query params
        if "/runs/" in clean_path:
            parts = clean_path.split("/runs/")
            if len(parts) == 2:
                clean_path = f"{parts[0]}/{parts[1]}"

        full_wandb_path = clean_path
        print(f"Wandb run (cleaned): {full_wandb_path}")
        print("Checkpoints will be downloaded inside OSMO containers")

    # Determine eval config to use
    if eval_config_path is None:
        # Use default eval config
        eval_config_path = "agile/algorithms/evaluation/configs/default_eval.yaml"
        print(f"Using default evaluation config: {eval_config_path}")
    else:
        print(f"Using custom evaluation config: {eval_config_path}")

    # Verify eval config exists
    eval_config_file = pathlib.Path(eval_config_path)
    if not eval_config_file.exists():
        raise FileNotFoundError(f"Evaluation config not found: {eval_config_path}")

    # Determine project name for eval results
    if project_name is None:
        project_name = f"{task_name}_eval"
    print(f"Using project name for eval results: {project_name}")

    # Get local checkpoint path if using local checkpoint
    local_checkpoint_path: pathlib.Path | None = None
    if use_local_checkpoint:
        print(f"Retrieving local checkpoint: {checkpoint_path}")
        checkpoints = get_checkpoints([checkpoint_path], run_config)
        assert len(checkpoints) == 1, f"Expected 1 checkpoint, got {len(checkpoints)}"
        local_checkpoint_path = checkpoints[0]
        print(f"Local checkpoint resolved to: {local_checkpoint_path}")

    # Handle Docker image
    image_name = None
    if use_existing and not rebuild and not use_local_checkpoint:
        # Only reuse image if not using local checkpoint (checkpoint needs to be bundled)
        existing_image = get_existing_image(name)
        if existing_image:
            print(f"Using existing image: {existing_image}")
            image_name = existing_image

    if image_name is None or rebuild:
        # Build new image (bundle local checkpoint if provided)
        if use_local_checkpoint:
            print(f"Building Docker image with bundled checkpoint: {local_checkpoint_path}")
            image_name = build_docker_image(
                run_config=run_config,
                resume_checkpoint_path=local_checkpoint_path,
            )
        else:
            # Build image without checkpoint (will download from wandb)
            image_name = build_docker_image(run_config=run_config)
        store_image_mapping(name, image_name)
        print(f"Built and stored image: {image_name}")

    # Submit OSMO job for each checkpoint
    for iteration in iterations:
        iter_str = str(iteration) if iteration != 0 else "latest"

        # Create clean workflow name: eval_{name}_{iteration}
        workflow_name = f"eval_{name}_{iter_str}"

        print(f"\nSubmitting evaluation job for iteration {iter_str}...")
        print(f"  Workflow name: {workflow_name}")

        # Prepare workflow arguments
        if use_local_checkpoint:
            print("  Using bundled local checkpoint")
            workflow_set_args = [
                f"workflow_name={workflow_name}",
                f"image={image_name}",
                f"omni_server={run_config.omni_server_url}",
                f"image_default={run_config.image_name}:latest",
                f"task_name={task_name}",
                f"project_name={project_name}",
                f"run_name={name}",
                f"eval_config_path={eval_config_path}",
                "use_local_checkpoint=true",  # Signal to use bundled checkpoint
            ] + set_args
        else:
            print(f"  Wandb run: {full_wandb_path}")
            print(f"  Checkpoint: iter {iteration} (will be downloaded in container)")
            workflow_set_args = [
                f"workflow_name={workflow_name}",
                f"image={image_name}",
                f"omni_server={run_config.omni_server_url}",
                f"image_default={run_config.image_name}:latest",
                f"task_name={task_name}",
                f"project_name={project_name}",
                f"run_name={name}",
                f"wandb_run_path={full_wandb_path}",  # Container will download from this
                f"eval_config_path={eval_config_path}",
                f"iteration={iteration}",
                f"training_wandb_run={full_wandb_path}",  # Track which training run was evaluated
                "use_local_checkpoint=false",  # Signal to download from wandb
            ] + set_args

        # Submit workflow
        submit_osmo_workflow(run_config.eval_workflow, workflow_set_args, run_config.osmo_pools["eval"])

        print(f"  ✓ Submitted {workflow_name}")

    print(f"\n✅ Submitted {len(iterations)} evaluation job(s) to OSMO")
    print("Monitor at: https://osmo.nvidia.com/workflows")


def main():
    parser = argparse.ArgumentParser(description="CLI tool for training, evaluating, and collecting results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train a teacher or student policy.",
    )
    train_parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Name of the experiment.",
    )
    train_parser.add_argument(
        "--task_name",
        "-t",
        default="Standing-G1-v0",
        help="Name of the task.",
    )
    train_parser.add_argument(
        "--project_name",
        "-p",
        default=None,
        help="Name of the project. If not provided, defaults to the task name.",
    )
    train_parser.add_argument(
        "--resume_checkpoint",
        "-r",
        type=str,
        help="Checkpoint used to resume training. " + CHECKPOINT_FORMAT_DESCRIPTION,
    )
    train_parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing Docker image for this experiment name if available, otherwise build new one.",
    )
    train_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of Docker image even if existing one is available.",
    )
    train_parser.add_argument(
        "--logger",
        "-l",
        default="wandb",
        help="Logger to use (wandb or tensorboard).",
    )

    def parse_seeds(value):
        """Parse seeds from comma-separated or single int value."""
        if "," in value:
            return [int(x.strip()) for x in value.split(",")]
        return [int(value)]

    train_parser.add_argument(
        "--seeds",
        "-s",
        type=parse_seeds,
        nargs="+",
        help="List of random seeds to run (space or comma separated, e.g., '0 1 2' or '0,1,2').",
    )
    train_parser.add_argument(
        "--max_iterations",
        "-m",
        type=int,
        default=20000,
        help="Maximum number of training iterations. Defaults to 30000.",
    )
    train_parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Key to use for image lookup/storage instead of experiment name. Useful for reusing images across experiments.",
    )
    train_parser.add_argument(
        "--set",
        type=str,
        default=[],
        nargs="+",
        help="Use this to pass additional arguments to the OSMO workflow in the form key=value.",
    )

    # Sweep subcommand
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run a hyperparameter sweep.",
    )
    sweep_parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Name of the experiment.",
    )
    sweep_parser.add_argument(
        "--sweep_name",
        "-s",
        required=True,
        help="Name of the sweep configuration to use.",
    )
    sweep_parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing Docker image for this experiment name if available, otherwise build new one.",
    )
    sweep_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of Docker image even if existing one is available.",
    )
    sweep_parser.add_argument(
        "--set",
        type=str,
        default=[],
        nargs="+",
        help="Use this to pass additional arguments to the OSMO workflow in the form key=value.",
    )

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate trained checkpoints with deterministic scenarios.",
    )
    eval_parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Name of the evaluation experiment.",
    )
    eval_parser.add_argument(
        "--wandb_run",
        "-w",
        default=None,
        help="Full wandb run path to download checkpoints from. Format: 'team/project/run_id' or 'project/run_id' "
        "(e.g., 'nvidia-isaac/Locomotion-G1-29DoF-v0/hk87x2ms'). Mutually exclusive with --checkpoint_path.",
    )
    eval_parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to local checkpoint file to evaluate. " + CHECKPOINT_FORMAT_DESCRIPTION + " "
        "Mutually exclusive with --wandb_run.",
    )
    eval_parser.add_argument(
        "--checkpoints",
        "-c",
        default="latest",
        help="Comma-separated checkpoint iterations to evaluate (e.g., '5000,10000,15000') or 'latest' for most recent. "
        "Only used with --wandb_run.",
    )
    eval_parser.add_argument(
        "--eval_config",
        "-e",
        default=None,
        help="Path to evaluation scenario YAML. If not provided, uses default_eval.yaml.",
    )
    eval_parser.add_argument(
        "--task_name",
        "-t",
        required=True,
        help="Task name (must match the task used in training, e.g., Velocity-Height-G1-Dev-v0).",
    )
    eval_parser.add_argument(
        "--project_name",
        "-p",
        default=None,
        help="Wandb project name for eval results. Defaults to 'nvidia-isaac/{task_name}_eval'.",
    )
    eval_parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing Docker image if available.",
    )
    eval_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of Docker image.",
    )
    eval_parser.add_argument(
        "--set",
        type=str,
        default=[],
        nargs="+",
        help="Additional arguments to pass to OSMO workflow (key=value format).",
    )

    args = parser.parse_args()

    # Load run configuration
    run_config_path = SCRIPT_DIR / "workflows" / "run_config.yaml"
    if not run_config_path.exists():
        print("=" * 80)
        print("ERROR: run_config.yaml not found!")
        print("=" * 80)
        print()
        print("The run configuration file is required to run workflows.")
        print("Please create it by copying the example template:")
        print()
        print(f"  cp {SCRIPT_DIR / 'workflows' / 'run_config.example.yaml'} {run_config_path}")
        print()
        print("Then customize the values for your infrastructure (container registry,")
        print("OSMO pools, Omniverse server, etc.)")
        print()
        print("See run_config.example.yaml for detailed instructions.")
        print("=" * 80)
        import sys

        sys.exit(1)

    run_config = RunConfig.load_from_path(run_config_path)

    # Get the arguments to set in the workflow
    set_args = args.set
    if set_args is None:
        set_args = []

    # Add wandb and huggingface credentials
    set_args = add_wandb_args(set_args, run_config.wandb_team_name)
    set_args = add_huggingface_token(set_args)

    if args.command == "train":
        # Use task_name as default project_name if not provided
        project_name = args.project_name if args.project_name is not None else args.task_name
        handle_train(
            name=args.name,
            task_name=args.task_name,
            project_name=project_name,
            resume_checkpoint=args.resume_checkpoint,
            use_existing=args.use_existing,
            rebuild=args.rebuild,
            logger=args.logger,
            seeds=[s for sublist in args.seeds for s in sublist] if args.seeds else None,
            max_iterations=args.max_iterations,
            set_args=set_args,
            run_config=run_config,
            image_lookup_name=args.image_key,
        )
    elif args.command == "sweep":
        handle_sweep(
            name=args.name,
            sweep_name=args.sweep_name,
            use_existing=args.use_existing,
            rebuild=args.rebuild,
            set_args=set_args,
            run_config=run_config,
        )
    elif args.command == "eval":
        handle_eval(
            name=args.name,
            wandb_run_path=args.wandb_run,
            checkpoint_path=args.checkpoint_path,
            checkpoints_spec=args.checkpoints,
            eval_config_path=args.eval_config,
            task_name=args.task_name,
            project_name=args.project_name,
            use_existing=args.use_existing,
            rebuild=args.rebuild,
            set_args=set_args,
            run_config=run_config,
        )


if __name__ == "__main__":
    main()
