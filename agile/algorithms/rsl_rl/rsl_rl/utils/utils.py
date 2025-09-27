# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

from __future__ import annotations

import git
import importlib
import os
import pathlib
import torch
from typing import Callable
from tensordict.tensordict import TensorDictBase  # type: ignore


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def _build_trajectory_lengths(dones: torch.Tensor):
    """Return the length (in time steps) of every trajectory contained in
    the *dones* flag tensor.

    The expected shape of *dones* is ``[time, num_envs, 1]`` where the last
    dimension is a byte/bool flag indicating terminal states.  A value of
    ``1`` marks the *last* step of a trajectory.
    """
    dones = dones.clone()
    dones[-1] = 1
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    done_idx = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    return done_idx[1:] - done_idx[:-1]


def split_and_pad_trajectories(data, dones):
    """Split rollouts into separate trajectories and pad them to equal length.

    The function accepts either a plain :class:`torch.Tensor` or a
    :class:`~tensordict.TensorDict`.  The required leading dimensions of the
    input are ``[time, num_envs, …]``.  The operation proceeds as follows:

    1.  Each environment's sequence is segmented at indices where
        ``dones == 1``.
    2.  The resulting variable-length trajectories are concatenated along a
        new *trajectory* axis.
    3.  All trajectories are left-padded with zeros so they share a common
        temporal length ``T_max``.

    Returns
    -------
    padded_data : same type as *data*
        A tensor / TensorDict of shape ``[T_max, N_traj, …]`` where the first
        dimension is time and the second indexes the individual trajectories.
    masks : torch.BoolTensor
        Boolean mask of shape ``[T_max, N_traj]`` identifying the valid (i.e.
        non-padded) elements of *padded_data*.
    """

    # ------------------------------------------------------------------
    # 1. Compute trajectory lengths and helper variables (identical for all
    #    input types).
    # ------------------------------------------------------------------
    trajectory_lengths = _build_trajectory_lengths(dones)
    lengths_list = trajectory_lengths.tolist()
    T_max = data.shape[0]

    # Padding mask (shared for every key / tensor)
    masks = trajectory_lengths > torch.arange(0, T_max, device=data.device).unsqueeze(1)

    # ------------------------------------------------------------------
    # 2. Fast path: plain tensors – keep the original implementation.
    # ------------------------------------------------------------------
    if not isinstance(data, TensorDictBase):
        trajectories = torch.split(data.transpose(1, 0).flatten(0, 1), lengths_list)
        trajectories = trajectories + (torch.zeros(T_max, *data.shape[2:], device=data.device, dtype=data.dtype),)
        padded = torch.nn.utils.rnn.pad_sequence(trajectories)[:, :-1]
        return padded, masks

    # ------------------------------------------------------------------
    # 3. TensorDict path – pad *each stored tensor* individually and then
    #    reconstruct a new TensorDict with a common batch shape.
    # ------------------------------------------------------------------
    padded_dict = {}

    # Flatten env/time dims to mimic the original behaviour
    flat_td = data.transpose(1, 0).flatten(0, 1)

    for key, tensor in flat_td.items():  # type: ignore[attr-defined]
        # Split according to trajectory lengths
        traj_tensors = torch.split(tensor, lengths_list)
        # Append a dummy full-length trajectory so pad_sequence always pads to
        # *exactly* T_max (mirrors legacy logic)
        traj_tensors = list(traj_tensors) + [
            torch.zeros(T_max, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(traj_tensors)[:, :-1]  # (T_max, N_traj, ...)
        padded_dict[key] = padded_tensor

    from tensordict.tensordict import (
        TensorDict,
    )  # local import to avoid global dependency if not needed

    padded_td = TensorDict(padded_dict, batch_size=(T_max, len(lengths_list)))
    return padded_td, masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def read_git_info_from_files(repo_path):
    """Read git information from lightweight .git_info files."""
    # First try the provided repo_path
    git_info_dir = os.path.join(repo_path, ".git_info")

    # If not found, try searching upwards for .git_info (common case in Docker)
    if not os.path.exists(git_info_dir):
        current_path = (
            repo_path if os.path.isdir(repo_path) else os.path.dirname(repo_path)
        )
        # Search up to 5 levels for .git_info directory
        for _ in range(5):
            test_path = os.path.join(current_path, ".git_info")
            if os.path.exists(test_path):
                git_info_dir = test_path
                break
            parent = os.path.dirname(current_path)
            if parent == current_path:  # Reached root
                break
            current_path = parent

    # If still not found, try workspace root (Docker case)
    if not os.path.exists(git_info_dir):
        workspace_root = "/workspace/agile"
        if os.path.exists(workspace_root):
            test_path = os.path.join(workspace_root, ".git_info")
            if os.path.exists(test_path):
                git_info_dir = test_path

    if not os.path.exists(git_info_dir):
        return None

    print(f"Found git info directory at: {git_info_dir}")

    git_info = {}

    # Define all the git info files we expect
    info_files = {
        "commit_hash": "commit_hash",
        "branch": "branch",
        "history": "history",
        "status": "status",
        "diff": "diff",
        "staged_diff": "staged_diff",
    }

    for key, filename in info_files.items():
        file_path = os.path.join(git_info_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    git_info[key] = content if content else f"No {key} available"
            except Exception:
                git_info[key] = f"Error reading {filename}"
        else:
            git_info[key] = f"No {key} available"

    return git_info


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:

        # First try to find a regular git repository
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            use_git_repo = True
            repo_name = pathlib.Path(repo.working_dir).name
            print(f"[store_code_state] Found git repo at: {repo.working_dir}")
        except Exception as e:
            print(
                f"Could not find git repository in {repository_file_path}. Error: {e}"
            )
            # Try to use lightweight git info files instead
            repo_dir = (
                os.path.dirname(repository_file_path)
                if os.path.isfile(repository_file_path)
                else repository_file_path
            )
            git_info = read_git_info_from_files(repo_dir)
            if git_info is None:
                print(
                    f"No git info files found either. Skipping {repository_file_path}."
                )
                continue
            use_git_repo = False

            # Use a consistent repository name for git info files
            repo_name = "agile"

            print(f"[store_code_state] Using git info files for repo: {repo_name}")

        history_file_name = os.path.join(git_log_dir, f"{repo_name}_history.log")

        # Check if file already exists
        if os.path.isfile(history_file_name):
            print(
                f"[store_code_state] History file already exists: {history_file_name}"
            )
            continue

        if use_git_repo:
            # Use regular git repository - get current branch history
            print(f"Storing git history for '{repo_name}' in: {history_file_name}")
            try:
                with open(history_file_name, "x", encoding="utf-8") as f:
                    f.write("--- Current Repository State ---\n")
                    try:
                        f.write(f"Branch: {repo.active_branch.name}\n")
                    except Exception as e:
                        f.write(f"Branch: Unable to determine active branch ({e})\n")
                    f.write(f"Current commit: {repo.head.commit.hexsha}\n")
                    f.write(f"Working directory: {repo.working_dir}\n\n")

                    f.write("--- Current Branch Commit History ---\n")
                    try:
                        # Get the last 50 commits from current branch
                        history = repo.git.log("--oneline", "-n", "50")
                        f.write(history or "No commit history available")
                    except Exception as e:
                        f.write(f"Error getting commit history: {e}")
                    f.write("\n\n")

                    # Add remote information if available
                    f.write("--- Remote Information ---\n")
                    try:
                        for remote in repo.remotes:
                            f.write(f"Remote: {remote.name}\n")
                            for url in remote.urls:
                                f.write(f"  URL: {url}\n")
                            f.write("\n")
                    except Exception as e:
                        f.write(f"Error getting remote info: {e}\n")

                file_paths.append(history_file_name)

            except Exception as e:
                print(f"Error storing git history for {repo_name}: {e}")
        else:
            # Use lightweight git info files
            print(f"Using lightweight git info for '{repo_name}'")

            try:
                with open(history_file_name, "x", encoding="utf-8") as f:
                    f.write("--- Repository State (from .git_info) ---\n")
                    f.write(f"Branch: {git_info.get('branch', 'Unknown')}\n")
                    f.write(
                        f"Current commit: {git_info.get('commit_hash', 'Unknown')}\n\n"
                    )

                    # Add git status information
                    f.write("--- Git Status (Uncommitted Changes) ---\n")
                    status = git_info.get("status", "No status available")
                    if status and status != "No status available":
                        f.write("WARNING: Repository has uncommitted changes!\n")
                        f.write(f"{status}\n")
                    else:
                        f.write("Working directory is clean (no uncommitted changes)\n")
                    f.write("\n")

                    # Add diff information
                    f.write("--- Uncommitted Changes (git diff) ---\n")
                    diff = git_info.get("diff", "No diff available")
                    if diff and diff != "No diff available":
                        f.write(f"{diff}\n")
                    else:
                        f.write("No uncommitted changes\n")
                    f.write("\n")

                    # Add staged changes
                    f.write("--- Staged Changes (git diff --cached) ---\n")
                    staged_diff = git_info.get("staged_diff", "No staged changes")
                    if staged_diff and staged_diff != "No staged changes":
                        f.write(f"{staged_diff}\n")
                    else:
                        f.write("No staged changes\n")
                    f.write("\n")

                    # Add commit history
                    f.write("--- Commit History ---\n")
                    f.write(git_info.get("history", "No history available"))
                    f.write("\n")

                file_paths.append(history_file_name)
                print(
                    f"[store_code_state] Successfully created history file: {history_file_name}"
                )
            except Exception as e:
                print(f"Error writing git history from info files for {repo_name}: {e}")

    print(f"[store_code_state] Generated {len(file_paths)} files: {file_paths}")
    return file_paths

def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name (str): The function name. The format should be 'module:attribute_name'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.

    Returns:
        Callable: The function loaded from the module.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)


def flatten_dict(td: TensorDictBase) -> torch.Tensor:
    """Flatten a TensorDict into a single tensor.

    Args:
        td (TensorDictBase): The TensorDict to flatten.

    Returns:
        torch.Tensor: The flattened tensor.
    """
    return torch.cat([td[key].flatten(td.batch_dims) for key in td.keys()], dim=-1)
