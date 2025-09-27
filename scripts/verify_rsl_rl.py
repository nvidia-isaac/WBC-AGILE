#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Verification script for custom rsl_rl installation.

This script ensures that the custom rsl_rl package (with TensorDict support)
is correctly installed and takes precedence over Isaac Lab's bundled version.

Usage:
    ${ISAACLAB_PATH}/isaaclab.sh -p scripts/verify_rsl_rl.py

The script will exit with code 0 if successful, 1 if verification fails.
"""

import importlib.util
import subprocess
import sys


def check_rsl_rl():
    """Check which rsl_rl is being used and verify it has TensorDict support."""

    # First check what's installed via pip
    print("Checking installed packages...")
    try:
        result = subprocess.run(["pip", "list", "--format=freeze"], capture_output=True, text=True, check=True)
        for line in result.stdout.split("\n"):
            if "rsl" in line.lower() and "rl" in line.lower():
                print(f"  Found: {line}")
    except Exception as e:
        print(f"  Could not check pip list: {e}")

    print("\nChecking rsl_rl import...")
    try:
        # Import rsl_rl and check its location
        import rsl_rl

        print("✓ rsl_rl imported successfully")
        print(f"  Module location: {rsl_rl.__file__}")

        # Check if it's in the expected custom location
        if "/workspace/agile/agile/algorithms/rsl_rl" in str(rsl_rl.__file__):
            print("✓ Using custom rsl_rl from agile/algorithms/rsl_rl")
        elif "site-packages" in str(rsl_rl.__file__):
            print("⚠ Using rsl_rl from site-packages (might be Isaac Lab's version)")

        # Check if it's the custom version by looking for TensorDict import
        runner_spec = importlib.util.find_spec("rsl_rl.runners.on_policy_runner")
        if runner_spec and runner_spec.origin:
            print(f"  on_policy_runner location: {runner_spec.origin}")

            # Check if the file contains TensorDict import (indicating custom version)
            with open(runner_spec.origin) as f:
                content = f.read()
                if "from tensordict.tensordict import TensorDict" in content:
                    print("✓ Custom rsl_rl version detected (has TensorDict support)")
                    return True
                else:
                    print("✗ Standard rsl_rl version detected (no TensorDict support)")
                    print("  This appears to be the Isaac Lab bundled version")
                    print("  The custom version should be installed from agile/algorithms/rsl_rl")
                    return False
        else:
            print("✗ Could not find on_policy_runner module")
            return False

    except ImportError as e:
        print(f"✗ Failed to import rsl_rl: {e}")
        return False


if __name__ == "__main__":
    success = check_rsl_rl()
    if not success:
        print("\n⚠ Installation verification failed!")
        print("The custom rsl_rl package was not properly installed.")
        sys.exit(1)
    else:
        print("\n✓ Installation verified successfully!")
        sys.exit(0)
