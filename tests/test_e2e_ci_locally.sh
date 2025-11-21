#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#=============================================================================
# TEST E2E IN CI ENVIRONMENT LOCALLY
#
# This script simulates the GitLab CI E2E test environment locally by running
# the same Docker image and commands that CI uses for end-to-end testing.
#=============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== TESTING E2E IN CI ENVIRONMENT LOCALLY ===${NC}"

# Docker image used in CI
DOCKER_IMAGE="nvcr.io/nvidia/isaac-lab:2.3.0"

echo -e "${YELLOW}This will pull and run the Docker image: ${DOCKER_IMAGE}${NC}"
echo -e "${YELLOW}Make sure you have Docker installed and running.${NC}"
echo ""

# Parse command line arguments
RUN_UNIT_TESTS=false
RUN_E2E_TESTS=true
SPECIFIC_TASK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT_TESTS=true
            RUN_E2E_TESTS=false
            shift
            ;;
        --all)
            RUN_UNIT_TESTS=true
            RUN_E2E_TESTS=true
            shift
            ;;
        --task)
            SPECIFIC_TASK="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --unit        Run unit tests only"
            echo "  --all         Run both unit and E2E tests"
            echo "  --task TASK   Test specific task (e.g., Velocity-G1-v0)"
            echo "  --help        Show this help message"
            echo ""
            echo "Default: Run E2E tests only"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show what will be tested
echo -e "${GREEN}Test configuration:${NC}"
echo -e "  Unit tests: $([ "$RUN_UNIT_TESTS" = true ] && echo "${GREEN}Yes${NC}" || echo "${YELLOW}No${NC}")"
echo -e "  E2E tests:  $([ "$RUN_E2E_TESTS" = true ] && echo "${GREEN}Yes${NC}" || echo "${YELLOW}No${NC}")"
if [ -n "$SPECIFIC_TASK" ]; then
    echo -e "  Specific task: ${BLUE}$SPECIFIC_TASK${NC}"
fi
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Check for GPU support
GPU_AVAILABLE="false"
DOCKER_GPU_ARGS=""

# Method 1: Check for nvidia-docker (legacy)
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
    GPU_AVAILABLE="true"
    echo -e "${GREEN}Found nvidia-docker (legacy), GPU support enabled${NC}"
# Method 2: Check for nvidia-container-toolkit (modern)
elif docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    DOCKER_CMD="docker"
    DOCKER_GPU_ARGS="--gpus all"
    GPU_AVAILABLE="true"
    echo -e "${GREEN}Found nvidia-container-toolkit, GPU support enabled${NC}"
else
    DOCKER_CMD="docker"
    echo -e "${YELLOW}No GPU support detected. Tests will likely fail!${NC}"
    echo -e "${YELLOW}To enable GPU support, install nvidia-container-toolkit:${NC}"
    echo -e "${YELLOW}  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    echo ""
    echo -e "${RED}WARNING: Isaac Sim requires CUDA. E2E tests WILL fail without GPU.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Pull the image if needed
echo -e "${GREEN}Pulling Docker image (if needed)...${NC}"
docker pull ${DOCKER_IMAGE} 2>/dev/null || true

# Check if we're in a TTY
if [ -t 0 ]; then
    DOCKER_FLAGS="-it"
else
    DOCKER_FLAGS=""
fi

# Build the test command
TEST_COMMANDS=""

if [ "$RUN_UNIT_TESTS" = true ]; then
    TEST_COMMANDS="${TEST_COMMANDS}
        echo ''
        echo '========================================'
        echo 'Running Unit Tests'
        echo '========================================'
        ./tests/run_unit_tests.sh
        echo ''
    "
fi

if [ "$RUN_E2E_TESTS" = true ]; then
    if [ -n "$SPECIFIC_TASK" ]; then
        # Test specific task
        TEST_COMMANDS="${TEST_COMMANDS}
            echo ''
            echo '========================================'
            echo 'Testing Specific Task: ${SPECIFIC_TASK}'
            echo '========================================'

            # Create a temporary test file for the specific task
            cat > /tmp/test_specific_task.py << 'EOF'
#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

task = '${SPECIFIC_TASK}'
project_root = Path('/workspace/agile').absolute()
train_script = project_root / 'scripts' / 'train.py'
isaaclab_path = os.environ.get('ISAACLAB_PATH', '/workspace/isaaclab')
isaaclab_script = os.path.join(isaaclab_path, 'isaaclab.sh')

print(f'Testing task: {task}')

cmd = [
    isaaclab_script, '-p', str(train_script),
    '--task', task,
    '--max_iterations', '2',
    '--num_envs', '4',
    '--headless',
    '--logger', 'wandb',
    '--run_name', f'test_{task}'
]

env = dict(os.environ)
env['WANDB_MODE'] = 'disabled'
env['OMNI_HEADLESS'] = '1'
env['DISPLAY'] = ':1'

print(f'Running: {\" \".join(cmd)}')
result = subprocess.run(cmd, env=env, capture_output=True, text=True)

if result.returncode != 0:
    print('STDERR:', result.stderr[-2000:] if result.stderr else 'No stderr')
    sys.exit(1)

print(f'✅ Task {task} passed!')
EOF

            \${ISAACLAB_PATH}/isaaclab.sh -p /tmp/test_specific_task.py
        "
    else
        # Run all E2E tests
        TEST_COMMANDS="${TEST_COMMANDS}
            echo ''
            echo '========================================'
            echo 'Running E2E Tests'
            echo '========================================'
            echo ''
            echo 'Testing all task environments...'
            \${ISAACLAB_PATH}/isaaclab.sh -p tests/test_all_tasks_e2e.py
            \${ISAACLAB_PATH}/isaaclab.sh -p tests/test_deterministic_eval_e2e.py
            echo 'Testing Sim2MuJoCo pipeline...'
            \${ISAACLAB_PATH}/isaaclab.sh -p tests/test_sim2mujoco_e2e.py
        "
    fi
fi

# Run the container with tests
echo -e "${GREEN}Starting container for testing...${NC}"

${DOCKER_CMD} run \
    --rm \
    ${DOCKER_FLAGS} \
    ${DOCKER_GPU_ARGS} \
    -v "$(pwd):/workspace/agile" \
    -w /workspace/agile \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=:1 \
    -e OMNI_HEADLESS=1 \
    -e CI=true \
    -e ISAACLAB_PATH=/workspace/isaaclab \
    --entrypoint /bin/bash \
    ${DOCKER_IMAGE} \
    -c "
        set -e
        echo '=== Running Tests in CI Environment ==='
        echo ''

        # Install system dependencies (same as CI)
        echo '📦 Installing dependencies...'
        apt-get update > /dev/null 2>&1 && apt-get install -y python3-pip git > /dev/null 2>&1

            # Fix Git ownership issue in Docker
            git config --global --add safe.directory /workspace/agile

            # Initialize git submodules (for robot assets)
            echo '📂 Initializing git submodules...'
            # Note: In local testing, we assume submodules are already initialized on host
            # Just verify they exist
            if [ ! -d 'agile/rl_env/assets/robot_menagerie/unitree' ]; then
                echo 'Warning: Robot assets not found. Please run: git submodule update --init --recursive'
            fi

            # Install agile with CI script
        echo '🔧 Installing agile...'
        ./scripts/setup/install_deps_ci.sh

        # Run the tests
        ${TEST_COMMANDS}

        echo ''
        echo '=== ALL TESTS COMPLETE ==='
    "

echo -e "${GREEN}Local CI test completed!${NC}"
