#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Script to run unit tests for the project


# Color codes for pretty output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Global flag to track if any test failed
ANY_TEST_FAILED=0

check_isaaclab_path() {
    # Check if ISAACLAB_PATH is set
    if [ -z "${ISAACLAB_PATH}" ]; then
        echo -e "${RED}Error: ISAACLAB_PATH environment variable is not set.${NC}"
        echo "Please set it to the path of your Isaac Lab installation."
        echo "Example: export ISAACLAB_PATH=/path/to/isaac_lab"
        return 1  # Return error code
    fi

    # Check if the path actually exists
    if [ ! -d "${ISAACLAB_PATH}" ]; then
        echo -e "${RED}Error: Directory specified by ISAACLAB_PATH does not exist: ${ISAACLAB_PATH}${NC}"
        echo "Please ensure the path is correct."
        return 1
    fi

    echo -e "${GREEN}Using Isaac Lab installation at: ${ISAACLAB_PATH}${NC}"
    return 0  # Return success code
}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Get command line arguments
TEST_PATTERN="test_*.py"
VERBOSE=false
SHOW_HELP=false
EXCLUDE_TESTS=()

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -e|--exclude)
            EXCLUDE_TESTS+=("$2")
            shift 2
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            SHOW_HELP=true
            shift
            ;;
    esac
done

# Show help message
if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --pattern PATTERN  File pattern for test files (default: test_*.py)"
    echo "  -e, --exclude FILE     Exclude a specific test file (can be used multiple times)"
    echo "  -v, --verbose          Show more detailed output"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -p 'test_*.py' -v"
    echo "  $0 -e test_slow_test.py -e test_gpu_intensive.py"
    exit 0
fi

# Define fixed list of test files to exclude (E2E tests)
ALWAYS_EXCLUDE=(
    "tests/test_all_tasks_e2e.py"
)

# Combine the fixed exclusion list with command-line exclusions
ALL_EXCLUDE=("${ALWAYS_EXCLUDE[@]}" "${EXCLUDE_TESTS[@]}")

# Function to check if a file should be excluded
is_excluded() {
    local test_file="$1"
    local filename=$(basename "$test_file")

    for excluded in "${ALL_EXCLUDE[@]}"; do
        if [[ "$filename" == "$excluded" || "$test_file" == *"$excluded"* ]]; then
            return 0  # True, should be excluded
        fi
    done

    return 1  # False, should not be excluded
}

# Print header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running unit tests for AGILE${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Test pattern: $TEST_PATTERN"
if [ ${#ALL_EXCLUDE[@]} -gt 0 ]; then
    echo "Excluded tests:"
    for excluded in "${ALL_EXCLUDE[@]}"; do
        echo "  - $excluded"
    done
fi
echo -e "${BLUE}========================================${NC}"

    # Activate the conda environment if it exists and is not already activated
    if [ -n "$CONDA_PREFIX" ]; then
        echo "Using conda environment: $CONDA_PREFIX"
    else
        echo "No conda environment activated. Consider activating your environment before running tests."
    fi

    # Always check for Isaac Lab - fail loudly if not available
    echo "Checking ISAACLAB_PATH..."
    check_isaaclab_path
    if [ $? -ne 0 ]; then
        # Handle the error (e.g., exit)
        exit 1
    fi

# Define test directories to scan
TEST_DIRS=(
    "$PROJECT_ROOT/agile/rl_env/tests"
    "$PROJECT_ROOT/agile/algorithms/evaluation/tests"
)

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
FAILED_TEST_FILES=()

# More comprehensive failure patterns to detect
FAILURE_PATTERNS=(
    "FAILED"
    "ERROR"
    "FAILURES"
    "ERRORS"
    "Traceback"
    "Exception"
    "AssertionError"
    "ImportError"
    "exit code: 1"
    "exit status 1"
    "Ran 0 tests"  # Often indicates test discovery failure
)

# Function to check if output contains failure indicators
contains_failure() {
    local output="$1"

    for pattern in "${FAILURE_PATTERNS[@]}"; do
        if echo "$output" | grep -q "$pattern"; then
            return 0  # True, contains failure
        fi
    done

    return 1  # False, no failure detected
}

# Function to run a single test file
run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .py)
    local test_dir=$(dirname "$test_file")
    local relative_path=${test_file#$PROJECT_ROOT/}
    local test_failed=0
    local exit_code=0

    # Check if the file should be excluded
    if is_excluded "$test_file"; then
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        if [ "$VERBOSE" = true ]; then
            echo -e "${YELLOW}Skipping excluded test: $relative_path${NC}"
        fi
        return 0
    fi

    if [ "$VERBOSE" = true ]; then
        echo -e "\n\n${BLUE}========================================${NC}"
        echo -e "${BLUE}Running test: $relative_path${NC}"
        echo -e "${BLUE}========================================${NC}"
    else
        echo -e "Testing: ${YELLOW}$relative_path${NC}"
    fi

    if [ -f "$test_file" ]; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        # Set PYTHONPATH to include the project root so imports work correctly
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

        # For CI, set additional environment variables to help with Isaac Sim
        if [ "$CI" = "true" ]; then
            export OMNI_KIT_ACCEPT_EULA=YES
            export ACCEPT_EULA=Y
        fi

        # Run the test and capture output
        OUTPUT=$(${ISAACLAB_PATH}/isaaclab.sh -p "$test_file" 2>&1)
        exit_code=$?

        # Check for failure both via exit code and output content
        if [ $exit_code -ne 0 ] || contains_failure "$OUTPUT"; then
            test_failed=1
        fi

        # If output doesn't contain any test results, that's also a failure
        if ! echo "$OUTPUT" | grep -q "test"; then
            test_failed=1
        fi

        if [ $test_failed -eq 0 ]; then
            PASSED_TESTS=$((PASSED_TESTS + 1))
            if [ "$VERBOSE" = true ]; then
                echo -e "${GREEN}✅ Test $test_name passed${NC}"
                echo "$OUTPUT"
            else
                echo -e "  ${GREEN}✅ Passed${NC}"
            fi
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            FAILED_TEST_FILES+=("$relative_path")
            echo -e "${RED}❌ Test $test_name failed${NC}"
            echo "$OUTPUT"
            echo -e "${RED}Exit code: $exit_code${NC}"
            ANY_TEST_FAILED=1  # Set global flag
            return 1  # Return failure status
        fi
    else
        echo -e "${RED}❌ Error: Test file $test_file not found${NC}"
        ANY_TEST_FAILED=1  # Set global flag
        return 1  # Return failure status
    fi

    return 0  # Return success status
}

# Find and run all tests in the specified directories
echo -e "\n${BLUE}Discovering tests...${NC}"
ALL_TEST_FILES=()

for dir in "${TEST_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        if [ "$VERBOSE" = true ]; then
            echo "Scanning directory: $dir"
        fi
        FOUND_FILES=$(find "$dir" -name "$TEST_PATTERN" -type f | sort)
        if [ -n "$FOUND_FILES" ]; then
            while IFS= read -r file; do
                ALL_TEST_FILES+=("$file")
            done <<< "$FOUND_FILES"
        fi
    else
        echo -e "${YELLOW}Warning: Directory $dir not found, skipping...${NC}"
    fi
done

# Report test discovery results
FOUND_COUNT=${#ALL_TEST_FILES[@]}
if [ $FOUND_COUNT -eq 0 ]; then
    echo -e "${YELLOW}No test files matching $TEST_PATTERN found in the specified directories${NC}"
    exit 0
else
    echo -e "${GREEN}Found $FOUND_COUNT test files${NC}"
fi

# Run all discovered tests
echo -e "\n${BLUE}Running tests...${NC}"

for test_file in "${ALL_TEST_FILES[@]}"; do
    # Run each test but don't exit on failure
    run_test "$test_file" || true
done

# Print summary
echo -e "\n\n${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total discovered tests: $FOUND_COUNT"
echo "Total executed tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
if [ $SKIPPED_TESTS -gt 0 ]; then
    echo -e "Skipped: ${YELLOW}$SKIPPED_TESTS${NC}"
fi

if [ $FAILED_TESTS -gt 0 ] || [ $ANY_TEST_FAILED -eq 1 ]; then
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "\n${RED}Failed tests:${NC}"
    for failed in "${FAILED_TEST_FILES[@]}"; do
        echo -e "  ${RED}$failed${NC}"
    done
    echo -e "\n${RED}❌ Some tests failed${NC}"
    exit 1
else
    echo -e "\n${GREEN}✅ All tests passed successfully!${NC}"
    exit 0
fi
