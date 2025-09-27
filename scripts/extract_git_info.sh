#!/bin/bash
# Script to extract git information before Docker build

set -e

echo "Extracting git information..."

# Create .git_info directory
mkdir -p .git_info

# Extract commit hash
(git rev-parse HEAD 2>/dev/null || echo "unknown") > .git_info/commit_hash

# Extract branch name
(git branch --show-current 2>/dev/null || git symbolic-ref --short HEAD 2>/dev/null || echo "unknown") > .git_info/branch

# Extract commit history (last 50 commits)
(git log --oneline -n 50 2>/dev/null || echo "No history available") > .git_info/history

# Extract git status to capture uncommitted changes
(git status --porcelain 2>/dev/null || echo "No status available") > .git_info/status

# Extract git diff to show exact uncommitted changes
(git diff HEAD 2>/dev/null || echo "No diff available") > .git_info/diff

# Extract staged changes (if any)
(git diff --cached 2>/dev/null || echo "No staged changes") > .git_info/staged_diff

echo "Git information extracted to .git_info/"
echo "  Commit: $(cat .git_info/commit_hash)"
echo "  Branch: $(cat .git_info/branch)"
echo "  History: $(wc -l < .git_info/history) commits"

# Check for uncommitted changes and warn user
if [ -s .git_info/status ]; then
    echo "  ⚠️  WARNING: Uncommitted changes detected!"
    echo "  Modified files:"
    cat .git_info/status | head -10
    if [ $(wc -l < .git_info/status) -gt 10 ]; then
        echo "  ... and $(( $(wc -l < .git_info/status) - 10 )) more files"
    fi
    echo "  These changes will be included in the Docker image for reproducibility."
else
    echo "  ✅ Working directory is clean (no uncommitted changes)"
fi
