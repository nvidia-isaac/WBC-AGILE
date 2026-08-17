#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a command as a labelled CI step and report how long it took.
#
# Usage:
#   bash scripts/ci/execute_with_timer.sh "<label>" <command> [args...]
#
# Wraps the command in a GitLab CI collapsible section (the job log shows a
# duration badge and the step can be folded) and also prints an explicit
# elapsed-time line so the timing is visible in plain-text logs too.

# Note: no `set -e` -- we must capture the command's exit status and still
# print the timing/close the section before propagating it.
set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <label> <command> [args...]" >&2
  exit 2
fi

label=$1
shift

# GitLab section names must contain only alphanumerics and underscores.
section="timer_$(printf '%s' "$label" | tr -c '[:alnum:]' '_')"
clear=$'\033[0K'
cyan=$'\033[36m'
green=$'\033[32m'
red=$'\033[31m'
reset=$'\033[0m'

printf '%ssection_start:%s:%s\r%s%s>>> %s%s\n' \
  "$clear" "$(date +%s)" "$section" "$clear" "$cyan" "$label" "$reset"

start_ns=$(date +%s%N)
"$@"
status=$?
elapsed_ms=$(( ($(date +%s%N) - start_ns) / 1000000 ))
mins=$(( elapsed_ms / 60000 ))
secs=$(( (elapsed_ms / 1000) % 60 ))

printf '%ssection_end:%s:%s\r%s' "$clear" "$(date +%s)" "$section" "$clear"
if [ "$status" -eq 0 ]; then
  printf '%s<<< %s finished in %dm%02ds (%dms)%s\n' \
    "$green" "$label" "$mins" "$secs" "$elapsed_ms" "$reset"
else
  printf '%s<<< %s FAILED after %dm%02ds (%dms), exit %d%s\n' \
    "$red" "$label" "$mins" "$secs" "$elapsed_ms" "$status" "$reset"
fi

exit "$status"
