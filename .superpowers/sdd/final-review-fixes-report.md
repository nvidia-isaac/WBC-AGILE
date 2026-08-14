# Final review fixes report

## Scope

Remediate the final security review at base commit `71610173` as one commit:

- make `promote_release.py` the sole public-repository writer;
- close existing public and internal tag retry races without weakening immutability;
- remove superseded public publication contracts and tests;
- bound GitHub API calls while preserving sanitized errors;
- preserve release authority, atomicity, credentials, public boundary, CI behavior, and unrelated files.

## TDD evidence

### RED

Command (global Git configuration disabled):

```bash
GIT_CONFIG_GLOBAL=/dev/null pytest -q \
  tests/test_release_wrapper.py::test_parse_args_rejects_direct_github_publication \
  tests/test_public_release_promotion.py::test_existing_public_tag_is_an_exact_leased_noop_in_atomic_push \
  tests/test_public_release_promotion.py::test_internal_existing_tag_rejects_remote_movement_after_validation \
  tests/test_public_release_promotion.py::test_github_release_client_creates_versioned_release_without_exposing_token
```

Result: `4 failed` for the intended missing behavior:

1. `release.py --to-github` did not raise `SystemExit`.
2. Existing public tags had neither an exact tag lease nor a captured-object no-op refspec.
3. Internal tag movement after validation was accepted.
4. `urlopen()` received no `timeout` argument.

The internal movement harness trigger was subsequently corrected from `command[-4:]` to `command[-3:]`; the corrected harness physically force-moves the test remote tag after content validation and is covered by the final GREEN suite.

### Implementation

- Removed the `--to-github` action, public branch bootstrap helper, GitHub write selection, and public confirmation path from `release.py`; authenticated public dry-run and local folder generation remain.
- Existing public annotated tag retries now add `<captured-tag-object>:refs/tags/<version>` and `--force-with-lease=refs/tags/<version>:<captured-tag-object>` to the same atomic push.
- Existing internal annotated tag retries now capture the validated local tag object and perform an exact leased no-op push before returning `existing`.
- Removed `preflight_public_refs()`, `publish_main_and_tag()`, `validate_public_copy()`, `public_destination_refs()`, and their legacy tests.
- Added `GITHUB_API_TIMEOUT = 30` to every GitHub Release API `urlopen()` call.
- Updated the release runbook and CI testing documentation to state the sole-writer and exact retry contracts.

## Verification

Fresh completion-gate results:

- Focused release implementation suite:
  `GIT_CONFIG_GLOBAL=/dev/null pytest -q tests/test_release_wrapper.py tests/test_public_release_promotion.py tests/test_copybara_staging_policy_assets.py`
  -> `111 passed in 1.16s`.
- Public-boundary suite in a temporary tracked-tree snapshot containing only this diff:
  `GIT_CONFIG_GLOBAL=/dev/null pytest -q tests/test_public_release_boundary.py`
  -> `18 passed in 0.15s`. The clean snapshot avoided unrelated ignored workspace artifacts; the source workspace artifacts were preserved. `git archive` emitted a pre-existing Git LFS pointer warning for `docs/videos/unitree_g1_dancing_sim.gif`, but exited successfully and all boundary tests passed.
- Scoped non-manual pre-commit hooks on all six modified code/test/doc files -> all passed.
- GitLab CI syntax: `pre-commit run check-yaml --files .gitlab-ci.yml` -> passed.
- CI contract tests for validation provenance and protected serialized promotion -> `2 passed`.
- `python -m py_compile` for both release scripts and focused tests -> passed.
- `git diff --check` -> passed.
- Static scans found no `--to-github` in production/docs/CI and no remaining definitions or references to `public_destination_refs`, `preflight_public_refs`, `publish_main_and_tag`, or `validate_public_copy` outside historical design documents.

## Workspace preservation

Pre-existing untracked `AGENTS.md` and `external_assets/`, plus ignored logs, build outputs, and prior `.superpowers` artifacts, were not modified or removed. The required report is force-added because `.superpowers/sdd/` is intentionally ignored.

## Senior re-review documentation correction

A follow-up review found that `docs/source/testing.md` still described a removed `publish_public`
job and incorrectly implied that `validation-release.json` supplied promotion authority.

### RED

Added `test_testing_guide_documents_current_release_authority` and ran it alone. It failed because
the guide still contained `` `publish_public` ``. The new contract also rejects validation-manifest
consumption language and requires both current job names, protected-branch resolution at job start,
and each job's fixed public-main policy.

### GREEN and validation

- Focused docs contract -> `1 passed`.
- Clean tracked-tree public-boundary suite -> `19 passed in 0.16s`.
- Repository docs build (`./docs/build.sh`) -> succeeded. It reported one pre-existing unrelated
  warning at `docs/source/evaluation.md:262` for missing MyST target
  `analyzing-trajectories-python-jupyter`; `docs/source/testing.md` produced no warning.
- A strict `sphinx-build -W --keep-going` read and rendered `testing.md`, then failed only on that
  same unrelated `evaluation.md:262` warning.
