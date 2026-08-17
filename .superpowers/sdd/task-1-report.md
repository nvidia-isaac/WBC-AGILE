# Task 1 report: stable branch-head/staging proof

## Changed files

- `promote_release.py`
  - Removes CI-supplied `--source-sha` and artifact `--validation-manifest` authority.
  - Resolves the protected internal `release-X.Y` head with the GitLab credential, verifies staging provenance through `release.verify_destination_provenance`, then re-probes the source branch before any public-copy work.
  - Carries that resolved SHA as the immutable source snapshot through public candidate generation, public-tag attribution, and the final manifest.
- `tests/test_public_release_promotion.py`
  - Replaces manifest-attestation coverage with protected-head, malformed/absent head, source movement, exact staging provenance label, staging movement, and CLI-contract tests.
  - Updates retained cleanup/publication tests for the immutable snapshot API.
- `tests/test_release_wrapper.py`
  - Renames the existing stable-head provenance test to reflect the shared staging-provenance mechanism used by promotion.

## TDD evidence

RED command:

```bash
uv run --frozen pytest -q tests/test_public_release_promotion.py tests/test_release_wrapper.py
```

RED result: `21 failed, 90 passed in 1.22s`. The new tests failed because `resolve_source_snapshot`, `verify_source_snapshot`, and `verify_staging_provenance` did not yet exist; the remaining failures were the expected old-test/removed-argument contract mismatch.

GREEN command:

```bash
uv run --frozen pytest -q tests/test_public_release_promotion.py tests/test_release_wrapper.py
```

GREEN result: `98 passed in 1.03s` (re-run after final formatting-only correction also completed successfully). The final full rerun passed as `98 passed in 0.59s` with `GIT_CONFIG_GLOBAL=/dev/null` to isolate the developer machine's unavailable SSH commit-signing configuration.

Additional verification:

```bash
python -m py_compile promote_release.py release.py
git diff --check
```

Both completed successfully. A targeted `uv run --frozen ruff check ...` could not run because `ruff` is not installed in the frozen environment.

## Self-review

- The release branch SHA is now read from the internal GitLab remote only after the combined credential store is created; it is no longer accepted from CLI or a downloaded artifact.
- The staging ref is checked by the existing exact-one-`GitOrigin-RevId` and stable-head helper in `release.py`.
- The internal source branch is probed again after staging verification and before public snapshots, Copybara dry run, candidate creation, or publication.
- Existing public snapshot, fast-forward proof, leased atomic push, protected environment/resource-group configuration, and credential cleanup paths were not changed.
- The final manifest records the resolved source SHA and observed staging commit for auditability, but neither is trusted input.

## Concerns

- `publish_public` in `.gitlab-ci.yml` still invokes the removed CLI flags. The Task 1 brief explicitly limited modifications to `promote_release.py` and the two focused test files, so this report does not change CI configuration; that invocation must be updated by the owning follow-on task before this workflow is runnable.
- The frozen test environment lacks `ruff`, so lint was not independently executed; focused tests, Python compilation, and whitespace validation passed. A plain final rerun hit the workstation's global `commit.gpgsign=true`/SSH-signing setup in tests that make temporary commits; isolating that global configuration yielded 98/98 passing tests.
