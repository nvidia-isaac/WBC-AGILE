# Task 4: Internal annotated tag after public completion — DONE

## Outcome

Implemented the final internal release-tag step in `promote_release.py`.

- The internal `vX.Y.Z` tag is an annotated tag at the immutable internal source SHA.
- Its annotation reuses the canonical checked-in release-note body and appends the independently observed internal source, staging, and public SHAs.
- A temporary bare repository fetches the exact source SHA using the existing credential-store command path.
- Creation uses the `AGILE Release Automation <agile-release@nvidia.com>` identity and pushes only the tag with an expected-absence lease.
- An existing internal tag is accepted only if it is annotated, peels to the exact source SHA, and has the exact canonical annotation. Lightweight, target, or message mismatches fail closed.
- GitHub Release reconciliation completes before any internal-tag attempt.
- A retry with already-complete public refs can create a missing internal tag without changing any public ref object ID.
- `public-release.json` now reports source/staging/public SHAs, release-note path and SHA-256 digest, GitHub Release result, internal-tag result, pipeline URL, approver, and whether `main` actually advanced.
- Post-public GitHub or internal-tag failures retain a diagnostic incomplete manifest and return nonzero; the manifest remains output only.

Files changed:

- `promote_release.py`
- `tests/test_public_release_promotion.py`
- `.superpowers/sdd/task-4-report.md`

Preserved without modification: untracked `AGENTS.md` and `external_assets/`.

## TDD evidence

### RED 1: internal annotated-tag contract

Command:

```text
GIT_CONFIG_GLOBAL=/dev/null uv run --frozen pytest -q tests/test_public_release_promotion.py -k internal_annotated_tag
```

Expected result before implementation: `5 failed, 72 deselected`.

All five failures were `AttributeError` for the missing `reconcile_internal_annotated_tag` helper. The tests covered creation, exact existing-tag idempotence, and target/message/lightweight mismatches.

### GREEN 1

The same focused command passed:

```text
5 passed, 72 deselected in 0.30s
```

### RED 2: public-completion sequencing and diagnostic manifest

Command:

```text
GIT_CONFIG_GLOBAL=/dev/null uv run --frozen pytest -q tests/test_public_release_promotion.py -k 'main_publishes_and_writes_final_provenance or main_does_not_attempt_internal_tag_before'
```

Expected result before orchestration wiring: `2 failed, 76 deselected`.

The success path never called the internal helper, and the GitHub-incomplete manifest lacked `internal_tag_result: not_attempted`.

### GREEN 2

The combined Task 4 set passed:

```text
7 passed, 71 deselected in 11.29s
```

### RED 3: retryable internal failure diagnostic

Command:

```text
GIT_CONFIG_GLOBAL=/dev/null uv run --frozen pytest -q tests/test_public_release_promotion.py -k reports_retryable_internal_tag_failure
```

Expected result before the diagnostic fix: `1 failed, 79 deselected`.

The internal-tag failure returned nonzero but removed `public-release.json`, so no confirmed GitHub result or incomplete internal-tag result remained.

### GREEN 3

The same focused command passed:

```text
1 passed, 79 deselected in 0.09s
```

## Final verification

Full focused promotion suite:

```text
GIT_CONFIG_GLOBAL=/dev/null uv run --frozen pytest -q tests/test_public_release_promotion.py
........................................................................ [ 90%]
........                                                                 [100%]
80 passed in 2.03s
```

Scoped repository hooks:

```text
GIT_CONFIG_GLOBAL=/dev/null uv run --frozen pre-commit run --files promote_release.py tests/test_public_release_promotion.py
```

Result: all applicable hooks passed, including Python AST, whitespace, private-key detection, Ruff lint, and Ruff format.

## Concerns

No product-code concerns remain in Task 4 scope. CI wiring and operator documentation were intentionally not changed.

The required `apply_patch` editor was unavailable because the environment sandbox helper could not initialize loopback networking. Per the task's explicit fallback authorization, edits were made with narrowly scoped explicit file-replacement scripts and inspected through Git diffs.

## Review fix: reject nested annotated tags

A blocking review finding identified that recursive `refs/tags/<version>^{}` peeling could accept an outer annotated tag whose direct target was another tag object, as long as the chain eventually reached the source commit.

TDD RED:

```text
GIT_CONFIG_GLOBAL=/dev/null uv run --frozen pytest -q tests/test_public_release_promotion.py -k nested_tag
F                                                                        [100%]
1 failed, 80 deselected
```

The real nested-tag fixture peeled to the expected source and was incorrectly accepted.

GREEN:

```text
GIT_CONFIG_GLOBAL=/dev/null uv run --frozen pytest -q tests/test_public_release_promotion.py -k 'nested_tag or internal_annotated_tag'
......                                                                   [100%]
6 passed, 75 deselected in 0.30s
```

Validation now compares the annotated tag object's direct first `object <sha>` header to the exact resolved source SHA. Recursive peeling is no longer used, so tag-to-tag indirection fails closed while exact existing annotated tags remain idempotent.
