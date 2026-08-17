# Task 3 report: annotated public tags and GitHub Releases

## TDD evidence

RED: focused tests for annotated tag construction, local annotated-tag atomic refspec, GitHub Release creation, matching-release no-op, and tag/target/body conflicts initially failed because the tag creator, HTTPS client, and local tag refspec did not exist (7 failures).

GREEN:

```text
uv run --frozen pytest -q tests/test_public_release_promotion.py
70 passed in 1.15s
```

`uv run --frozen python -m py_compile promote_release.py` and `git diff --check` also passed.

## Delivered

- An annotated `vX.Y.Z` tag at the generated public commit, preserving checked-in Markdown verbatim and appending Internal-Source-SHA, Staging-Commit-SHA, and Public-Commit-SHA.
- Atomic leased publication uses the local tag object (`refs/tags/vX.Y.Z:refs/tags/vX.Y.Z`) alongside the release branch and optional main.
- Post-push stdlib HTTPS GitHub Release reconciliation: GET-by-tag retries, matching releases no-op, conflicts fail closed, and the token appears only in Authorization.
- A GitHub API failure after the atomic push returns nonzero but retains a manifest with `github_release_state: incomplete` for safe retry.

## Scope / concern

Only Task 3 promotion code and tests changed; internal tags, CI, and docs automation are untouched. The optional standalone `ruff` executable is unavailable, but the repository pre-commit formatter ran during commit preparation.

## Review fix: existing public tag validation

Added retry-time validation after the generated public candidate is known. An existing tag is fetched into the candidate repository and must be an annotated `tag` object, peel exactly to the generated public commit, and have a tag message exactly equal to canonical release notes plus the three SHA footer fields. Focused RED coverage rejected lightweight tags and mismatched provenance footers; GREEN was `3 passed`. The complete promotion suite then passed with `72 passed` using `GIT_CONFIG_GLOBAL=/dev/null`, because the environment global config requires an unavailable commit-signing agent for the temporary test repositories.

## Final verification fix: reject nested public tags

Added a real Git regression test proving that an annotated public release tag which directly targets another tag is rejected even when recursive peeling reaches the generated candidate. Validation now parses the fetched tag object's direct `object` header and requires that SHA to equal the candidate before performing the unchanged exact canonical-message check. RED: the nested tag was accepted and the test did not raise. GREEN: focused existing-public-tag tests passed (`3 passed`); the complete promotion suite passed (`82 passed`) with test-only global Git signing disabled.
