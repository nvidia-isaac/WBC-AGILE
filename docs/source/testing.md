# Testing

## Quick Start

```bash
# Unit tests
uv run pytest

# E2E tests (requires GPU)
uv run pytest -m e2e
```

## Test Types

### Unit Tests

Located under `agile/**/tests/` and `tests/`. Unit tests are any pytest-discovered tests that are not marked `e2e`. They run automatically in CI on every push.

### End-to-End (E2E) Tests

Located in `tests/` and marked with `@pytest.mark.e2e`:

- **`test_deterministic_eval_e2e.py`**: Deterministic evaluation pipeline
- **`test_sim2mujoco_e2e.py`**: Sim2MuJoCo pipeline
- **`test_task_smoke_e2e.py`**: Task training, evaluation, and play smoke tests

Ensure new features do not break existing functionality. Run on the main branch or manually trigger them in CI.

## Running Tests

### Local Testing

Requires the locked uv environment:

```bash
uv run pytest
uv run pytest -m e2e
uv run pytest -m e2e tests/test_task_smoke_e2e.py::test_task_training_smoke
```

### Release workflow tests

Run the focused release suite after changing release code, CI validation, or the local release runbook:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --frozen pytest -q \
  tests/test_release_wrapper.py \
  tests/test_release_auth.py \
  tests/test_public_release_promotion.py \
  tests/test_copybara_staging_policy_assets.py \
  tests/test_public_release_boundary.py
```

The documented production path is a confirmed local `release.py` invocation. Automatic `deploy_validation` still exports `main` and protected `release-X.Y` branches to staging and retains `validation-release.json` as diagnostic evidence. Public promotion never runs automatically. Authenticate local GitLab and GitHub clients before promotion, validate the matching staging branch, and then run `release.py` with `--source-ref`, `--target-repo`, and `--release-version`. The command previews its immutable plan, requires confirmation unless `--yes` is supplied, and writes `public-release.json` after success. Use `--update-target-main` only for the newest stable line. Retry a partial promotion with identical inputs; matching existing state is validated and reconciled safely.

## Adding Tests

### Unit Tests for New MDP Components

When you add new MDP components (rewards, terminations, actions, and similar behavior), add corresponding unit tests:

1. Find the appropriate test file in `agile/rl_env/tests/`:
   - `test_mdp_actions.py` -- for action-related functions
   - `test_mdp_rewards.py` -- for reward functions
   - `test_mdp_terminations.py` -- for termination conditions
   - `test_mdp_utils.py` -- for utility functions
2. Add a test method to the appropriate test class:

   ```python
   def test_your_new_function(self):
       mock_env = self._create_mock_env()
       result = your_function(mock_env, param1=value1)
       self.assertEqual(result.shape, (self.num_envs,))
       self.assertTrue(torch.all(result >= 0))
   ```

3. Use mocking for Isaac Sim dependencies:

   ```python
   with patch("agile.rl_env.mdp.module.some_isaac_function") as mock_func:
       mock_func.return_value = expected_value
       result = your_function(env)
   ```

### E2E Tests for New Tasks

When you create a new task, make sure it is registered in `agile/rl_env/tasks/**/__init__.py` and covered by the task smoke test policy in `tests/test_task_smoke_e2e.py`.

1. Register your task in `agile/rl_env/tasks/<category>/<robot>/__init__.py`.
2. Test locally before pushing:

   ```bash
   uv run pytest -m e2e tests/test_task_smoke_e2e.py
   ```

