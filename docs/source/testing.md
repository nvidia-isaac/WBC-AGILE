# Testing

## Quick Start

```bash
# Before pushing code, run this to ensure CI will pass:
./tests/test_e2e_ci_locally.sh --all
```

This runs both unit tests and E2E tests in the same Docker environment as CI.

## Test Types

### Unit Tests

Located in `agile/rl_env/tests/`. Test individual MDP components (actions, rewards, terminations, etc.). Quick to execute (~1 minute). Run automatically in CI on every push.

### End-to-End (E2E) Tests

Located in `tests/`:

- **`test_all_tasks_e2e.py`**: Complete training pipelines for all registered tasks
- **`test_deterministic_eval_e2e.py`**: Deterministic evaluation pipeline

Ensure new features don't break existing functionality. Run on main branch or manually triggered in CI.

## Running Tests

### Docker Testing (Recommended)

Matches the CI environment exactly:

```bash
./tests/test_e2e_ci_locally.sh --all                # All tests (unit + E2E)
./tests/test_e2e_ci_locally.sh                       # E2E tests only (default)
./tests/test_e2e_ci_locally.sh --unit                # Unit tests only
./tests/test_e2e_ci_locally.sh --task Velocity-G1-v0 # Specific task
```

### Local Testing

Requires Isaac Lab installed locally:

```bash
# Unit tests
./tests/run_unit_tests.sh
./tests/run_unit_tests.sh -v  # verbose output

# E2E tests (requires GPU)
${ISAACLAB_PATH}/isaaclab.sh -p tests/test_all_tasks_e2e.py

# Deterministic evaluation E2E test
${ISAACLAB_PATH}/isaaclab.sh -p tests/test_deterministic_eval_e2e.py
```

## Adding Tests

### Unit Tests for New MDP Components

When you add new MDP components (rewards, terminations, actions, etc.), add corresponding unit tests:

1. Find the appropriate test file in `agile/rl_env/tests/`:
   - `test_mdp_actions.py` -- for action-related functions
   - `test_mdp_rewards.py` -- for reward functions
   - `test_mdp_terminations.py` -- for termination conditions
   - `test_mdp_utils.py` -- for utility functions

2. Add a test method to the appropriate test class:

```python
def test_your_new_function(self):
    """Test description."""
    mock_env = self._create_mock_env()
    result = your_new_function(mock_env, param1=value1)
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

When you create a new task, add it to the E2E test suite:

1. Register your task in `agile/rl_env/tasks/<category>/<robot>/__init__.py`

2. Add to `tests/test_all_tasks_e2e.py` in the marked section:

```python
# ====================================================================
# ADD YOUR NEW TASKS HERE!
# ====================================================================
"YourTask-Robot-v0",  # Brief description of your task
```

3. Test locally before pushing:

```bash
./tests/test_e2e_ci_locally.sh --task YourTask-Robot-v0
```

## CI/CD Pipeline

The CI pipeline runs the following stages:

| Stage | Description | GPU | Runs on |
|---|---|---|---|
| **lint** | `pre-commit run --all-files` | No | Every push |
| **unit_test** | Unit tests with mocked CUDA | No | Every push |
| **e2e_test** | Full training pipeline tests | Yes | Main branch or manual |

E2E tests are allowed to fail without blocking the pipeline (30-minute timeout).

### CI Test Configuration

- Headless mode enabled
- W&B disabled
- Small number of environments (4)
- Minimal iterations (2)

For local development, you can run more comprehensive tests by editing `test_all_tasks_e2e.py` to increase `num_iterations` and `num_envs`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Unit tests failing | Check `ISAACLAB_PATH` is set, re-run `install_deps_local.sh` |
| Import errors in tests | Check for missing dependencies |
| E2E tests failing | Verify task is registered in `__init__.py`, check task config exists |
| CUDA/GPU errors locally | Ensure GPU access and CUDA drivers are installed |
| CI tests failing | Check `datasets` dependency in `install_deps_ci.sh`, verify Docker image `nvcr.io/nvidia/isaac-lab:2.3.2` |
| CI runner errors | Check runner has GPU access for E2E tests |
