# Testing Guide

This directory contains all tests for the AGILE project.

## Quick Start

```bash
# Before pushing code, run this to ensure CI will pass:
./tests/test_e2e_ci_locally.sh --all
```

This runs both unit tests and E2E tests in the same Docker environment as CI.

## Test Types

### 1. Unit Tests
Located in `agile/rl_env/tests/`
- Test individual MDP components (actions, rewards, terminations, etc.)
- Run automatically in CI on every push
- Quick to execute (~1 minute)

### 2. End-to-End (E2E) Tests
Located in `tests/`
- **Training E2E** (`test_all_tasks_e2e.py`): Test complete training pipelines for all tasks
- **Evaluation E2E** (`test_deterministic_eval_e2e.py`): Test deterministic evaluation pipeline
- Ensure new features don't break existing functionality
- Run on main branch or manually triggered in CI

## Running Tests

### 🚀 Docker Testing (Recommended - Matches CI)

The `test_e2e_ci_locally.sh` script runs tests in Docker, exactly as they run in CI:

```bash
# Run ALL tests (unit + E2E) - recommended before pushing
./tests/test_e2e_ci_locally.sh --all

# Run only E2E tests (default)
./tests/test_e2e_ci_locally.sh

# Run only unit tests
./tests/test_e2e_ci_locally.sh --unit

# Test a specific task
./tests/test_e2e_ci_locally.sh --task Velocity-G1-v0
```

This ensures your code will pass CI before you push!

### Local Testing (Without Docker)

If you have Isaac Lab installed locally:

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

### Adding Unit Tests for New Functions

When you add new MDP components (rewards, terminations, actions, etc.), add corresponding unit tests:

1. **Find the appropriate test file** in `agile/rl_env/tests/`:
   - `test_mdp_actions.py` - for action-related functions
   - `test_mdp_rewards.py` - for reward functions
   - `test_mdp_terminations.py` - for termination conditions
   - `test_mdp_utils.py` - for utility functions

2. **Add a test method** to the appropriate test class:
```python
def test_your_new_function(self):
    """Test description."""
    # Setup
    mock_env = self._create_mock_env()

    # Test your function
    result = your_new_function(mock_env, param1=value1)

    # Assert expected behavior
    self.assertEqual(result.shape, (self.num_envs,))
    self.assertTrue(torch.all(result >= 0))
```

3. **Use mocking** for Isaac Sim dependencies:
```python
with patch("agile.rl_env.mdp.module.some_isaac_function") as mock_func:
    mock_func.return_value = expected_value
    result = your_function(env)
```

### Adding E2E Tests for New Tasks

⚠️ **IMPORTANT**: When you create a new task, add it to the E2E test suite!

1. **Register your task** in `agile/rl_env/tasks/<category>/<robot>/__init__.py`:
```python
gym.register(
    id="YourTask-Robot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={...}
)
```

2. **Add to E2E tests** in `tests/test_all_tasks_e2e.py`:
```python
# Find the section marked with 🆕
# ====================================================================
# 🆕 ADD YOUR NEW TASKS HERE!
# ====================================================================
"YourTask-Robot-v0",  # Brief description of your task
```

3. **Test locally** before pushing:
```bash
# Quick test of your specific task
./tests/test_e2e_ci_locally.sh --task YourTask-Robot-v0
```

## CI Pipeline

The CI pipeline runs in three stages:

1. **Lint** - Code quality checks (always runs)
2. **Unit Tests** - Component testing (always runs)
3. **E2E Tests** - Full training tests (main branch or manual trigger)

### E2E Tests in CI
- Run automatically on `main` branch
- Can be manually triggered on any branch
- Allowed to fail without blocking the pipeline
- 30-minute timeout

## Test Configuration

### For CI Tests
- Headless mode enabled
- WandB disabled
- Small number of environments (4)
- Minimal iterations (2)

### For Local Development
You can run more comprehensive tests locally:
```bash
# Edit test_all_tasks_e2e.py to increase:
num_iterations = 100  # More training iterations
num_envs = 32        # More parallel environments
```

## Troubleshooting

### Unit Tests Failing
1. Check if `ISAACLAB_PATH` is set
2. Ensure dependencies are installed: `./scripts/setup/install_deps_local.sh`
3. Check for import errors in test files

### E2E Tests Failing
1. Verify task is registered in `agile/rl_env/tasks/<category>/<robot>/__init__.py`
2. Check if task config exists
3. Look for CUDA/GPU errors if running locally without GPU

### CI Tests Failing
1. Check if `datasets` dependency is in `install_deps_ci.sh`
2. Verify Docker image is correct: `nvcr.io/nvidia/isaac-lab:2.3.0`
3. Check GitLab runner has GPU access for E2E tests
