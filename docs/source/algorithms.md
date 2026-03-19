# Algorithms

AGILE uses a custom fork of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (BSD 3-Clause) for PPO and distillation training. See the base library for standard PPO, distillation, and recurrent policy documentation. This page covers AGILE-specific enhancements.

The custom fork lives in `agile/algorithms/rsl_rl/`.

## PPO Enhancements

### TensorDict Observations

Both actor and critic networks accept `TensorDict` inputs, enabling structured observation spaces where different sensor modalities are kept separate rather than concatenated into a flat vector.

### Symmetry Augmentation and Mirror Loss

Left-right symmetry data augmentation during rollout collection and optional mirror loss that penalizes asymmetric actions for mirrored observations. Also applicable during distillation (non-recurrent students only).

```python
algorithm = RslRlPpoAlgorithmCfg(
    ...,
    symmetry_cfg=RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=lr_mirror_T1,
    ),
)
```

### L2C2 Regularization

Lipschitz-Constrained Continuity regularization. Interpolates between consecutive observations `obs_t` and `obs_{t+1}` with a random factor, then penalizes large output changes via MSE. Applied to both actor and critic. Terminal states are excluded.

```python
algorithm = RslRlPpoAlgorithmCfg(
    ...,
    l2c2_cfg=RslRlL2C2Cfg(
        lambda_actor=1.0,
        lambda_critic=0.1,
    ),
)
```

### Reward Normalization

`ReturnVarianceNormalization` normalizes rewards to achieve approximately unit-variance returns using EMA statistics that adapt to curriculum changes.

```
scale = sigma * gamma_factor * return_correction + eps
normalized_reward = reward / scale
```

where `gamma_factor = 1 / sqrt(1 - gamma^2)`.

**Return-scale correction**: The i.i.d. formula underestimates return variance when rewards are temporally correlated (e.g., a fallen robot produces many consecutive low rewards). A correction factor is EMA-tracked from measured GAE return standard deviations once per rollout, using the product invariance `return_std * correction = K` for oscillation-free convergence.

**Outlier clipping**: Rewards beyond `outlier_threshold` standard deviations from the running mean are clipped before updating statistics and in the normalized output.

```python
algorithm = RslRlPpoAlgorithmCfg(
    ...,
    reward_normalization_cfg=RslRlRewardNormalizationCfg(
        decay=0.999,           # EMA decay (~693 steps half-life)
        epsilon=1e-2,          # Numerical stability
        return_scale_decay=0.999,  # Return-scale correction (None to disable)
        outlier_threshold=10.0,    # Clip beyond N std deviations (None to disable)
    ),
)
```

### Termination Handling

AGILE extends standard timeout bootstrapping with a configurable good/bad termination system that provides reward shaping at episode boundaries.

**Timeout bootstrapping** (standard): When an episode ends due to a time limit, the value estimate is bootstrapped to avoid biasing the value function:

```python
rewards += gamma * values * time_outs
```

**Good/bad termination handling** (`agile/rl_env/termination_cfg.py`): Each termination term can be annotated with a type and sigma value via `DoneTermCfg`:

```python
from agile.rl_env.termination_cfg import DoneTermCfg as DoneTermEx

# "bad" termination: bootstrap + subtract sigma (worse than continuing)
no_height_progress = DoneTermEx(func=mdp.no_height_progress, termination_type="bad", sigma=1.0, ...)

# "good" termination: bootstrap + add sigma (better than continuing)
reached_goal = DoneTermEx(func=mdp.reached_goal, termination_type="good", sigma=2.0, ...)
```

For both good and bad terminations, the value is first bootstrapped (making the termination value-neutral like timeouts), then sigma is added or subtracted. Since sigma operates post-normalization, it is largely scale-invariant and requires less tuning across different reward scales. Environments already bootstrapped by the timeout handler are excluded.

The `VecEnvWrapper` scans `DoneTermCfg` metadata at initialization and aggregates per-environment sigmas (max across fired terms) into `bad_termination_sigma` / `good_termination_sigma` tensors passed to PPO via the `infos` dict.

Unlike the other enhancements on this page, termination handling is configured in the **task env config** (in the `TerminationsCfg` class), not in the PPO algorithm config.

### Entropy Coefficient Annealing

Automatic entropy coefficient decay gated on training progress and episode quality:

1. Annealing begins when `progress > entropy_coef_annealing_start_progress`
2. AND `mean_episode_length > success_episode_length_threshold`
3. Decay is either linear (default) or exponential (if `entropy_annealing_decay_rate` is set)
4. Coefficient is clamped to `min_entropy_coef`

```python
runner_cfg = RslRlOnPolicyRunnerCfg(
    ...,
    enable_entropy_coef_annealing=True,
    entropy_annealing_decay_rate=0.9995,
    min_entropy_coef=0.001,
)
```

## Distillation Enhancements

AGILE extends RSL-RL's DAgger-style distillation with the following modules:

### Symmetry in Distillation

Mirror loss can be applied during distillation for non-recurrent student policies. The student is penalized for producing asymmetric actions given left-right mirrored observations. Automatically disabled for recurrent policies.

### StudentTrainedTeacher

A distillation variant (`student_trained_teacher.py`) where the teacher is loaded from a pre-exported TorchScript model rather than from a checkpoint's state dict. This enables distilling from any exported policy without needing to match the teacher's network architecture in code.

### StudentFinetuneRecurrent

After distillation, the student can be fine-tuned with RL (`student_finetune_recurrent.py`). The distilled student is paired with a fresh critic and trained with PPO, allowing the student to improve beyond the teacher's performance while retaining the distilled behavior as initialization.

## Configuration Examples

### PPO with Enhancements

```python
@configclass
class T1VelocityPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    empirical_normalization = False
    enable_entropy_coef_annealing = True
    entropy_annealing_decay_rate = 0.9995
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_T1,
        ),
        reward_normalization_cfg=RslRlRewardNormalizationCfg(
            decay=0.999,
            epsilon=1e-2,
        ),
        l2c2_cfg=RslRlL2C2Cfg(
            lambda_actor=1.0,
            lambda_critic=0.1,
        ),
    )
```

### Distillation

```python
@configclass
class G1DistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    max_iterations = 5_000
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss_type="mse",
    )
    policy = RslRlStudentTrainedTeacherCfg(
        class_name="StudentTrainedTeacherRecurrent",
        teacher_path="agile/data/policy/.../teacher.pt",
        student_hidden_dims=[256, 256, 128],
        activation="elu",
    )
```

## Evaluation Framework

See {doc}`evaluation` for the evaluation pipeline, scenario configs, report generation, and Sim-to-MuJoCo transfer.
