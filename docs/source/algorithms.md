# Algorithms

This page documents AGILE's RL algorithms, policy architectures, and evaluation framework. AGILE uses a **custom fork of RSL-RL** enhanced with TensorDict support, additional regularization methods, and multi-GPU training capabilities.

## Custom RSL-RL

AGILE's algorithm stack lives in `agile/algorithms/rsl_rl/` and is based on the [RSL-RL](https://github.com/leggedrobotics/rsl_rl) library (BSD 3-Clause). Key differences from vanilla RSL-RL:

- **TensorDict observations**: Both actor and critic networks accept `TensorDict` inputs, enabling structured observation spaces where different sensor modalities are kept separate rather than concatenated into a flat vector.
- **Symmetry augmentation and mirror loss**: Built-in support for left-right symmetry data augmentation and mirror loss regularization during PPO and distillation training.
- **Random Network Distillation (RND)**: Intrinsic curiosity rewards for exploration via a predictor-target network pair.
- **L2C2 regularization**: Lipschitz-Constrained Continuity regularization that penalizes large output changes for small input changes, encouraging smoother policies.
- **Reward normalization**: EMA-based return variance normalization that adapts to curriculum changes during training.
- **Critic warmup**: Option to train only the value function for an initial phase before introducing the policy gradient loss.
- **Multi-GPU training**: Gradient synchronization across GPUs via `torch.distributed` with NCCL backend.
- **Student-teacher distillation**: First-class support for distilling privileged teacher policies into deployable student policies, including recurrent variants.

## PPO Algorithm

The PPO implementation (`agile/algorithms/rsl_rl/rsl_rl/algorithms/ppo.py`) follows the standard Proximal Policy Optimization algorithm with several extensions.

### Core PPO Loop

Each training iteration consists of:

1. **Rollout collection**: The policy collects `num_steps_per_env` transitions from all parallel environments using `act()` and `process_env_step()`.
2. **Return computation**: GAE (Generalized Advantage Estimation) computes advantages and returns via `compute_returns()`.
3. **Policy update**: Mini-batch SGD over `num_learning_epochs` epochs with `num_mini_batches` per epoch.

### Loss Components

The total loss combines several terms:

```
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
     + [mirror_loss_coeff * symmetry_loss]     # if symmetry enabled
     + [lambda_actor * l2c2_actor_loss]         # if L2C2 enabled
     + [lambda_critic * l2c2_critic_loss]       # if L2C2 enabled
```

- **Surrogate loss**: Clipped PPO objective with configurable `clip_param` (default 0.2)
- **Value loss**: Optionally clipped MSE between predicted and target values
- **Entropy bonus**: Encourages exploration; can be annealed during training
- **RND loss**: Trained separately with its own optimizer when RND is enabled

### Adaptive Learning Rate

When `schedule="adaptive"`, the learning rate adjusts based on the KL divergence between old and new policies:

- If KL > 2 * `desired_kl`: learning rate is divided by 1.5
- If KL < `desired_kl` / 2: learning rate is multiplied by 1.5
- Learning rate is clamped to [1e-5, 1e-2]

### Key Hyperparameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `num_learning_epochs` | SGD epochs per rollout | 5 |
| `num_mini_batches` | Mini-batches per epoch | 4 |
| `clip_param` | PPO clipping parameter | 0.2 |
| `gamma` | Discount factor | 0.99 |
| `lam` | GAE lambda | 0.95 |
| `learning_rate` | Adam learning rate | 1e-3 |
| `entropy_coef` | Entropy bonus coefficient | 0.005-0.01 |
| `desired_kl` | Target KL for adaptive LR | 0.01 |
| `max_grad_norm` | Gradient clipping norm | 1.0 |
| `value_loss_coef` | Value loss weight | 1.0 |

### Timeout Bootstrapping

When an episode ends due to a time limit (not a true terminal state), the value estimate is bootstrapped into the reward to avoid biasing the value function:

```python
rewards += gamma * values * time_outs
```

## Distillation Algorithm

The distillation algorithm (`agile/algorithms/rsl_rl/rsl_rl/algorithms/distillation.py`) trains a student policy to mimic a pre-trained teacher policy through behavior cloning.

### How It Works

1. At each step, both student and teacher produce actions from their respective observations
2. The student sees limited observations (e.g., proprioception only); the teacher sees privileged observations (e.g., terrain height maps, true velocities)
3. The loss is the MSE (or Huber) between student and teacher actions
4. For recurrent students, gradients flow through `gradient_length` time steps via BPTT

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `num_learning_epochs` | Epochs per rollout | 5 |
| `gradient_length` | BPTT truncation length | 15 |
| `learning_rate` | Adam learning rate | 1e-3 |
| `loss_type` | Loss function (`mse` or `huber`) | `mse` |
| `weight_decay` | L2 regularization | 0.0 (or 1e-4) |

### Symmetry in Distillation

Mirror loss can be applied during distillation for non-recurrent student policies. When enabled, the student is penalized for producing asymmetric actions given left-right mirrored observations. This is automatically disabled for recurrent policies.

## Policy Architectures

All policy modules are in `agile/algorithms/rsl_rl/rsl_rl/modules/`.

### ActorCritic (MLP)

The base feedforward policy (`actor_critic.py`) with separate actor and critic MLP networks.

**Architecture:**
- Actor: `SimpleMLP(obs_dim -> hidden_dims -> action_dim)`
- Critic: `SimpleMLP(critic_obs_dim -> hidden_dims -> 1)`
- Action distribution: Gaussian with configurable noise type

**Noise standard deviation types:**
- `scalar`: Single learnable parameter shared across actions (default)
- `log`: Log-space learnable parameter per action dimension
- `pred`: Network predicts both mean and log-std (doubles actor output dimension)

**Example configuration:**
```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[256, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)
```

### ActorCriticRecurrent (RNN)

Extends `ActorCritic` with GRU or LSTM memory layers (`actor_critic_recurrent.py`).

**Architecture:**
- Actor: `Memory(obs_dim -> rnn_hidden_dim)` followed by `MLP(rnn_hidden_dim -> action_dim)`
- Critic: `Memory(critic_obs_dim -> rnn_hidden_dim)` followed by `MLP(rnn_hidden_dim -> 1)`

Separate RNN memories are maintained for actor and critic. Hidden states are reset when episodes terminate.

### StudentTeacher (MLP)

For distillation training (`student_teacher.py`). The teacher network is frozen and produces target actions from privileged observations.

**Architecture:**
- Student: `MLP(student_obs_dim -> hidden_dims -> action_dim)` (trainable)
- Teacher: `MLP(teacher_obs_dim -> hidden_dims -> action_dim)` (frozen)

The `load_state_dict` method intelligently handles loading:
- From RL training checkpoint: maps `actor.*` weights to teacher network
- From distillation checkpoint: loads both student and teacher

Supports optional dropout regularization to prevent overfitting during distillation.

### StudentTeacherRecurrent (RNN)

Recurrent variant (`student_teacher_recurrent.py`) that adds LSTM/GRU memory to the student network.

**Architecture:**
- Student: `Memory(student_obs_dim -> rnn_hidden_dim)` followed by `MLP(rnn_hidden_dim -> action_dim)`
- Teacher: optionally recurrent if `teacher_recurrent=True`

Hidden states are properly managed: detached at gradient boundaries and reset on episode termination.

### StudentTrainedTeacher

A variant (`student_trained_teacher.py`) where the teacher is loaded from a pre-exported TorchScript model rather than from a checkpoint's state dict. This enables distilling from any exported policy without needing to match the teacher's network architecture in code.

## Observation Normalization

Two normalization approaches are available (`agile/algorithms/rsl_rl/rsl_rl/modules/normalizer.py`):

### EmpiricalNormalization

Running mean/variance normalization applied to observations:

```
normalized = (x - running_mean) / (running_std + eps)
```

Statistics are updated during training and frozen during evaluation. Controlled by `empirical_normalization` in the runner config.

### ReturnVarianceNormalization

Normalizes rewards to achieve approximately unit-variance returns. Uses EMA (exponential moving average) for variance tracking, which adapts to curriculum changes better than cumulative statistics.

For returns `G = sum(gamma^t * r_t)`:
```
normalized_reward = reward / (sigma * gamma_factor + eps)
```
where `gamma_factor = 1 / sqrt(1 - gamma^2)`.

## Rollout Storage

The `RolloutStorage` class (`agile/algorithms/rsl_rl/rsl_rl/storage/rollout_storage.py`) manages trajectory data during training. It supports two training modes:

- **RL mode**: Stores observations, actions, rewards, values, log-probs, advantages, and returns for PPO training. Provides both `mini_batch_generator` (feedforward) and `recurrent_mini_batch_generator` (RNN) iterators.
- **Distillation mode**: Stores observations and privileged teacher actions. Provides a sequential `generator()` for BPTT.

Both modes support optional RND states and L2C2 consecutive-observation pairs.

## On-Policy Runner

The `OnPolicyRunner` (`agile/algorithms/rsl_rl/rsl_rl/runners/on_policy_runner.py`) orchestrates the full training loop.

### Training Loop

```
for iteration in range(max_iterations):
    1. Anneal entropy coefficient (if enabled)
    2. Collect rollouts (num_steps_per_env steps per environment)
    3. Compute returns (RL only)
    4. Update policy (PPO or Distillation)
    5. Log metrics to W&B/TensorBoard/Neptune
    6. Save checkpoint every save_interval iterations
```

### Logging

The runner logs the following metrics:

- **Train/**: mean reward, mean episode length, entropy coefficient
- **Loss/**: surrogate, value function, entropy, learning rate, plus optional RND/symmetry/L2C2 losses
- **Policy/**: mean action noise std
- **Perf/**: FPS, collection time, learning time
- **Episode/**: per-episode info from the environment (reward terms, success metrics)
- **Rewards_Raw/**, **Rewards_Weighted/**: per-term reward statistics (if recorder manager is active)

### Entropy Coefficient Annealing

The runner supports automatic entropy coefficient decay:

1. Entropy annealing begins when `progress > entropy_coef_annealing_start_progress`
2. AND `mean_episode_length > success_episode_length_threshold`
3. Decay is either linear (default) or exponential (if `entropy_annealing_decay_rate` is set)
4. Coefficient is clamped to `min_entropy_coef` to prevent training instability

### Checkpoint Format

Saved checkpoints (`model_{iteration}.pt`) contain:

- `model_state_dict`: Policy network parameters
- `optimizer_state_dict`: Adam optimizer state
- `iter`: Current training iteration
- `obs_norm_state_dict` / `privileged_obs_norm_state_dict`: Normalization statistics (if enabled)
- `rnd_state_dict` / `rnd_optimizer_state_dict`: RND state (if enabled)
- `reward_norm_state_dict`: Reward normalizer state (if enabled)

## Evaluation Framework

AGILE includes a comprehensive evaluation framework in `agile/algorithms/evaluation/`. See {doc}`evaluation` for usage instructions, scenario configs, report generation, and framework internals.

## Configuration Classes

RL configuration is defined through dataclass-based configs in `agile/rl_env/rsl_rl/rl_cfg.py`.

### RslRlOnPolicyRunnerCfg

Top-level runner configuration:

```python
@configclass
class RslRlOnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int     # Steps per env per rollout
    max_iterations: int         # Total training iterations
    empirical_normalization: bool  # Enable observation normalization
    policy: RslRlPpoActorCriticCfg  # Policy architecture
    algorithm: RslRlPpoAlgorithmCfg  # Algorithm hyperparameters
    save_interval: int          # Checkpoint frequency
    logger: str = "tensorboard"  # "tensorboard", "wandb", or "neptune"
    # Entropy annealing
    enable_entropy_coef_annealing: bool = False
    entropy_annealing_decay_rate: float | None = None
    min_entropy_coef: float = 0.001
```

### RslRlPpoAlgorithmCfg

PPO hyperparameters with optional extensions:

```python
@configclass
class RslRlPpoAlgorithmCfg:
    class_name: str = "PPO"
    num_learning_epochs: int
    num_mini_batches: int
    learning_rate: float
    schedule: str              # "fixed" or "adaptive"
    gamma: float
    lam: float
    entropy_coef: float
    clip_param: float
    # Optional extensions
    symmetry_cfg: RslRlSymmetryCfg | None = None
    rnd_cfg: RslRlRndCfg | None = None
    l2c2_cfg: RslRlL2C2Cfg | None = None
    reward_normalization_cfg: RslRlRewardNormalizationCfg | None = None
```

### Example: Full PPO Config

```python
@configclass
class T1VelocityPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "velocity_t1_lower"
    wandb_project = "Velocity-T1-Lower"
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
    )
```

### Example: Distillation Config

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
