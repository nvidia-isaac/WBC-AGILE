# Algorithms

AGILE uses public [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
`rsl-rl-lib==5.4.1` as its reinforcement-learning backend. The package is
installed from PyPI and then patched with the small AGILE delta in
`third_party/rsl_rl/patches/rsl_rl_5_4_1_agile.patch`.

The patch keeps AGILE-specific behavior that is not yet available in public
RSL-RL:

- TorchScript teacher loading for distillation through `JitTeacherModel`
- optional distillation symmetry loss and optimizer weight decay
- PPO entropy coefficient annealing
- PPO L2C2 regularization
- return-variance reward normalization
- AGILE termination sigma handling for good and bad terminal states

The patch is applied by the setup scripts and verified by
`scripts/verify_rsl_rl.py`. AGILE does not carry a copied RSL-RL source tree.

## Integration

AGILE task configs live in `agile/rl_env/rsl_rl/`. The helper
`rsl_rl_cfg_to_dict()` converts AGILE configclasses into the native RSL-RL 5.x
runner schema, and `make_rsl_rl_runner()` instantiates the configured runner.

Observation groups are represented as TensorDict entries. `RslRlVecEnvWrapper`
adapts Isaac Lab observations into the RSL-RL 5.x `VecEnv` interface and maps
AGILE's grouped observations to the actor, critic, student, and teacher inputs
expected by RSL-RL.

## PPO

PPO uses the public RSL-RL 5.x actor, critic, storage, logging, and runner
interfaces. AGILE's patch preserves the remaining PPO extensions listed above
without replacing the full upstream package.

Checkpoint keys follow RSL-RL 5.x:

- PPO: `actor_state_dict`, `critic_state_dict`, `optimizer_state_dict`
- Distillation: `student_state_dict`, `teacher_state_dict`,
  `optimizer_state_dict`
- Optional: `rnd_state_dict`, `rnd_optimizer_state_dict`,
  `reward_normalizer_state_dict`
- Common: `iter`, `infos`

## Distillation

Distillation uses a student model built from task observations and a teacher
model loaded from an exported TorchScript policy. For AGILE policies, the
teacher config points at `JitTeacherModel`, which exposes the RSL-RL teacher
interface while delegating inference to the exported policy.

See {doc}`training` for training commands and {doc}`evaluation` for evaluation
workflows.
