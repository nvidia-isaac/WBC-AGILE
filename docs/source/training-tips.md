# Training Tips

Practical insights and rules of thumb from the team's experience developing locomotion and manipulation policies. These are specific to our setup and may not generalize to every application.

## Robot Modeling

Get the model right before any RL -- if the model is wrong, the policy learns the wrong physics and those "skills" won't survive on hardware.

### USD Model Checklist

- Mass and inertia are realistic and consistent; verify COM per link and whole body
- Joint limits match real mechanical limits (position, velocity); include soft limits if applicable
- Geometry is clean and accurate for kinematics/self-collision; avoid interpenetration
- Contact shapes approximate real surfaces; use plausible friction/restitution; prefer simple, watertight collision shapes
- Sensor frames are correct (IMU orientation, camera extrinsics, sign conventions)
- Units are consistent (m, kg, s, N) across the entire model

### Robot Configuration (Beyond USD)

- Actuator limits: set realistic torque/current and velocity limits; clamp in sim exactly as on the robot
- Task-aware limits: lowering joint velocity limits can stabilize early training; relax later as needed
- Friction, damping, armature baselines: choose plausible values; unrealistic damping can hide unstable policies
- Controller settings: match PD gains and effort saturations to your deployment stack

### Validate Before Training

- Gravity settle: robot rests without jitter/drift; COM/inertias are likely correct
- Drop/impulse tests on key links to sanity-check inertials
- Joint limit sweep: drive to limits; verify clamping/signs/units
- Zero-command stand: robot should not "creep" due to bad frames/offsets
- Contact sanity: visualize contact points/normals; check friction behavior
- Log model metadata (masses, inertias, limits) at startup; version-control it

### Sim-to-Real Alignment

- Establish a verified baseline first; randomize around that baseline
- When hardware disagrees, fix USD/config to match reality -- don't compensate with rewards
- Version USD and config together; keep a short change log for physical edits

## General Philosophy

- **Start simple, then iterate.** Start with the simplest environment, reward, and network (e.g., 3-layer MLP). Give the agent all observations it might need (even privileged ones). Get a basic setup working before thinking about deployment.
- **Be robust to seeds.** Never rely on a single lucky seed. Test final hyperparameters across at least 5 different seeds.

## Environment & Reward Design

### Reward Recipe

```{tip}
A good reward recipe combines **Task + Style + Regularization**:
- **Task**: The main goal (e.g., velocity tracking). Start here with the highest weight.
- **Style**: How to do it (e.g., feet parallel, upright torso). Add when you see ugly behavior.
- **Regularization**: What to avoid (e.g., joint limits, torque, action rate). Add once the task is learnable. This is critical for sim-to-real.
```

### Curriculum Design

Exploration is one of the hardest problems in RL. A good curriculum guides the agent from an easy-to-solve task to the final, complex one:

- **Fading guidance**: Start with "helper" aids (external forces, simplified physics, strong reward shaping) and gradually fade them out as the policy improves.
- **Increasing difficulty**: Start with minimal penalties and a simple task (no obstacles, low regularization). Gradually increase difficulty by ramping up style penalties, adding regularization, or introducing more complex environments.

### Terminations

```{warning}
If `episode_length` plots crash to zero, the agent is "suicidal" -- the expected return from continuing (accumulating negative rewards) is worse than terminating and resetting. The immediate fix is to increase the termination penalty (make it more negative), but this requires per-task tuning and can be tricky to get right as the value landscape shifts during training. Value-bootstrapped terminations offer a more principled solution by making termination value-neutral, removing the need for per-task penalty tuning.
```

Don't let the agent waste time. If the agent is in a bad or stuck state it can't recover from (e.g., robot has fallen), terminate the episode. This massively speeds up learning. You can remove these terminations later with a curriculum.

### Observations

- **Start privileged**: Begin training with privileged info (true velocities, contact forces, randomizations) for both actor and critic to find a performance upper bound.
- **History helps**: Use observation history (stacking) to help the agent handle partial observability.
- **Final policy**: The critic can remain privileged, but the actor *must* eventually use only real-world-available sensors (via distillation to a student policy).
- **Clamp and scale**: Some observations can get very large (raycasting can return infinite values). Others have high magnitude and should be scaled down (fixed value or empirical normalization).

### Symmetry

Only use symmetry augmentation if the skill should be symmetric (e.g., walking, general object tracking).

## Training & Monitoring

### Key Plots to Watch

1. **Metrics vs. Reward**: If reward rises but task metrics stagnate, the agent is **reward hacking** -- your reward function is wrong
2. **Value loss**: Should converge below ~1.0. If too high, scale down all reward weights. This can make a bigger difference than one would think
3. **Policy noise std**: Should decrease over training. If it doesn't, tune the entropy coefficient
4. **Curriculum distribution**: Verify the agent reaches the hardest curriculum stage
5. **Videos**: Always record periodic training videos -- plots alone can be misleading

### Entropy Coefficient

The entropy coefficient (`ent_coef`) controls exploration vs. exploitation:

- **Too high**: The policy maximizes its own noise and never learns to exploit
- **Too low**: Premature convergence to a bad local minimum
- **Recommended**: Start at 0.005-0.01, decay to 0.0 over training (e.g., `entropy_annealing_decay_rate=0.9995`)

### Policy Noise Explosion

If policy noise keeps growing instead of converging, the entropy bonus gradient may be dominating the task gradient. The entropy bonus always provides a clear, unambiguous gradient direction: increase noise. Even though the entropy coefficient is small, this gradient is consistent. In contrast, if the task reward signal is noisy or contradictory across the batch, advantage gradients from different data points pull in different directions and largely cancel out. The result: the small but consistent entropy gradient wins, and the policy gets noisier and noisier.

Fixes, in order of preference:

1. **Improve the reward function.** A cleaner reward signal produces more consistent advantage estimates, giving the task gradient a clear direction that outweighs the entropy bonus. This is the best fix.
2. **Reduce the entropy coefficient.** Directly shrinks the competing gradient.
3. **Run a fraction of environments deterministically.** Setting ~20% of environments to run without domain randomization or observation noise guarantees a subset of clean gradient signal in every batch.

### Value Function Stability

Good value function learning is critical -- a bad value function can destabilize the entire training loop. The mechanism: value bootstrapping uses the value function's own predictions to estimate returns. If those predictions are noisy or high-magnitude errors, bootstrapping amplifies and propagates the errors back into the return estimates, making the value function even harder to learn. This creates a vicious cycle.

This is especially problematic with **bootstrapped terminations** (e.g., timeout terminations where V(s') is used instead of 0), because the value estimate at the boundary directly contaminates the return for the entire episode tail.

```{tip}
If you notice training instability, check the value loss. If it's high or diverging, the root cause is often a noisy or poorly shaped reward function. A smoother, more predictable reward landscape makes the value function's job easier, breaking the bootstrapping error cycle.
```

## Sim-to-Real Transfer

Successful sim-to-real transfer rests on two pillars: **robustness** (via domain randomization) and **smoothness** (via action regularization). Getting either one wrong will result in policies that fail on hardware.

### Domain Randomization

Domain randomization trains the policy to be robust to the sim-to-real gap by exposing it to a range of physics parameters during training. The key is to randomize around a **verified baseline** -- not arbitrary ranges.

What to randomize (roughly in order of impact):

- **Friction coefficients**: Ground and foot contact friction. This has outsized impact on locomotion transfer.
- **Mass and COM**: Per-link mass perturbations and center-of-mass offsets.
- **PD gains and motor strength**: Randomize actuator stiffness/damping and effort limits. Randomization within proper armature and PD gain ranges matters a lot.
- **Joint properties**: Default positions, damping, armature.
- **External disturbances**: Random pushes, varying terrain.
- **Sensor noise**: Add noise to observations (IMU, joint encoders) to match real sensor characteristics.
- **Actuator delays**: Randomize action delay to handle variable control loop timing on hardware.

```{tip}
Start with narrow randomization ranges and widen them as the policy becomes more capable. If the policy can't solve the task with wide randomization, the ranges are too aggressive -- the policy needs to learn the basic skill first.
```

### Action Smoothness

```{warning}
A policy that looks smooth in simulation may be relying on high sim friction/damping to mask aggressive, noisy actions. Real hardware will reject such policies. The policy *itself* must output smooth actions.
```

Key regularization terms for smooth hardware behavior:

- **Action rate penalty**: Penalizes the difference between consecutive actions. This is the single most important regularization for sim-to-real. Use L2 (squared difference) rather than L1 for smoother gradients.
- **Action acceleration penalty**: Penalizes the second derivative of actions (change in action rate). Produces even smoother trajectories.
- **Joint velocity/acceleration penalty**: Discourages fast, jerky joint motion.
- **Torque penalty**: Keeps motor commands within comfortable operating ranges.
- **L2C2 regularization**: Penalizes the Lipschitz constant of the policy, producing inherently smoother mappings from observations to actions.

These regularization terms should be introduced gradually via curriculum -- add them once the basic task is learnable, then ramp up their weight.

### Sim-to-Sim Validation

Before going to hardware, deploy your trained policy in a test-only simulation (see {doc}`sim2mujoco`). Check joint-level commands: how aggressive is it? Is it constantly hitting torque or position limits? This is your last chance to catch bad behavior before it breaks hardware.

### Hardware Specifics

- **PD gains**: Lower gains can lead to smoother, more compliant actions. These are a valid way to make the behavior smoother.
- **Parallel joints**: Joints like the ankles on some humanoids can be problematic. They may require extra, specific regularization terms and low PD gains.
- **Timing**: Depending on your deployment stack you might have timing issues such as delays. Make your policy robust to that too by using delayed actuator models.

### The Number One Rule

If the behavior on hardware looks very different from simulation, **your simulation is wrong.** Fix the simulation to match reality. Do *not* try to hack the simulation to make policy learning simpler.
