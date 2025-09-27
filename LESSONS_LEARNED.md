# Robotic RL: Lessons Learned

This document is a living collection of practical lessons and rules of thumb our team has learned while developing the policies in this repository. It's not a scientific guide, but rather our own set of experiences.

Please take these notes with a grain of salt, as they are specific to our setup and may not generalize to very different applications. We'll update this as we learn!

---


### 1. Robot Modeling First: Get It Right Before Any RL

* **Why it matters:** If the model is wrong, the policy learns the wrong physics—and those “skills” won’t survive on hardware.
* **USD model: non‑negotiables**
    * Mass and inertia are realistic and consistent; verify COM per link and whole body.
    * Joint limits match real mechanical limits (position, velocity); include soft limits if applicable.
    * Geometry is clean and accurate for kinematics/self-collision; avoid interpenetration.
    * Contact shapes approximate real surfaces; use plausible friction/restitution; prefer simple, watertight collision shapes.
    * Sensor frames are correct (IMU orientation, camera extrinsics, sign conventions).
    * Units are consistent (m, kg, s, N) across the entire model.
* **Robot configuration (beyond USD)**
    * Actuator limits: set realistic torque/current and velocity limits; clamp in sim exactly as on the robot.
    * Task-aware limits: lowering joint velocity limits can stabilize early training; relax later as needed.
    * Friction, damping, armature baselines: choose plausible values; unrealistic damping can hide unstable policies.
    * Controller settings: match PD gains and effort saturations to your deployment stack.
* **Validate before training**
    * Gravity settle: rests without jitter/drift; COM/inertias are likely correct.
    * Drop/impulse tests on key links to sanity‑check inertials.
    * Joint limit sweep: drive to limits; verify clamping/signs/units.
    * Zero-command stand: robot should not “creep” due to bad frames/offsets.
    * Contact sanity: visualize contact points/normals; check friction behavior.
    * Log model metadata (masses, inertias, limits) at startup; version-control it.
* **Sim‑to‑Real alignment**
    * Establish a verified baseline first; randomize around that baseline.
    * When hardware disagrees, fix USD/config to match reality—don’t compensate with rewards.
    * Version USD and config together; keep a short change log for physical edits.

### 2. General Philosophy & Setup

* **Start Simple, Then Iterate:** Don't over-engineer your first-pass.
    * Start with the simplest possible environment and reward.
    * Start with simple networks (e.g., a 3-layer MLP); this is easier/faster to train.
    * Give the agent all observations it might need (even privileged ones). You can prune them later.
    * Get a basic setup working before you even think about real-world deployment.
* **Be Robust to Seeds:** You should not rely on a few lucky seeds. Always test your final hyperparameters on ~5 different seeds.

---

### 3. Environment & Reward Design

* **Reward Recipe:** A good starting point is a combination of `Task + Style + Regularization`.
    * **Task:** The main goal (e.g., `velocity_tracking`). Start with this; it should have the highest weight.
    * **Style:** How to do it (e.g., `feet_parallel`, `torso_orientation`). Add this if you see ugly behavior.
    * **Regularization:** What to avoid (e.g., `joint_limit_penalty`, `torque_penalty`, `action_rate`). Add this once the task can be learned. This is crucial for sim‑to‑real transfer.
* **Curricula:** Exploration is one of the hardest problems in RL. A good curriculum guides the agent from an easy-to-solve task to the final, complex one. We typically see two types:
    * **Fading Guidance:** Start with "helper" aids (like external forces, simplified physics, or strong reward shaping) and gradually fade them out as the policy improves.
    * **Increasing Difficulty:** Start with minimal penalties and a simple task (e.g., no obstacles, low regularization). Gradually increase difficulty by ramping up style penalties, adding regularization, or introducing more complex environmental elements.
* **Terminations are Critical:** Don't let the agent waste time.
    * If the agent is in a bad or "stuck" state it can't recover from (e.g., robot has fallen), terminate the episode. This massively speeds up learning. You can remove these terminations later on with a curriculum.
    * If `episode_length` plots are crashing to zero, your agent is "suicidal." This means your termination penalties are too high (not negative enough) or your positive rewards are too low.
* **Symmetry:** Only use symmetry augmentation if the skill should be symmetric (e.g., walking, general object tracking).
* **Observations:**
    * **Start Privileged:** Begin training with privileged info (e.g., true velocities, contact forces, randomizations) for both actor and critic to find a performance upper bound.
    * **History Helps:** Use observation history (stacking) to help the agent handle partial observability.
    * **Final Policy:** The critic can and should remain privileged, but the actor *must* eventually use only real-world-available sensors (e.g., via distillation to a student policy).
    * **Clamp and Scale Observations:** Some observation terms should be clamped as they can get very large (raycasting can return infinite values). Other observations have high magnitude and should be scaled down (either with a fixed value or via empirical normalization).
---

### 4. Training & Monitoring

* **Watch Videos!** Plots can be misleading. Always record videos of your policy at different stages. This is the fastest way to *qualitatively* check what behavior the agent is *actually* learning.
* **PPO's Entropy Coefficient (`ent_coef`):** This is a critical hyperparameter.
    * It controls the exploration/exploitation tradeoff.
    * **Too high:** The policy maximizes its own noise. It's just "vibing" and never learns to exploit.
    * **Too low:** The policy converges too quickly to a (likely bad) local minimum and never explores.
    * **Suggestion:** Start with the default (e.g., `0.005`) and decay it to `0.0` over the course of training.
* **Key Plots to Watch:**
    * **`Metrics vs. Reward`:** If your `reward` is going up but your core `task_metric` (e.g., `success_rate`) is not, the agent is **reward hacking**. Your reward function is wrong.
    * **`Losses`:** `value_loss` should converge to a small-ish value (e.g., <1.0). If it's higher, simply scale down all reward weights. This can make a bigger difference than one would think.
    * **`Curriculum`:** Check the curriculum distribution. Is the agent actually reaching the final, hardest stage, or is it stuck somewhere? If so, your curriculum may be too hard.
    * **Policy noise:** The policy's action distribution should converge to a small variance; if not, tune the entropy coefficient.

---

### 5. Sim2Real & Deployment

* **The "Fake Smoothness" Trap:**
    * Your policy might look smooth in simulation, but this can be because high **friction/damping** in the sim is "eating" a very noisy, aggressive policy.
    * Real hardware *hates* noisy, aggressive actions.
    * **Solution:** Use **policy-level regularization** (e.g., penalizing action rate, velocity, or acceleration, l2c2 regularization) to make the policy *itself* smooth.
* **Sim-to-Sim Deploy First:** Before going to hardware, deploy your trained policy in a test-only simulation.
    * **Check joint-level commands.** How aggressive is it? Is it constantly hitting torque or position limits? This is your last chance to catch bad behavior before it breaks hardware.
* **Hardware specifics matter:**
    * **PD Gains:** These are a valid way to make the behavior smoother. Lower gains can lead to smoother, more compliant actions.
    * **Parallel Joints:** Joints like the ankles on some humanoids can be problematic. They may require extra, specific regularization terms and low PD gains.
    * **Timing:** Depending on your deployment stack you might have timing issues such as delays. Make your policy robust to that too by using delayed actuator models.
* **The #1 Rule of Sim‑to‑Real:**
    * If the behavior on hardware looks very different from the one in simulation, **your simulation is wrong.**
    * **You must fix the simulation to match reality.** Do *not* try to hack the simulation to make policy learning simpler.


---
### A Final Word: It's a Bit Magic 🧙
Honestly, RL can feel like a dark art. There is no single, general recipe that works every time. We've found that a single hyperparameter, a small change in the reward, or a minor physics tweak can be the difference between a deployable policy and a total failure.

When you're stuck (and you will be), go talk to another "RL wizard." Discussing your problem is often the fastest way to find a solution.
