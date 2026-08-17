# AGILE Robotics Office Hours (2026) — FAQ

## Quantitative Evaluation and Sim-to-Real

**Q: How do you define success beyond reward curves?**

**A:** AGILE uses both qualitative and quantitative signals.
- **Qualitative:** how natural the motion looks (video review).
- **Quantitative:** tracking errors (velocity/height), episode survival time, robustness under repeated disturbances (pushes), and checks for joint-limit violations. Policies that violate joint limits tend to transfer poorly across simulators and to real hardware. AGILE also validates sim-to-sim transfer by switching to MuJoCo after Isaac Lab checks.

**Q: What metrics predict sim-to-real transfer best?**

**A:** Tracking accuracy alone is not sufficient. The team highlighted:
- survival time under disturbances,
- joint limit compliance (position/velocity),
- sim-to-sim validation (including MuJoCo),
- plus qualitative review to catch unnatural gaits.

---

## Physics, Contact, and Realism

**Q: How do you ensure "real-world physics" (friction/contact) in simulation?**

**A:** The team described a practical approach:
- **Robot parameters:** full system identification has not been completed yet. They start from open-source system ID parameters and adjust using a loop (train → deploy → tune, e.g., PD gains).
- **Object parameters:** mass/inertia and friction are often rough estimates.
- **Robustness strategy:** randomize parameters (e.g., friction coefficients) so the policy is less sensitive to modeling inaccuracies.

**Q: How do you handle complex contact dynamics for sim-to-real?**

**A:** In addition to the underlying physics solver and USD collision setup, they:
- randomize friction/contact-related parameters,
- use contact modeling tricks (e.g., more point-like / "severe" foot contact behavior rather than idealized flat cuboid contact) to better match real-world contact.

---

## Training Workflow and Fine-Tuning

**Q: What does "fine-tuning" mean in this AGILE context?**

**A:** Here, "fine-tuning" refers to starting from a baseline RL policy and tuning a smaller set of parameters (often via sweeps) to improve performance (tracking, robustness). This is different from "foundation model fine-tuning" (LLM/VLM/VLA) which is a separate context.

**Q: How do you run hyperparameter sweeps and keep experiments organized?**

**A:** AGILE uses Weights & Biases (W&B) to log runs, track configurations, and compare results. The workflow supports sweeps over not just RL hyperparameters but also task/environment parameters (MDP terms like rewards, curriculum, events).

**Q: When W&B reports parameter importance, how do you handle correlated parameters?**

**A:** The team currently relies on planned grid search and manual iteration:
- tune to a baseline first,
- then sweep parameters they already believe matter most.

They noted W&B offers more advanced search algorithms, but they have not tried those yet.

---

## Robustness and Error Recovery

**Q: If the robot does something wrong, can it self-correct? Will it "behave like a child"?**

**A:** During training, the team injects disturbances (push torso/hand) and apply randomization so the policy experiences varied conditions. For locomotion, the work also randomizes upper-body motion during training so the lower-body controller learns to cope. However, disturbances beyond the policy/robot limits (e.g., a hard kick) can still cause failure.

---

## Embodiments and Adding New Robots

**Q: How hard is it to add a new embodiment (example: H1)?**

**A:** AGILE aims to reduce effort by:
- keeping MDP terms as robot-agnostic as possible,
- putting robot-specific details (joint/link names, limits, actuator configs) in a single robot asset/config file,
- emphasizing model verification early (contact geometry, joint limits, PD gains, actuator model).

They showed that similar tasks across robots (e.g., G1 vs T1 velocity tracking) can share almost identical configurations aside from minimal robot-specific parameters.

**Q: What are the biggest causes of poor policies when adding a robot?**

**A:** Incorrect model setup is the most common: wrong joint scales/ranges, incorrect contact geometry/sensors, mismatched PD gains, etc. These issues can waste days/weeks if discovered late.

---

## Tools and UI

**Q: Is the model/parameter visualization UI part of Isaac Sim or AGILE?**

**A:** It is part of the AGILE repo. You install Isaac Lab first, then install AGILE on top, and the tooling comes with AGILE.

**Q: Do you have heatmaps (power activation, balance), or a robot "pilot/ego" view?**

**A:** AGILE does not currently provide those visualizations. For first-person view in teleop/calibration, Isaac Lab has VR teleop examples that can provide an ego-view workflow. Check this [link](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#cloudxr-teleoperation) for Cloud XR setup and this [link](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html) for data collection with teleoperation.

---

## Teleoperation and Deployment

**Q: Is MuJoCo transfer supported? How is deployment handled?**

**A:** AGILE supports MuJoCo evaluation with a trained policy checkpoint and matching YAML config, so policies can be tested across simulators and compared across architectures/observations without rewriting the pipeline.

**Q: Beyond keyboard teleop, what devices can be used?**

**A:** AGILE is mainly designed for policy training. The trained policy can be then copied to IsaacLab for teleoperation. IsaacLab provides keyboard, spacemouse and Cloud XR teleoperation devices. An example of using AGILE in such a setup can be found in this [link](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#demo-2-data-generation-and-policy-training-for-humanoid-robot-locomanipulation-with-unitree-g1).

---

## Joint Control and Actuator Modeling

**Q: What control/actuator models are used for joints during training?**

**A:** AGILE provides two common actuator setups and both are verified with real world testing:
- a delayed motor configuration to model more realistic actuator behavior (e.g., torque-speed envelope near limits),
- an implicit actuator setup where the simulator handles integration.

They indicated both have been validated on real robots.

---

## Modular Control and Distillation

**Q: If lower and upper body are trained separately, how do you prevent "policy exploitation" between controllers?**

**A:** In the shown approach, the team does not train the two controllers together. They train locomotion first, then freeze it and use it as an API while training the upper-body policy, reducing non-stationarity between controllers. This approach is similar to the work [VIRAL](https://arxiv.org/pdf/2511.15200).

---

## Generalization and Foundation-Style Policies

**Q: Are there task-agnostic/generalizable policies available?**

**A:** Not directly from AGILE, but they referenced NVIDIA work (from GR00T lab) called "Sonic" described as a general tracking policy that can follow many reference motions. See [link](https://nvlabs.github.io/SONIC/).

**Q: Are you planning a full VLA pipeline (data → training scale → inference) in Isaac Lab soon?**

**A:** The team acknowledged the question and said they have relevant use cases to share, but did not provide a specific timeline or commitment in the transcript. Here is a suite of eval [tasks](https://github.com/isaac-sim/IsaacLabEvalTasks) that does the pipeline, which involves both Isaac Lab and GR00T.

---

## Physics Solvers and Sim-to-Sim

**Q: Can policies transfer across physics solvers (PhysX vs Newton)?**

**A:** This is not the focus of AGILE. The team noted that internal experiments running policies on both PhysX and Newton, and suggested that once Newton is officially integrated into Isaac Lab, switching solvers should become easier.

---

## Multi-Arm / Non-Humanoid Setups

**Q: Can AGILE train coordinated multi-arm systems (e.g., dual Franka arms)?**

**A:** The team suggested yes in principle. AGILE's architecture is flexible and many MDP terms can be reused for different robot setups, including tasks where multiple effectors coordinate.

---

## Domain Randomization

**Q: What domain randomization strategy works best for sim-to-sim adaptation?**

**A:** The team suggested the reader to read the [training tips](docs/source/training-tips.md) page with guidance on randomization terms and practical recommendations. Empirical experience shows that randomization within proper armature and PD gain ranges matters a lot.
