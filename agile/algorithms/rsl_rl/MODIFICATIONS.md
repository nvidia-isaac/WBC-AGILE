## [rsl_rl](https://github.com/leggedrobotics/rsl_rl) Modifications

This module modifies the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) reinforcement learning library (based on [v2.3.3](https://github.com/leggedrobotics/rsl_rl/releases/tag/v2.3.3)).

The following modifications have been implemented:

### Support for [TensorDict](https://docs.pytorch.org/tensordict/stable/index.html)
The actor and critic implementation now supports both torch.Tensors and TensorDicts as observations.

### Teacher-Student Setup
The teacher-student distillation setup has been redesigned. The teacher is now assumed to be a JIT-exported policy that is directly loaded without requiring architecture reconstruction. This eliminates the need for a dedicated teacher-student module, instead utilizing a separate teacher network (standard RL) and the exported policy for student training.

### Entropy Annealing
Implementation of entropy coefficient scheduling during training.

### Distillation Symmetry Loss
Implemented a symmetry loss term during distillation to explicitly encourage the student policy to match the symmetry properties present in the teacher policy.

### Locally Lipschitz Continuous Constraint
Implemented [L2C2](https://arxiv.org/abs/2202.07152) (Locally Lipschitz Continuous Constraint) regularization, which directly penalizes the actor and critic to encourage smoother motion.
