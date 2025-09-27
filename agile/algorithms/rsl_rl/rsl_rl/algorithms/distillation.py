# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDictBase
from rsl_rl.utils import flatten_dict, string_to_callable

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        device="cpu",
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg.get("use_mirror_loss", False)
            # Check if policy is recurrent
            is_recurrent = hasattr(policy, 'is_recurrent') and policy.is_recurrent
            
            # Disable symmetry for recurrent policies
            if use_symmetry and is_recurrent:
                print(
                    "[WARNING] Symmetry loss is not supported for recurrent policies. "
                    "Symmetry will be disabled for this training run."
                )
                symmetry_cfg = None
                self.symmetry = None
            else:
                # If function is a string then resolve it to a function
                if isinstance(symmetry_cfg.get("data_augmentation_func"), str):
                    symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
                # Check valid configuration
                if use_symmetry and not callable(symmetry_cfg.get("data_augmentation_func")):
                    raise ValueError(
                        "Mirror loss enabled but the data_augmentation_func is not callable: "
                        f"{symmetry_cfg.get('data_augmentation_func')}"
                    )
                # Store symmetry configuration
                self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        teacher_obs_shape,
        actions_shape,
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            rnd_state_shape=None,
            use_l2c2=False,
            device=self.device,
        )

    def act(self, obs, teacher_obs):
        # compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        if isinstance(teacher_obs, TensorDictBase):
            self.transition.privileged_actions = self.policy.evaluate(flatten_dict(teacher_obs)).detach()
        else:
            self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        loss = torch.tensor(0.0, device=self.device)
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, dones in self.storage.generator():

                # inference the student for gradient computation
                actions = self.policy.act_inference(obs)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # Symmetry loss (only for non-recurrent policies)
                if self.symmetry and self.symmetry.get("use_mirror_loss", False):
                    # Get the data augmentation function
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    
                    # Apply augmentation to observations and actions
                    obs_aug, actions_aug = data_augmentation_func(
                        obs=obs,
                        actions=actions.detach(),  # Use detached actions for augmentation
                        env=self.symmetry["_env"],
                        obs_type="policy",
                    )
                    
                    # Get the augmented batch size info
                    # Handle both TensorDict and Tensor cases
                    if isinstance(obs, TensorDictBase):
                        original_batch_size = obs.batch_size[0]
                    else:
                        original_batch_size = obs.shape[0]
                    
                    # Compute actions only for the mirrored observations
                    if isinstance(obs_aug, TensorDictBase):
                        obs_mirrored = obs_aug[original_batch_size:]
                    else:
                        obs_mirrored = obs_aug[original_batch_size:]
                    
                    actions_sym = self.policy.act_inference(obs_mirrored)
                    
                    # Compute symmetry loss between predicted and augmented actions
                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        actions_sym,
                        actions_aug[original_batch_size:].detach()
                    )
                    
                    # Add symmetry loss to total loss with coefficient
                    mirror_loss_coeff = self.symmetry.get("mirror_loss_coeff", 0.1)
                    loss = loss + behavior_loss + mirror_loss_coeff * symmetry_loss
                    
                    if mean_symmetry_loss is not None:
                        mean_symmetry_loss += symmetry_loss.item()
                else:
                    # No symmetry loss
                    loss = loss + behavior_loss
                
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = torch.tensor(0.0, device=self.device)

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
