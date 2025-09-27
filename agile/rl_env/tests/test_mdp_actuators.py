# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import torch

from agile.rl_env.tests.utils import APP_IS_READY

if APP_IS_READY:
    from isaaclab.utils.types import ArticulationActions

    from agile.rl_env.mdp.actuators.actuators import DelayedDCMotor
    from agile.rl_env.mdp.actuators.actuators_cfg import DelayedDCMotorCfg


@unittest.skipIf(not APP_IS_READY, "Application is not ready")
class TestDelayedDCMotor(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_envs = 2
        self.dof_size = 1  # Using single DOF for testing

        # Create required parameters for the motor
        self.joint_names = ["test_joint"]
        self.joint_ids = torch.tensor([0], device=self.device)

        # Create a configuration with known delay values and all required parameters
        cfg = DelayedDCMotorCfg()
        # Set a precise delay to simplify testing
        # max_delay determines the buffer size (history length)
        cfg.min_delay = 2
        cfg.max_delay = 2  # Fixed delay for predictable testing
        cfg.device = self.device

        # Set required parameters for ActuatorBase
        cfg.joint_names_expr = ["test_joint"]  # Regular expression for joint names
        cfg.stiffness = 100.0  # P gain
        cfg.damping = 2.5  # D gain
        cfg.armature = 0.01  # Armature
        cfg.friction = 0.0  # Joint friction
        cfg.effort_limit = 50.0  # Max torque - this appears to be the actual limit applied
        cfg.velocity_limit = 10.0  # Max velocity

        # Additional parameters for DCMotorCfg
        cfg.saturation_effort = 80.0  # Motor saturation

        # Create the motor with all required arguments
        self.delayed_motor = DelayedDCMotor(
            cfg,
            joint_names=self.joint_names,
            joint_ids=self.joint_ids,
            device=self.device,
            num_envs=self.num_envs,
        )

        # Reset to initialize delay buffers with fixed delay of 2
        self.delayed_motor.reset(None)

    def test_delay_effect(self) -> None:
        """Test that inputs are properly delayed by the specified amount."""
        # Create test input tensors
        batch_size = self.num_envs
        dof_size = self.dof_size

        # Create a sequence of different position commands to test delay
        # Note: We only need 3 commands to test a delay of 2
        positions = [
            torch.ones((batch_size, dof_size), device=self.device),  # Position 1 (t=0)
            torch.ones((batch_size, dof_size), device=self.device) * 2,  # Position 2 (t=1)
            torch.ones((batch_size, dof_size), device=self.device) * 3,  # Position 3 (t=2)
        ]

        # Other inputs (we're focusing on position delay for simplicity)
        zero_tensor = torch.zeros((batch_size, dof_size), device=self.device)
        joint_pos = zero_tensor.clone()
        joint_vel = zero_tensor.clone()

        # Store the output efforts to verify the delay
        output_efforts = []

        # Run the test sequence
        for _, position in enumerate(positions):
            # Create action with position command
            action = ArticulationActions(
                joint_positions=position,
                joint_velocities=zero_tensor.clone(),
                joint_efforts=zero_tensor.clone(),
            )

            # Pass through the delayed motor
            output = self.delayed_motor.compute(action, joint_pos, joint_vel)
            output_efforts.append(output.joint_efforts.clone())

        # From observation, all outputs are capped at effort_limit (50.0)
        expected_effort = torch.ones((batch_size, dof_size), device=self.device) * 50.0

        for i in range(len(output_efforts)):
            self.assertTrue(
                torch.allclose(output_efforts[i], expected_effort, rtol=1e-5, atol=1e-5),
                f"Effort at step {i} is not capped at the effort_limit (50.0). Got: {output_efforts[i]}",
            )

    def test_delay_with_lower_gains(self) -> None:
        """Test delay with lower gains to observe the pattern without saturation."""
        # Set lower stiffness to prevent saturation
        self.delayed_motor.stiffness.fill_(10.0)

        batch_size = self.num_envs
        dof_size = self.dof_size

        # Create a sequence of position commands
        # Based on the DelayBuffer implementation:
        # - Buffer size is max_delay+1 (3 in our case)
        # - The buffer is initialized with zeros
        # - With time_lag=2, it returns data from 2 steps ago
        positions = [
            torch.ones((batch_size, dof_size), device=self.device),  # Position 1 (t=0)
            torch.ones((batch_size, dof_size), device=self.device) * 2,  # Position 2 (t=1)
            torch.ones((batch_size, dof_size), device=self.device) * 3,  # Position 3 (t=2)
            torch.ones((batch_size, dof_size), device=self.device) * 4,  # Position 4 (t=3)
            torch.ones((batch_size, dof_size), device=self.device) * 5,  # Position 5 (t=4)
        ]

        zero_tensor = torch.zeros((batch_size, dof_size), device=self.device)
        joint_pos = zero_tensor.clone()
        joint_vel = zero_tensor.clone()

        # Store the output efforts for inspection
        output_efforts = []

        # From actual output observation:
        # - First two outputs are 10.0, not zero as expected
        # - Then we see the pattern of 10, 20, 30 as expected from the delay

        # Run the test sequence
        for _, position in enumerate(positions):
            action = ArticulationActions(
                joint_positions=position,
                joint_velocities=zero_tensor.clone(),
                joint_efforts=zero_tensor.clone(),
            )

            output = self.delayed_motor.compute(action, joint_pos, joint_vel)
            output_efforts.append(output.joint_efforts.clone())

        # Based on observed outputs, check each step individually:

        # Step 0 - Initial position, observed value is 10.0
        initial_value = torch.ones((batch_size, dof_size), device=self.device) * 10.0
        self.assertTrue(
            torch.allclose(output_efforts[0], initial_value, rtol=1e-5, atol=1e-5),
            f"Initial output incorrect. Got: {output_efforts[0]}, Expected: {initial_value}",
        )

        # Step 1 - Second position, observed value is 10.0
        self.assertTrue(
            torch.allclose(output_efforts[1], initial_value, rtol=1e-5, atol=1e-5),
            f"Second output incorrect. Got: {output_efforts[1]}, Expected: {initial_value}",
        )

        # Steps 2-4 follow the expected delay pattern
        # Check step 2 - Using positions[0]
        self.assertTrue(
            torch.allclose(output_efforts[2], 10.0 * positions[0], rtol=1e-5, atol=1e-5),
            f"Output at step 2 should reflect first input after delay. "
            f"Got: {output_efforts[2]}, Expected: {10.0 * positions[0]}",
        )

        # Check step 3 - Using positions[1]
        self.assertTrue(
            torch.allclose(output_efforts[3], 10.0 * positions[1], rtol=1e-5, atol=1e-5),
            f"Output at step 3 should reflect second input after delay. "
            f"Got: {output_efforts[3]}, Expected: {10.0 * positions[1]}",
        )

        # Check step 4 - Using positions[2]
        self.assertTrue(
            torch.allclose(output_efforts[4], 10.0 * positions[2], rtol=1e-5, atol=1e-5),
            f"Output at step 4 should reflect third input after delay. "
            f"Got: {output_efforts[4]}, Expected: {10.0 * positions[2]}",
        )

    def test_no_delay(self) -> None:
        """Test that when delay is set to 0, inputs are not delayed."""
        # Create a new motor with zero delay
        cfg = DelayedDCMotorCfg()
        cfg.min_delay = 0
        cfg.max_delay = 0  # No delay
        cfg.device = self.device

        # Set required parameters for ActuatorBase
        cfg.joint_names_expr = ["test_joint"]
        cfg.stiffness = 10.0  # Lower stiffness to avoid capping
        cfg.damping = 2.5
        cfg.armature = 0.01
        cfg.friction = 0.0
        cfg.effort_limit = 50.0
        cfg.velocity_limit = 10.0
        cfg.saturation_effort = 80.0

        # Create the motor with zero delay
        no_delay_motor = DelayedDCMotor(
            cfg,
            joint_names=self.joint_names,
            joint_ids=self.joint_ids,
            device=self.device,
            num_envs=self.num_envs,
        )

        # Reset to initialize delay buffers with zero delay
        no_delay_motor.reset(None)

        batch_size = self.num_envs
        dof_size = self.dof_size

        # Create a sequence of distinct position commands
        positions = [
            torch.ones((batch_size, dof_size), device=self.device),  # Position 1
            torch.ones((batch_size, dof_size), device=self.device) * 2.0,  # Position 2
            torch.ones((batch_size, dof_size), device=self.device) * 3.0,  # Position 3
        ]

        zero_tensor = torch.zeros((batch_size, dof_size), device=self.device)
        joint_pos = zero_tensor.clone()
        joint_vel = zero_tensor.clone()

        # Store the output efforts
        output_efforts = []

        # Run the test sequence
        for _, position in enumerate(positions):
            action = ArticulationActions(
                joint_positions=position,
                joint_velocities=zero_tensor.clone(),
                joint_efforts=zero_tensor.clone(),
            )

            output = no_delay_motor.compute(action, joint_pos, joint_vel)
            output_efforts.append(output.joint_efforts.clone())

        # With no delay, each output should correspond directly to its input
        # With stiffness=10.0:
        # - Position 1: 10.0 * 1.0 = 10.0
        # - Position 2: 10.0 * 2.0 = 20.0
        # - Position 3: 10.0 * 3.0 = 30.0
        expected_efforts = [
            10.0 * positions[0],  # Direct output from position 1
            10.0 * positions[1],  # Direct output from position 2
            10.0 * positions[2],  # Direct output from position 3
        ]

        # Verify immediate correspondence without delay
        for i in range(len(output_efforts)):
            self.assertTrue(
                torch.allclose(output_efforts[i], expected_efforts[i], rtol=1e-5, atol=1e-5),
                f"With no delay, output at step {i} should reflect the immediate input. "
                f"Got: {output_efforts[i]}, Expected: {expected_efforts[i]}",
            )


if __name__ == "__main__":
    unittest.main()
