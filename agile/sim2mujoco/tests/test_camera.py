import numpy as np

from agile.sim2mujoco.camera import robot_root_body_id, robot_tracking_target


def test_camera_tracks_the_resolved_free_joint_root() -> None:
    body_id = robot_root_body_id(np.array([4, 2]), np.array([3, 0]), free_joint_type=0)
    assert body_id == 2
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 4], [0, 0, 0], [9, 9, 9]])
    np.testing.assert_array_equal(robot_tracking_target(positions, body_id), [2, 3, 4])
