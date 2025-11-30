import torch
import numpy as np
import pytorch3d.transforms

from mppiisaac.utils.mppi_utils import quat_to_yaw

# Default weights identical to examples/xarm6_gripper_push/server.py for task "push"
PUSH_WEIGHTS = {
    "w_robot_to_block_pos": 5.0,
    "w_block_to_goal_pos": 30.0,
    "w_block_to_goal_ori": 10.0,
    "w_ee_hover": 8.5,
    "w_ee_align": 0.5,
    "w_push_align": 15.0,
}


def compute_push_cost(sim, goal_pose: torch.Tensor, weights: dict = PUSH_WEIGHTS):
    """
    Compute push task cost matching Objective.compute_cost in server.py.
    """
    ee_idx = sim.get_link_index("xarm6", "xarm6_ee_tip")
    ee_pos = sim.rigid_body_state[:, ee_idx, :3]
    ee_ori = sim.rigid_body_state[:, ee_idx, 3:7]

    block_idx = sim.get_actor_index("obj_to_push")
    block_pos = sim.root_state[:, block_idx, :3]
    block_ori = sim.root_state[:, block_idx, 3:7]

    ee_height = ee_pos[:, 2]
    robot_to_block = ee_pos - block_pos
    block_to_goal = goal_pose[0:2] - block_pos[:, 0:2]

    robot_euler = pytorch3d.transforms.matrix_to_euler_angles(
        pytorch3d.transforms.quaternion_to_matrix(ee_ori), "ZYX"
    )
    block_yaw = quat_to_yaw(block_ori[:, 0], block_ori[:, 1], block_ori[:, 2], block_ori[:, 3])
    goal_yaw = quat_to_yaw(goal_pose[3], goal_pose[4], goal_pose[5], goal_pose[6])

    robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis=1)
    block_to_goal_dist = torch.linalg.norm(block_to_goal, axis=1)
    block_to_goal_ori = torch.abs(block_yaw - goal_yaw)
    ee_hover_dist = torch.abs(ee_height - 0.03)
    ee_align = torch.linalg.norm(robot_euler[:, 0:2], axis=1)
    push_align = torch.sum(robot_to_block[:, 0:2] * block_to_goal, axis=1) / (
        torch.linalg.norm(robot_to_block[:, 0:2], axis=1) * torch.linalg.norm(block_to_goal, axis=1) + 1e-6
    ) + 1

    w = weights
    total_cost = (
        w["w_robot_to_block_pos"] * robot_to_block_dist
        + w["w_block_to_goal_pos"] * block_to_goal_dist
        + w["w_block_to_goal_ori"] * block_to_goal_ori
        + w["w_ee_hover"] * ee_hover_dist
        + w["w_ee_align"] * ee_align
        + w["w_push_align"] * push_align
    )
    return total_cost
