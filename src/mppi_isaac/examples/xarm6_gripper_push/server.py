import os
import gym
import time
import hydra
import zerorpc
import numpy as np

from omegaconf import OmegaConf
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.prior.offline_sac_prior import OfflineSACPrior, PriorConfig

import torch
import pytorch3d.transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from mppiisaac.utils.mppi_utils import quat_to_yaw
from mppiisaac.utils.config_store import ExampleConfig


class Objective(object):
    """ Objective function for the push task """
    def __init__(self, cfg, device):
        # Choose task
        self.task = "push" # push, hovering

        if self.task == "push":
            self.ee_hover_height = 0.03
            self.w_robot_to_block_pos = 5
            self.w_block_to_goal_pos =  30
            self.w_block_to_goal_ori =  10
            self.w_ee_hover =           8.5
            self.w_ee_align =           0.5
            self.w_push_align =         15

        elif self.task == "hovering":
            self.ee_hover_height = 0.3
            self.w_robot_to_block_pos = 30
            self.w_block_to_goal_pos =  0
            self.w_block_to_goal_ori =  0
            self.w_ee_hover =           15
            self.w_ee_align =           15
            self.w_push_align =         0
    
        # Actor indices
        self.ee_index = 11  
        self.block_index = 1 

        # Set goal pose
        self.goal_pose = torch.tensor([0.7, 0.2, 0.5] + [0.0, 0.0, 0.258819, 0.9659258], device="cuda:0") # Rotation 30 deg

        # Select goal pose and compute yaw
        self.goal_pose = torch.clone(self.goal_pose)
        self.goal_ori = torch.clone(self.goal_pose[3:7])
        self.goal_yaw = quat_to_yaw(self.goal_ori[0], self.goal_ori[1], self.goal_ori[2], self.goal_ori[3])

        # Push task specific parameters
        self.robot_ref_euler = torch.tensor([0, 0, 0], device=cfg.mppi.device)

       # Setting TensorBoard Logging
        self.step_count = 0
        log_dir = f"logs/server"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        print(f"[Server] TensorBoard logging to: {log_dir}")

        self.log_frequency = getattr(cfg, 'log_frequency', 10)        # Default: log every 10 steps
        self.log_every_step = getattr(cfg, 'log_every_step', False)   # Option to log every step
        if self.log_every_step:                                       # Override log_frequency if log_every_step is True
            self.log_frequency = 1
        print(f"[Server] Logging frequency: every {self.log_frequency} step(s)")


    def compute_metrics(self, block_pos, block_ori):
        """ Compute metrics for evaluating the push task """
        block_yaw = quat_to_yaw(block_ori[:, 0], block_ori[:, 1], block_ori[:, 2], block_ori[:, 3])
        Ex = torch.abs(self.goal_pose[0] - block_pos[:, 0])
        Ey = torch.abs(self.goal_pose[1] - block_pos[:, 1])
        Etheta = torch.abs(block_yaw - self.goal_yaw)

        return Ex, Ey, Etheta
    

    def compute_cost(self, sim):
        """ Compute cost for evaluating the push task """
        # 1) End-effector (robot component) from rigid_body_state
        ee_idx = sim.get_link_index("xarm6", "xarm6_ee_tip")
        ee_pos = sim.rigid_body_state[:, ee_idx, :3]
        ee_ori = sim.rigid_body_state[:, ee_idx, 3:7]

        # 2) Block (object) from root_state
        block_idx = sim.get_actor_index("obj_to_push")
        block_pos = sim.root_state[:, block_idx, :3]
        block_ori = sim.root_state[:, block_idx, 3:7]

        # End-effector height
        ee_height = ee_pos[:, 2]

        # Distances between robot pos and block pos
        robot_to_block = ee_pos - block_pos

        # Distances between block pos and goal pos
        block_to_goal = self.goal_pose[0:2] - block_pos[:, 0:2]

        # End-effector euler angles
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(ee_ori), "ZYX")   
        robot_euler_deg = robot_euler * 180 / np.pi

        # Block euler angles
        block_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(block_ori), "ZYX")
        block_euler_deg = block_euler * 180 / np.pi

        # Block yaw angle
        block_yaw = quat_to_yaw(block_ori[:, 0], block_ori[:, 1], block_ori[:, 2], block_ori[:, 3])

        # Distance costs
        robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis=1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis=1)

        # Orientation costs
        block_to_goal_ori = torch.abs(block_yaw - self.goal_yaw)

        # End-effector height costs
        ee_hover_dist = torch.abs(ee_height - self.ee_hover_height)

        # End-effector orientation costs
        ee_align = torch.linalg.norm(robot_euler[:, 0:2] - self.robot_ref_euler[0:2], axis=1)
        
        # End-effector push align costs
        push_align = torch.sum(robot_to_block[:, 0:2] * block_to_goal, axis=1) / (robot_to_block_dist * block_to_goal_dist) + 1
        
        total_cost = (
            self.w_robot_to_block_pos * robot_to_block_dist    # scale: [0, ee_to_block_dist]
            + self.w_block_to_goal_pos * block_to_goal_dist    # scale: [0, block_to_goal_dist]
            + self.w_block_to_goal_ori * block_to_goal_ori     # scale: [0, pi]
            + self.w_ee_hover * ee_hover_dist                  # scale: [0, ee_height - hover_height]
            + self.w_ee_align * ee_align                       # scale: [0, (ee_roll - ref_roll)^2 + (ee_pitch - ref_pitch)^2]
            + self.w_push_align * push_align                   # scale: [0, 2]
        )

        # Log end-effector pose from the parallel world
        self.writer.add_scalar('Position/EE_X', ee_pos[:, 0].mean().item(), self.step_count)
        self.writer.add_scalar('Position/EE_Y', ee_pos[:, 1].mean().item(), self.step_count)
        self.writer.add_scalar('Position/EE_Z', ee_pos[:, 2].mean().item(), self.step_count)
        self.writer.add_scalar('Orientation/EE_Roll', robot_euler_deg[:, 0].mean().item(), self.step_count)
        self.writer.add_scalar('Orientation/EE_Pitch', robot_euler_deg[:, 1].mean().item(), self.step_count)
        self.writer.add_scalar('Orientation/EE_Yaw', robot_euler_deg[:, 2].mean().item(), self.step_count)

        # Log block pose from the parallel world
        self.writer.add_scalar('Position/Block_X', block_pos[:, 0].mean().item(), self.step_count)
        self.writer.add_scalar('Position/Block_Y', block_pos[:, 1].mean().item(), self.step_count)
        self.writer.add_scalar('Position/Block_Z', torch.abs(block_pos[:, 2]).mean().item(), self.step_count)
        self.writer.add_scalar('Orientation/Block_Roll', torch.abs(block_euler_deg[:, 0]).mean().item(), self.step_count)
        self.writer.add_scalar('Orientation/Block_Pitch', torch.abs(block_euler_deg[:, 1]).mean().item(), self.step_count)
        self.writer.add_scalar('Orientation/Block_Yaw', block_euler_deg[:, 2].mean().item(), self.step_count)

        # Log cost components
        if self.step_count % self.log_frequency == 0:
            # Cost components by average
            self.writer.add_scalar('Costs/robot_to_block_dist', robot_to_block_dist.mean().item(), self.step_count)
            self.writer.add_scalar('Costs/block_to_goal_dist', block_to_goal_dist.mean().item(), self.step_count)
            self.writer.add_scalar('Costs/block_to_goal_ori', block_to_goal_ori.mean().item(), self.step_count)
            self.writer.add_scalar('Costs/ee_hover_dist', ee_hover_dist.mean().item(), self.step_count)
            self.writer.add_scalar('Costs/ee_align', ee_align.mean().item(), self.step_count)
            self.writer.add_scalar('Costs/push_align', push_align.mean().item(), self.step_count)

            # Weighted cost components by average
            self.writer.add_scalar('Weighted_Costs/robot_to_block_dist_weighted', (self.w_robot_to_block_pos * robot_to_block_dist).mean().item(), self.step_count)
            self.writer.add_scalar('Weighted_Costs/block_to_goal_dist_weighted', (self.w_block_to_goal_pos * block_to_goal_dist).mean().item(), self.step_count)
            self.writer.add_scalar('Weighted_Costs/block_to_goal_ori_weighted', (self.w_block_to_goal_ori * block_to_goal_ori).mean().item(), self.step_count)
            self.writer.add_scalar('Weighted_Costs/ee_hover_dist_weighted', (self.w_ee_hover * ee_hover_dist).mean().item(), self.step_count)
            self.writer.add_scalar('Weighted_Costs/ee_align_weighted', (self.w_ee_align * ee_align).mean().item(), self.step_count)
            self.writer.add_scalar('Weighted_Costs/push_align_weighted', (self.w_push_align * push_align).mean().item(), self.step_count)

            # Total cost by average
            self.writer.add_scalar('Costs/total_cost', total_cost.mean().item(), self.step_count)

        self.step_count += 1

        return total_cost


@hydra.main(version_base=None, config_path="../../conf", config_name="config_xarm6_gripper_push")
def run_xarm6_robot(cfg: ExampleConfig):
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg, cfg.mppi.device)
    prior = None
    if getattr(cfg, "prior_checkpoint", None):
        print(f"[Server] Loading prior from {cfg.prior_checkpoint}")
        prior = OfflineSACPrior(
            PriorConfig(
                checkpoint=cfg.prior_checkpoint,
                device=cfg.mppi.device,
            )
        )
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=prior))
    planner.bind("tcp://0.0.0.0:4242")
    
    try:
        planner.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping MPPI server.")
    finally:
        if hasattr(objective, 'writer'):
            objective.writer.close()
            print(f"TensorBoard writer closed.")

if __name__ == "__main__":
    run_xarm6_robot()
