import io
import os
import gym
import time
import yaml
import hydra
import zerorpc
import mppiisaac

import numpy as np

from yaml import SafeLoader
from isaacgym import gymapi
from server import Objective
from datetime import datetime
from omegaconf import OmegaConf

from mppiisaac.utils.mppi_utils import quat_to_yaw
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper

import torch
import pytorch3d.transforms
from torch.utils.tensorboard import SummaryWriter


def torch_to_bytes(t: torch.Tensor) -> bytes:
    """ Convert torch tensor to bytes """
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    """ Convert bytes to torch tensor """
    buff = io.BytesIO(b)
    return torch.load(buff)


def reset_trial(sim, init_pos, init_vel):
    """ Reset the simulation for gripper robot """
    sim.stop_sim()
    sim.start_sim()
    sim.gym.viewer_camera_look_at(sim.viewer, None, gymapi.Vec3(1.5, 2, 3), gymapi.Vec3(1.5, 0, 0))
    
    # Create DOF state (6 arm joints only, gripper joints are fixed in URDF)
    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], 
                                           init_pos[2], init_vel[2], init_pos[3], init_vel[3], 
                                           init_pos[4], init_vel[4], init_pos[5], init_vel[5],
                                          ], device="cuda:0"))
        

@hydra.main(version_base=None, config_path="../../conf", config_name="config_xarm6_gripper_push")
def run_xarm6_robot(cfg: ExampleConfig):
    """ Rollout the simulation """
    cfg = OmegaConf.to_object(cfg)

    # Create actors
    actors=[]
    for actor_name in cfg.actors:
        with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/{actor_name}.yaml') as f:
            actors.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))

    # Create simulator for real physics simulation
    sim = IsaacGymWrapper(
        cfg.isaacgym,
        init_positions=cfg.initial_actor_positions,
        actors=actors,
        num_envs=1,
        viewer=False,
    )

    # Create boxes manually                          
    table_size = [0.8, 1.0, 0.005]            # table size
    table_pos = [0.5, 0., table_size[-1]/2]   # table position

    # Candidate box sizes
    obj_list = [
        #   l      w      h      mu     m      x      y
        [0.162, 0.086, 0.068, 0.300, 0.250, 0.400, 0.000],
        [0.078, 0.078, 0.078, 0.300, 0.025, 0.400, 0.000],
    ]

    # Set goal pose
    goal_pose = torch.tensor([0.7, 0.2, 0.5] + [0.0, 0.0, 0.258819, 0.9659258], device="cuda:0") # Rotation 30 deg
    
    # Select block and goal from the candidate lists
    obj_ = obj_list[0][:]
    goal_pose = torch.clone(goal_pose)

    # Initialize block size, position and orientation
    obj_size = [obj_[0], obj_[1], obj_[2]]
    obj_init_pos = [obj_[5], obj_[6], table_size[-1] + obj_[2] / 2]
    obj_init_ori = [0.0, 0.0, 0.0, 1.0]

    # Initialize goal size, position and orientation
    goal_size = [obj_[0], obj_[1], 0.005]
    goal_pos = [goal_pose[0].item(), goal_pose[1].item(), table_size[-1]]
    goal_ori = [goal_pose[3].item(), goal_pose[4].item(), goal_pose[5].item(), goal_pose[6].item()]

    # Print environment configuration
    print("------------------------------------------------")
    print("[Info] Table size: ", table_size)
    print("[Info] Table pos: ", table_pos)
    print("[Info] Block size: ", obj_size)
    print("[Info] Block init pos: ", obj_init_pos)
    print("[Info] Block init ori: ", obj_init_ori)
    print("[Info] Goal size: ", goal_size)
    print("[Info] Goal pos: ", goal_pos)
    print("[Info] Goal ori: ", goal_ori)
    print("------------------------------------------------")

    # Create objects
    objects = [
        {
            "type": "box",
            "name": "obj_to_push",
            "size": obj_size,
            "init_pos": obj_init_pos,
            "init_ori": obj_init_ori,
            "mass": obj_[4],
            "fixed": False,
            "handle": None,
            "color": [0.2, 0.2, 0.8],
            "friction": obj_[3],
        },
        {
            "type": "box",
            "name": "table",
            "size": table_size,
            "init_pos": table_pos,
            "fixed": True,
            "handle": None,
            "color": [255 / 255, 120 / 255, 57 / 255],
            "friction": obj_[3],
        },
        {
            "type": "box",
            "name": "goal",
            "size": goal_size,
            "init_pos": goal_pos,
            "init_ori": goal_ori,
            "fixed": True,
            "color": [119 / 255, 221 / 255, 119 / 255],
            "handle": None,
            "collision": False,
        }
    ]

    # Create objects with noise
    objects_noise = [
        {
            "type": "box",
            "name": "obj_to_push",
            "size": obj_size,
            "init_pos": obj_init_pos,
            "init_ori": obj_init_ori,
            "mass": obj_[4],
            "fixed": False,
            "handle": None,
            "color": [0.2, 0.2, 1.0],
            "friction": obj_[3],
            "noise_sigma_size": [0.002, 0.002, 0.0],
            "noise_percentage_friction": 0.3,
            "noise_percentage_mass": 0.3,
        },
        {
            "type": "box",
            "name": "table",
            "size": table_size,
            "init_pos": table_pos,
            "fixed": True,
            "handle": None,
            "color": [0.2, 0.2, 1.0],
            "friction": 0.25,
            "noise_sigma_size": [0.005, 0.005, 0.0],
            "noise_percentage_friction": 0.9,
        },
        {
            "type": "box",
            "name": "goal",
            "size": goal_size,
            "init_pos": goal_pos,
            "init_ori": goal_ori,
            "fixed": True,
            "color": [119 / 255, 221 / 255, 119 / 255],
            "handle": None,
            "collision": False,
        }
    ]

    # Add objects to simulation
    sim.add_to_envs(objects)

    # Connect to MPPI client
    print("[Client] Connecting to MPPI planner...")
    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("[Client] Mppi client found!")

    # Planning with noised environment
    planner.add_to_env(objects_noise)
    
    sim.gym.viewer_camera_look_at(
        sim.viewer, None, gymapi.Vec3(1.5, 2, 3), gymapi.Vec3(1.5, 0, 0)
    )
    
    # Initialize Xarm6 joint positions and velocities
    # Robot now has only 6 DOFs (gripper joints are fixed in URDF)
    init_pos = [0.0, -0.785, -0.785, 0.0, 1.5708, 0.0]
    init_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], 
                                           init_pos[2], init_vel[2], init_pos[3], init_vel[3], 
                                           init_pos[4], init_vel[4], init_pos[5], init_vel[5],
                                          ], device="cuda:0"))

    # Helpers
    count = 0
    step_count = 0               
    n_trials = 0 
    timeout = 50
    block_index = 1
    
    data_rt = []
    data_err = []
    data_time = []
    rt_factor_seq = []
    init_time = time.time()
    client_helper = Objective(cfg, cfg.mppi.device)
    
    # Setting TensorBoard Logging 
    log_dir = f"logs/client"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[Client] TensorBoard logging to: {log_dir}")
    
    # TensorBoard logging parameters
    log_frequency = getattr(cfg, 'log_frequency', 10)        # Default: log every 10 steps
    log_every_step = getattr(cfg, 'log_every_step', False)   # Option to log every step
    if log_every_step:                                       # Override log_frequency if log_every_step is True
        log_frequency = 1
    print(f"[Client] Logging frequency: every {log_frequency} step(s)")

    while n_trials < cfg.n_steps:
        t = time.time()
        # Reset state
        planner.reset_rollout_sim(
            torch_to_bytes(sim.dof_state[0]),
            torch_to_bytes(sim.root_state[0]),
            torch_to_bytes(sim.rigid_body_state[0]),
        )
        sim.gym.clear_lines(sim.viewer)
        
        # Compute action
        action = bytes_to_torch(planner.command())
        if torch.any(torch.isnan(action)):
            print("[Error] NaN action")
            action = torch.zeros_like(action)

        # Apply action - MPPI outputs 6 velocities for 6 arm joints
        sim.set_dof_velocity_target_tensor(action)

        # Step simulator
        sim.step()

        # End-effector pose in the environment
        ee_pos = sim.rigid_body_state[0, client_helper.ee_index][:3]
        ee_ori = sim.rigid_body_state[0, client_helper.ee_index][3:7]

        # Block pose in the environment
        block_pos = sim.root_state[0, block_index][:3] 
        block_ori = sim.root_state[0, block_index][3:7] 

        # End-effector euler angles
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(ee_ori), "ZYX")   
        robot_euler_deg = robot_euler * 180 / np.pi

        # Block euler angles
        block_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(block_ori), "ZYX")
        block_euler_deg = block_euler * 180 / np.pi

        # Log end-effector pose from the single world
        writer.add_scalar('Position/EE_X', ee_pos[0].item(), step_count)
        writer.add_scalar('Position/EE_Y', ee_pos[1].item(), step_count)
        writer.add_scalar('Position/EE_Z', ee_pos[2].item(), step_count)
        writer.add_scalar('Orientation/EE_Roll', robot_euler_deg[0].item(), step_count)
        writer.add_scalar('Orientation/EE_Pitch', robot_euler_deg[1].item(), step_count)
        writer.add_scalar('Orientation/EE_Yaw', robot_euler_deg[2].item(), step_count)

        # Log block pose from the single world
        writer.add_scalar('Position/Block_X', block_pos[0].item(), step_count)
        writer.add_scalar('Position/Block_Y', block_pos[1].item(), step_count)
        writer.add_scalar('Position/Block_Z', torch.abs(block_pos[2]).item(), step_count)
        writer.add_scalar('Orientation/Block_Roll', torch.abs(block_euler_deg[0]).item(), step_count)
        writer.add_scalar('Orientation/Block_Pitch', torch.abs(block_euler_deg[1]).item(), step_count)
        writer.add_scalar('Orientation/Block_Yaw', block_euler_deg[2].item(), step_count)

        # Log xarm6 joint positions and velocities (6 arm joints)
        # DOF state is flat: [pos0, vel0, pos1, vel1, ...]
        for i in range(6):
            joint_pos = sim.dof_state[0, 2*i].item()
            joint_vel = sim.dof_state[0, 2*i+1].item()
            writer.add_scalar(f'Joint_Position/Joint_{i+1}', joint_pos, step_count)
            writer.add_scalar(f'Joint_Velocity/Joint_{i+1}', joint_vel, step_count)

        step_count += 1

        # Monitoring
        # Evaluation metrics 
        # ------------------------------------------------------------------------
        if count > 10:
            block_pos = sim.root_state[:, block_index, :3]
            block_ori = sim.root_state[:, block_index, 3:7]

            Ex, Ey, Etheta = client_helper.compute_metrics(block_pos, block_ori)
            metrics = 1.5 * (Ex + Ey) + 0.01 * Etheta
            print("\n-------------------------------------------------------")
            print("[Info] Ex:", Ex)
            print("[Info] Ey:", Ey)
            print("[Info] Etheta:", Etheta)
            print("[Info] Metrics:", metrics)

            if Ex < 0.02 and Ey < 0.02 and Etheta < 0.1:
                print("\n-------------------------------------------------------")
                print("Task success!")
                final_time = time.time()
                time_taken = final_time - init_time
                print("Time to completion:", time_taken)

                # First success, keep the current simulation state
                if n_trials == 0:
                    print("First task execution successful!")
                    data_rt.append(np.sum(rt_factor_seq) / len(rt_factor_seq))
                    data_time.append(time_taken)
                    data_err.append(np.float64(metrics))
                    n_trials += 1
                    break
                # If not the first success, reset the simulation
                else:
                    reset_trial(sim, init_pos, init_vel)    
                    init_time = time.time()
                    count = 0
                    data_rt.append(np.sum(rt_factor_seq) / len(rt_factor_seq))
                    data_time.append(time_taken)
                    data_err.append(np.float64(metrics))
                    n_trials += 1
                    step_count = 0

            rt_factor_seq.append(cfg.isaacgym.dt/(time.time() - t))
            print(f"[Info] FPS: {1/(time.time() - t)} RT-factor: {cfg.isaacgym.dt/(time.time() - t)}")
            
            count = 0
        else:
            count += 1

        if time.time() - init_time >= timeout:
            reset_trial(sim, init_pos, init_vel)
            init_time = time.time()
            count = 0
            n_trials += 1
            step_count = 0


    # To array
    data_time = np.array(data_time)
    data_rt = np.array(data_rt)
    actual_time = data_time * data_rt
    data_err = np.array(data_err)

    if len(data_time) > 0: 
        print("[Info] Number of trials:", n_trials)
        print("[Info] Success rate:", len(data_time) / n_trials * 100)
        print("[Info] Avg. Time:", np.mean(actual_time))    
        print("[Info] Std. Time:", np.std(actual_time))
        print("[Info] Avg. error:", np.mean(data_err))
        print("[Info] Std. error:", np.std(data_err))
    else:
        print("Success rate is 0")

    print("Simulation finished.")
    
    try:
        while True:
            sim.gym.fetch_results(sim.sim, True)
            sim.gym.step_graphics(sim.sim)
            sim.gym.draw_viewer(sim.viewer, sim.sim, False)  
            sim.gym.sync_frame_time(sim.sim)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping simulation.")
        sim.stop_sim()
    finally:
        print("Simulation stopped.")
        writer.close()
    return {}


if __name__ == "__main__":
    res = run_xarm6_robot()