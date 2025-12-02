from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
from mppiisaac.planner.mppi import MPPIPlanner
import mppiisaac
from typing import Callable, Optional
import io
import os
import yaml
from yaml.loader import SafeLoader

from isaacgym import gymtorch
import torch

torch.set_printoptions(precision=2, sci_mode=False)


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)


class MPPIisaacPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable, prior: Optional[Callable] = None):
        self.cfg = cfg
        self.objective = objective

        actors = []
        for actor_name in cfg.actors:
            with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/{actor_name}.yaml') as f:
                actors.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))
        # print(actors)
        
        self.sim = IsaacGymWrapper(
            cfg.isaacgym,
            actors=actors,
            init_positions=cfg.initial_actor_positions,
            num_envs=cfg.mppi.num_samples,
            viewer=cfg.viewer,
        )

        if prior:
            def _prior_fn(state, t):
                # prior returns a batch (num_envs, act_dim); MPPI expects a single action vector.
                act = prior.compute_command(self.sim)
                if isinstance(act, tuple):
                    act = act[0]
                if act.dim() > 1:
                    act = act[0]
                return act.to(self.cfg.mppi.device)
            self.prior = _prior_fn
        else:
            self.prior = None

        self.mppi = MPPIPlanner(
            cfg.mppi,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
        )

        # Note: place_holder variable to pass to mppi so it doesn't complain, while the real state is actually the isaacgym simulator itself.
        self.state_place_holder = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))

    def dynamics(self, _, u, t=None):
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaacgym the state is already saved in the simulator itself, so we ignore it.
        # Note: t is an unused step dependent dynamics variable

        self.sim.apply_robot_cmd_velocity(u)

        self.sim.step()

        return (self.state_place_holder, u)

    def running_cost(self, _):
        # Note: again normally mppi passes the state as a parameter in the running cost call, but using isaacgym the state is already saved and accesible in the simulator itself, so we ignore it and pass a handle to the simulator.
        return self.objective.compute_cost(self.sim)

    def compute_action(self, q, qdot, obst=None, obst_tensor=None):
        self.sim.reset_root_state()
        self.sim.reset_robot_state(q, qdot)

        # NOTE: There are two different ways of updating obstacle root_states
        # Both update based on id in the list of obstacles
        if obst:
            self.sim.update_root_state_tensor_by_obstacles(obst)

        if obst_tensor:
            self.sim.update_root_state_tensor_by_obstacles_tensor(obst_tensor)

        self.sim.save_root_state()
        actions = self.mppi.command(self.state_place_holder).cpu()
        return actions

    def update_root_state_tensor_by_obstacles_tensor(self, obst_tensor):
        self.sim.update_block_state_tensor_by_obstacles(bytes_to_torch(obst_tensor))

    def reset_rollout_sim(
        self, dof_state_tensor, root_state_tensor, rigid_body_state_tensor
    ):
        self.sim.ee_positions_buffer = []
        self.sim.dof_state[:] = bytes_to_torch(dof_state_tensor)
        self.sim.root_state[:] = bytes_to_torch(root_state_tensor)
        self.sim.rigid_body_state[:] = bytes_to_torch(rigid_body_state_tensor)

        self.sim.gym.set_dof_state_tensor(self.sim.sim, gymtorch.unwrap_tensor(self.sim.dof_state))
        self.sim.gym.set_actor_root_state_tensor(self.sim.sim, gymtorch.unwrap_tensor(self.sim.root_state))

    def reset_rollout_robot(self, dof_state_tensor):
        # print("[Plan] Resetting rollout for robot only\n")
        # Clear ee_positions_buffer before new rollout
        if hasattr(self.sim, 'ee_link_present') and self.sim.ee_link_present:
            self.sim.ee_positions_buffer = []
        self.sim.dof_state[:] = bytes_to_torch(dof_state_tensor)
        self.sim.gym.set_dof_state_tensor(self.sim.sim, gymtorch.unwrap_tensor(self.sim.dof_state))

    def command(self):
        return torch_to_bytes(self.mppi.command(self.state_place_holder))

    def get_topk_trajectories(self, topk=10):
        """Get topk control sequences with lowest costs."""
        topk_actions = self.mppi.get_topk_trajectories(topk)
        if topk_actions is None:
            return None
        # Only return actions as bytes, shape validation can be done in client
        return torch_to_bytes(topk_actions)
    
    def get_topk_ee_positions(self, topk=10):
        """Get topk end-effector position trajectories (xyz) from parallel simulations."""
        # Get topk indices with lowest costs
        if self.mppi.actions is None or self.mppi.cost_total is None:
            return None
        
        _, topk_idx = torch.topk(self.mppi.cost_total, min(topk, self.mppi.K), largest=False)
        
        # Check if ee_positions_buffer has data
        if not hasattr(self.sim, 'ee_positions_buffer') or len(self.sim.ee_positions_buffer) == 0:
            return None
        
        # Stack ee_positions_buffer: list of (K, 3) -> (T, K, 3)
        ee_positions_traj = torch.stack(self.sim.ee_positions_buffer, dim=0)  # (T, K, 3)
        
        # Select topk trajectories: (T, K, 3) -> (T, topk, 3)
        topk_ee_positions = torch.index_select(ee_positions_traj, 1, topk_idx)  # (T, topk, 3)
        
        # Transpose to (topk, T, 3) to match the format expected by draw_lines_topk
        topk_ee_positions = topk_ee_positions.transpose(0, 1)  # (topk, T, 3)
        
        return torch_to_bytes(topk_ee_positions)

    def add_to_env(self, env_cfg_additions):
        self.sim.add_to_envs(env_cfg_additions)

    def get_rollouts(self):
        return torch_to_bytes(torch.stack(self.sim.ee_positions_buffer))
