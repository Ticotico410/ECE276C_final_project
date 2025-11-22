from dataclasses import dataclass, field
from mppiisaac.planner.mppi import MPPIConfig
from mppiisaac.planner.isaacgym_wrapper import IsaacGymConfig, ActorWrapper
from hydra.core.config_store import ConfigStore

from typing import List, Optional


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    isaacgym: IsaacGymConfig
    nx: int
    actors: List[str]
    initial_actor_positions: List[List[float]]
    viewer: bool

cs = ConfigStore.instance()
cs.store(name="config_xarm6_gripper_push", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
cs.store(group="isaacgym", name="base_isaacgym", node=IsaacGymConfig)


from hydra import compose, initialize
from omegaconf import OmegaConf
def load_isaacgym_config(name):
    with initialize(config_path="../../conf"):
        cfg = compose(config_name=name)
        print(OmegaConf.to_yaml(cfg))
    return cfg