import gym
from gym import spaces
import pandas as pd
import numpy as np

class MultiDataCenterEnvironment(gym.Env):
  def __init__(self, datacenters: list[dict[str, pd.DataFrame]]):
    # super(MultiDataCenterEnvironment, self).__init__()
    
    # Mapping from machine name to time series dataframe in each datacenter
    self.datacenters = datacenters

    # Current time of the simulation
    self.current_time = 0

    # [cpu1, cpu2, ..., cpu10, workload_requirement, data_center]
    self.observation_space = spaces.Dict(
      {
        "datacenter2": spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32),
        "datacenter1": spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32),
        "workload": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
        "workload_data_center": spaces.Discrete(2),
      }
    )
    self.action_space = spaces.Discrete(2)
  
  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)
  
    observation
    