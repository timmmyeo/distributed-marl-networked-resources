import gym
from gym import spaces
import pandas as pd
import numpy as np

class MultiDataCenterEnvironment(gym.Env):
  metadata = {"render_modes": ["human"]}

  def __init__(self, datacentres: list[dict[str, pd.DataFrame]], end_time: int):
    # super(MultiDataCenterEnvironment, self).__init__()
    
    # Each element is a datacenter consisting of machines (indexed from 0 to n-1):
    # [
    #   {0: pd.DataFrame, 1: pd.DataFrame ...},
    #   {0: pd.DataFrame, 1: pd.DataFrame ...}
    # ]
    self.datacentres = [{f"{i}": datacentre[machine_id] for (i, machine_id) in enumerate(datacentre)} for datacentre in datacentres]

    # Current time of the simulation
    self.current_time = 0
    # End time of the simulation
    self.end_time = end_time

    # [cpu1, cpu2, ..., cpu10, workload_requirement, data_center]
    self.observation_space = spaces.Dict(
      {
        **{f"{datacentre_number}": spaces.Box(low=0, high=100, shape=(len(datacentre),), dtype=np.float32) for datacentre_number, datacentre in enumerate(self.datacentres)},
        "workload": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
        "workload_datacentre": spaces.Discrete(2),
      }
    )
    
    # One action per machine and a no-op; choose to send the workload to a machine, or choose not to do anything
    self.action_space = spaces.Discrete(11)
  
  # def _get_obs(self):
  #   return {
      
  #   }
    
  def _generate_workload(self):
    cpu_requirement = np.random() * 100
    return cpu_requirement
  
  def _get_workload_datacentre(self, epsilon=0.5):
    r = np.random()
    data_centre_to_send_work = self._workload_datacentre
    if r <= epsilon:
      data_centre_to_send_work = 1 - data_centre_to_send_work
    return data_centre_to_send_work
    

  
  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    self.current_time = 0
    self._datacentre_to_send_work = 0

    # Each machine in a datacenter's first row in the pandas dataframe
    # [
    #   {1: 80.4, 2: 50.3 ...},
    #   {1: 30.5, 2: 40.1 ...}
    # ]
    self._datacentres_curr_state = [{f"{i}": datacentre[machine_id].loc[self.current_time, 'cpu_util_percent'] for (i, machine_id) in enumerate(datacentre)} for datacentre in self.datacentres]

    self._workload = self._generate_workload()
    self._workload_datacentre = self._get_workload_datacentre()

    observation = {
      **{f"{datacentre_number}": datacentre for datacentre_number, datacentre in enumerate(self._datacentres_curr_state)},
      "workload": self._workload,
      "workload_datacentre": self._workload_datacentre,
    }
    info = None # Not sure what to use this for

    return observation, info
  
  # 10 machines, action A0 - A9, choose, no machine
  def step(self, action):
    observation = None
    reward = 0
    terminated = self.current_time == self.end_time
    info = None # Not sure what to use this for
    if action == 10:
      # Do nothing
      reward = 0
    else:
      if action < 5:
        # First data centre
        spare_capacity = 100 - self._datacentres_curr_state[0][str(action)]
        if spare_capacity >= self._workload:
          reward = 100
        else:
          reward = max(0, int(spare_capacity - self._workload))
        
        # Half the reward if the datacentre is not where the workload originated
        if self._workload_datacentre != 0:
          reward /= 2
      else:
        # Second data centre
        spare_capacity = 100 - self._datacentres_curr_state[1][str(action - 5)]
        if spare_capacity >= self._workload:
          reward = 100
        else:
          reward = max(0, int(spare_capacity - self._workload))
        
        # Half the reward if the datacentre is not where the workload originated
        if self._workload_datacentre != 1:
          reward /= 2
    
    if not terminated:
      # Advance the time
      self.current_time += 1
      self._datacentres_curr_state = [{f"{i}": datacentre[machine_id].loc[self.current_time, 'cpu_util_percent'] for (i, machine_id) in enumerate(datacentre)} for datacentre in self.datacentres]

      self._workload = self._generate_workload()
      self._workload_datacentre = self._get_workload_datacentre()
      observation = {
        **{f"{datacentre_number}": datacentre for datacentre_number, datacentre in enumerate(self._datacentres_curr_state)},
        "workload": self._workload,
        "workload_datacentre": self._workload_datacentre,
      }

    return observation, reward, terminated, False, info
    
    