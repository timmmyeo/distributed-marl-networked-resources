import gym
from gym import spaces
import pandas as pd
import numpy as np

class MultiDataCenterEnvironment(gym.Env):
  metadata = {"render_modes": ["human"]}

  def _generate_workload(self):
    cpu_requirement = np.random.rand() * 100
    return cpu_requirement
  
  def _get_workload_datacentre(self, epsilon=0.5):
    r = np.random.rand()
    data_centre_to_send_work = self._workload_datacentre
    if r <= epsilon:
      data_centre_to_send_work = 1 - data_centre_to_send_work
    return data_centre_to_send_work

  def __init__(self, machines_data: list[list[np.float32]], datacentre_mapping: dict[int, int], num_datacentres: int):
    # super(MultiDataCenterEnvironment, self).__init__()
    
    assert len(machines_data) > 0
    assert machines_data[0] > 0
    assert num_datacentres <= len(machines_data)

    self.num_datacentres = num_datacentres
    self.machines_data = machines_data
    self.datacentre_mapping = datacentre_mapping

    # Current time of the simulation
    self.current_time = 0
    # Current state of the simulation
    self._workload = self._generate_workload()
    self._workload_datacentre = self._get_workload_datacentre()
    self._machines_curr_state: list[np.float32] = [machine_data[0] for machine_data in self.machines_data]

    # (cpu1, cpu2, ..., cpu10, workload_requirement, data_center)
    self.observation_space = spaces.Dict(
      {
        "machines_curr_state": spaces.Tuple((spaces.Box(low=0, high=100, dtype=np.float32) for _ in range(len(machines_data)))),
        "workload": spaces.Box(low=0, high=100, dtype=np.float32),
        "workload_datacentre": spaces.Discrete(num_datacentres),
      }
    )
    
    # One action per machine and a no-op; choose to send the workload to a machine, or choose not to do anything
    self.action_space = spaces.Discrete(len(machines_data) + 1)
  
  # def _get_obs(self):
  #   return {
      
  #   }

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    # Current time of the simulation
    self.current_time = 0
    # Current state of the simulation
    self._workload = self._generate_workload()
    self._workload_datacentre = self._get_workload_datacentre()
    self._machines_curr_state: list[np.float32] = [machine_data[0] for machine_data in self.machines_data]

    observation = {
      "machines_curr_state": tuple(self._machines_curr_state),
      "workload": self._workload,
      "workload_datacentre": self._workload_datacentre,
    }
    info = {} # Not sure what to use this for

    return observation, info
  
  # 10 machines, action A0 - A9, choose, no machine
  def step(self, action):
    observation = None
    reward = 0
    done = self.current_time >= (len(self.machines_data) - 2)
    info = {} # Not sure what to use this for

    # Apply the action
    if action == 10:
      # Do nothing
      reward = 0
    else:
      machine_picked = action
      spare_capacity = 100 - self._machines_curr_state[machine_picked]
      if spare_capacity >= self._workload:
          reward = 100
      else:
        reward = max(0, int(spare_capacity - self._workload))

      # Half the reward if the datacentre is not where the workload originated
        if self._workload_datacentre != self.datacentre_mapping[machine_picked]:
          reward /= 2
    
    # Advance the time
    self.current_time = 0
    # Advance the state of the simulation
    self._workload = self._generate_workload()
    self._workload_datacentre = self._get_workload_datacentre()
    self._machines_curr_state: list[np.float32] = [machine_data[0] for machine_data in self.machines_data]

    observation = {
      "machines_curr_state": tuple(self._machines_curr_state),
      "workload": self._workload,
      "workload_datacentre": self._workload_datacentre,
    }

    return observation, reward, done, False, info
    
    