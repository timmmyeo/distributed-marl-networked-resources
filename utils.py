import random
from collections import deque

import torch
import torch.nn.functional as F
from gymnasium.core import Env
from torch import nn

class ReplayBuffer():
    def __init__(self, size:int, early_transitions_buffer_percentage:int = 0.1):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.early_transitions_buffer_percentage = early_transitions_buffer_percentage
        self.early_transitions_buffer_size = int(early_transitions_buffer_percentage * size)
        # Reserve part of the buffer (default 10%) separate; do not overwrite any transitions here once filled
        self.early_transitions_buffer = deque([], self.early_transitions_buffer_size)
        self.rest_buffer_size = size - self.early_transitions_buffer_size
        self.rest_buffer = deque([], self.rest_buffer_size)
    
    def get_curr_size(self):
        return len(self.early_transitions_buffer) + len(self.rest_buffer)

    
    def push(self, transition):
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """
        # Fill up the early_transitions_buffer until it is full
        if len(self.early_transitions_buffer) < self.early_transitions_buffer_size:
            self.early_transitions_buffer.append(transition)
        # If the early_transitions_buffer is full, we use the rest_buffer as usual
        else:
            self.rest_buffer.append(transition)
        return list(self.early_transitions_buffer) + list(self.rest_buffer)

    def sample(self, batch_size:int):
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        # Return a random sample from both buffers combined
        return random.sample(list(self.early_transitions_buffer) + list(self.rest_buffer), batch_size)


class DQN(nn.Module):
    def __init__(self, layer_sizes):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        # torch.manual_seed(14597905165985114927) - This is a bad seed for a network of [4, 256, 2]
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor, ddqn = False)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """
    if ddqn:
        # DDQN Start:
        # torch.Size([BATCH_SIZE, 1])
        policy_dqn_best_action_batch = torch.argmax(policy_dqn(next_states), dim=1).reshape(-1, 1) # select best actions based on policy dqn
        # torch.Size([BATCH_SIZE, 2])
        target_dqn_next_states = target_dqn(next_states)
        # torch.Size([BATCH_SIZE, 1])
        selected_action_batch = target_dqn_next_states.gather(dim=1, index=policy_dqn_best_action_batch) # use actions on the target dqn

        # dones shape: torch.Size([BATCH_SIZE, 1]), reshaped(-1): torch.Size.([BATCH_SIZE])
        # rewards shape: torch.Size([BATCH_SIZE, 1]), reshaped(-1): torch.Size([BATCH_SIZE])
        # selected_action_batch reshaped(-1): torch.Size([BATCH_SIZE])
        # bellman_targets shape: torch.Size([BATCH_SIZE])
        bellman_targets = (~dones).reshape(-1)*(selected_action_batch).reshape(-1) + rewards.reshape(-1)
        # DDQN End
    else:
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)

    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()