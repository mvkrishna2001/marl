import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import time
import os
import copy  # Add import for copy module
from datetime import datetime
from skrl.agents.torch.dqn.dqn import DQN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.envs.wrappers.torch import wrap_env
from skrl.multi_agents.torch import MultiAgent

# Use gym_maze's built-in multi-agent support
from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import RandomBlockMazeGenerator

# QMIX Mixing Network as a skrl Model
class MixingNetwork(Model):
    def __init__(self, observation_space, action_space, device="cpu", clip_actions=False, 
                 num_agents=2, mixing_embed_dim=32, hypernet_embed=64):
        """
        Mixer network for QMIX that takes individual Q-values and mixes them monotonically.
        
        The network uses hypernetworks to generate weights that ensure monotonicity,
        which guarantees that the team value function increases whenever any agent's 
        individual Q-function increases.
        
        Args:
            observation_space: Global state space
            action_space: Action space (not used for mixing network)
            device: Device to use for computation ("cpu" or "cuda:X")
            clip_actions: Whether to clip actions
            num_agents: Number of agents in the environment
            mixing_embed_dim: Dimension of the mixing network
            hypernet_embed: Dimension of the hypernetwork embedding
        """
        super().__init__(observation_space, action_space, device)
        
        self.num_agents = num_agents
        self.state_dim = observation_space.shape[0]
        self.mixing_embed_dim = mixing_embed_dim
        self.hypernet_embed = hypernet_embed
        
        print(f"MixingNetwork initialized with state_dim={self.state_dim}, num_agents={self.num_agents}")
        
        # Hypernetworks to generate weights and biases for the mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, num_agents * mixing_embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed_dim)
        )
        
        self.hyper_b1 = nn.Linear(self.state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, 1)
        )
        
        self.to(device)
        
    def forward(self, agent_qs, states):
        """
        Forward pass through the mixing network.
        
        Args:
            agent_qs: Individual agent Q-values [batch_size, num_agents]
            states: Global state [batch_size, state_dim]
            
        Returns:
            q_tot: Global Q-value [batch_size, 1]
        """
        # Ensure states has the right dimensions and shape
        if states.dim() == 1:
            states = states.unsqueeze(0)  # Add batch dimension if missing
            
        # Print debug info on first forward pass
        if not hasattr(self, '_printed_shapes'):
            print(f"MixingNetwork forward: agent_qs shape={agent_qs.shape}, states shape={states.shape}")
            self._printed_shapes = True
            
        batch_size = agent_qs.size(0)
        
        # First layer weights
        w1 = self.hyper_w1(states).view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_embed_dim)
        
        # Second layer weights
        w2 = self.hyper_w2(states).view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        
        # Apply ELU activation to ensure positive weights for monotonicity
        w1 = torch.abs(w1)
        w2 = torch.abs(w2)
        
        # First layer
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # Second layer
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.squeeze(-1)


# QMIX Agent implementation using skrl's MultiAgent base class
class QMIXAgent(MultiAgent):
    def __init__(self, agents, models, mixing_network, memory, 
                 cfg=None, observation_space=None, action_space=None, device="cpu"):
        """
        Initialize the QMIX agent.
        
        QMIX is a multi-agent reinforcement learning algorithm that learns a centralized 
        value function that factors into a per-agent value function to allow decentralized execution.
        
        Args:
            agents: List of DQN instances
            models: List of models
            mixing_network: Mixing network instance
            memory: Memory instance
            cfg: Configuration dictionary
            observation_space: Observation space
            action_space: Action space
            device: Device to use for computation ("cpu" or "cuda:X")
        """
        super().__init__(
            possible_agents=[f"agent_{i}" for i in range(len(agents))],
            models={f"agent_{i}": models[i] for i in range(len(agents))},
            device=device,
        )
        
        self.mixing_network = mixing_network
        self.target_mixing_network = mixing_network.__class__(
            observation_space=mixing_network.observation_space,
            action_space=mixing_network.action_space,
            device=device,
            num_agents=mixing_network.num_agents,
            mixing_embed_dim=mixing_network.mixing_embed_dim,
            hypernet_embed=mixing_network.hypernet_embed
        )
        
        # Copy weights from main network to target network
        self.target_mixing_network.load_state_dict(mixing_network.state_dict())
        
        self.memory: QMIXMemory = memory
        self.cfg = cfg if cfg is not None else {}
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Configure optimizer for mixing network
        lr = self.cfg.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.mixing_network.parameters(), lr=lr)
        
        # Set other hyperparameters
        self.gamma = self.cfg.get("gamma", 0.99)
        self.batch_size = self.cfg.get("batch_size", 32)
        self.target_update_frequency = self.cfg.get("target_update_frequency", 100)
        self.num_agents = len(agents)
        self.agents: list[DQN] = agents
        # Training steps counter
        self.training_steps = 0
        
        # Flag to indicate we've checked DQN agent updates
        self.checked_dqn_updates = False
    
    def act(self, states, timestamp=0, timesteps=0):
        """
        Get actions for each agent based on their observations.
        
        Args:
            states: List of states, one for each agent
            
        Returns:
            actions: List of integer actions, one for each agent
        """
        actions = []
        
        # Print debug info on first call
        if not hasattr(self, '_printed_act_shapes'):
            print(f"QMIXAgent.act: states types and shapes:")
            for i, state in enumerate(states):
                print(f"  Agent {i}: type={type(state)}, shape={state.shape if hasattr(state, 'shape') else 'N/A'}")
            self._printed_act_shapes = True
        
        for i, state in enumerate(states):
            # Convert state to tensor if it's not already
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            
            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            # Ensure state has the expected shape
            if i == 0 and not hasattr(self, '_printed_state_shape'):
                print(f"Agent {i} input state shape: {state.shape}")
                self._printed_state_shape = True
            
            # Get action from agent's network
            with torch.no_grad():
                action, _, _ = self.agents[i].act(state, timestamp, timesteps)
                # Convert action to integer and move to CPU
                action = action.item() if isinstance(action, torch.Tensor) else int(action)
                actions.append(action)
        
        return actions
    
    def record_transition(self, states, actions, rewards, next_states, dones, global_state=None, next_global_state=None):
        """
        Record a transition in memory.
        
        Args:
            states: List of agent observations
            actions: List of agent actions
            rewards: List of agent rewards
            next_states: List of next agent observations
            dones: List of done flags
            global_state: Global state
            next_global_state: Next global state
        """
        # Convert individual agent observations into a batch tensor [batch=1, num_agents, obs_dim]
        batch_states = []
        batch_next_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for i in range(len(states)):
            # Convert state to tensor if needed
            if not isinstance(states[i], torch.Tensor):
                state = torch.tensor(states[i], dtype=torch.float32, device=self.device)
            else:
                state = states[i]
            
            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            batch_states.append(state)
            
            # Convert next_state to tensor if needed
            if not isinstance(next_states[i], torch.Tensor):
                next_state = torch.tensor(next_states[i], dtype=torch.float32, device=self.device)
            else:
                next_state = next_states[i]
            
            # Add batch dimension if needed
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
            batch_next_states.append(next_state)
            
            # Convert action to tensor if needed
            if not isinstance(actions[i], torch.Tensor):
                action = torch.tensor(actions[i], dtype=torch.long, device=self.device)
            else:
                action = actions[i]
            if action.dim() == 0:
                action = action.unsqueeze(0)
            batch_actions.append(action)
            
            # Convert reward to tensor if needed
            if not isinstance(rewards[i], torch.Tensor):
                reward = torch.tensor(rewards[i], dtype=torch.float32, device=self.device)
            else:
                reward = rewards[i]
            if reward.dim() == 0:
                reward = reward.unsqueeze(0)
            batch_rewards.append(reward)
            
            # Convert done to tensor if needed
            if not isinstance(dones[i], torch.Tensor):
                done = torch.tensor(dones[i], dtype=torch.bool, device=self.device)
            else:
                done = dones[i]
            if done.dim() == 0:
                done = done.unsqueeze(0)
            batch_dones.append(done)
            
            # Record transition in individual agent's memory
            self.agents[i].memory.add_samples(
                observations=state,
                actions=action,
                rewards=reward,
                next_observations=next_state,
                dones=done
            )
        
        # Stack into batch tensors [batch=1, num_agents, ...]
        states_tensor = torch.cat(batch_states, dim=0).unsqueeze(0)
        next_states_tensor = torch.cat(batch_next_states, dim=0).unsqueeze(0)
        actions_tensor = torch.cat(batch_actions, dim=0).unsqueeze(0)
        rewards_tensor = torch.cat(batch_rewards, dim=0).unsqueeze(0)
        dones_tensor = torch.cat(batch_dones, dim=0).unsqueeze(0)
        
        # Convert global states to tensors if needed
        if global_state is not None and not isinstance(global_state, torch.Tensor):
            global_state = torch.tensor(global_state, dtype=torch.float32, device=self.device)
        
        if next_global_state is not None and not isinstance(next_global_state, torch.Tensor):
            next_global_state = torch.tensor(next_global_state, dtype=torch.float32, device=self.device)
        
        # Add batch dimension to global states if needed
        if global_state is not None and global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        
        if next_global_state is not None and next_global_state.dim() == 1:
            next_global_state = next_global_state.unsqueeze(0)
        
        # Store transition in QMIX memory
        self.memory.add_samples(
            observations=states_tensor,
            actions=actions_tensor,
            rewards=rewards_tensor,
            next_observations=next_states_tensor,
            dones=dones_tensor,
            global_states=global_state,
            next_global_states=next_global_state
        )
    
    def post_interaction(self, current_step, training_steps):
        """
        Update the QMIX agent (mixing network and individual agents).
        
        This method samples from the replay buffer, computes the loss, and updates
        both the individual agent networks and the mixing network.
        
        Returns:
            info: Dictionary with training metrics
        """
        self.training_steps += 1
        
        # Skip update if we don't have enough samples
        if self.memory.filled < self.batch_size:
            return {}
        
        try:
            # Print memory debug info
            if not hasattr(self, '_printed_memory_debug'):
                print("\nMemory Debug Information:")
                print(f"QMIX Memory tensors: {list(self.memory.tensors.keys()) if hasattr(self.memory, 'tensors') else 'Unknown'}")
                print(f"QMIX Memory filled: {self.memory.filled}")
                
                for i, agent in enumerate(self.agents):
                    if hasattr(agent.memory, 'tensors'):
                        print(f"Agent {i} Memory tensors: {list(agent.memory.tensors.keys())}")
                        print(f"Agent {i} Memory filled: {agent.memory.filled}")
                    else:
                        print(f"Agent {i} Memory tensors: Unknown")
                self._printed_memory_debug = True
            
            # Get a list of available tensors in the memory
            if hasattr(self.memory, 'tensors'):
                available_tensors = list(self.memory.tensors.keys()) 
                if hasattr(self.memory, 'global_states'):
                    available_tensors += ["global_states", "next_global_states"]
            else:
                # If we can't determine available tensors, use a standard set
                available_tensors = ["observations", "actions", "rewards", "next_observations", "dones", 
                                    "global_states", "next_global_states"]
            
            # Check which tensors we actually need and are available
            required_names = []
            for name in ["observations", "actions", "rewards", "next_observations", "dones", "global_states", "next_global_states"]:
                if name in available_tensors:
                    required_names.append(name)
                else:
                    print(f"Warning: Tensor '{name}' not available in memory, skipping it in sampling")
            
            if len(required_names) < 5:  # We need at least observations, actions, rewards, next_observations, dones
                print(f"Not enough tensors available: {required_names}")
                return {}
            
            # Sample a batch from memory with only available tensors
            batch = self.memory.sample(names=required_names, batch_size=self.batch_size)[0]  # Get the first (and only) mini-batch
            
            # Extract batch data
            idx = 0
            for name in required_names:
                if name == "observations":
                    states = batch[idx]
                elif name == "actions":
                    actions = batch[idx]
                elif name == "rewards":
                    rewards = batch[idx]
                elif name == "next_observations":
                    next_states = batch[idx]
                elif name == "dones":
                    dones = batch[idx]
                elif name == "global_states":
                    global_states = batch[idx]
                elif name == "next_global_states":
                    next_global_states = batch[idx]
                idx += 1
            
            # Check that we have all required tensors
            for required_var in ["states", "actions", "rewards", "next_states", "dones"]:
                if required_var not in locals():
                    print(f"Error: Required tensor '{required_var}' not found in sampled batch")
                    return {}
            
            if "global_states" not in locals() or "next_global_states" not in locals():
                print("Error: Missing global states in sampled batch")
                return {}
            
            # Get Q-values for each agent for current observations
            agent_q_values = []
            target_agent_q_values = []
            chosen_actions_q_values = []
            max_next_q_values = []
            
            for i, agent in enumerate(self.agents):
                # Get Q-values from main network
                # Create input dictionary for the Q-network
                q_inputs = {"states": states[:, i]}  # [batch_size, state_dim]
                q_values = agent.models["q_network"](q_inputs)[0]  # [batch_size, action_dim]
                agent_q_values.append(q_values)
                
                # Get chosen action Q-values
                batch_indices = torch.arange(self.batch_size, device=self.device)
                chosen_actions_q = q_values[batch_indices, actions[:, i].long()]  # [batch_size]
                chosen_actions_q_values.append(chosen_actions_q.unsqueeze(1))  # [batch_size, 1]
                
                # Get target Q-values for next state
                target_q_inputs = {"states": next_states[:, i]}  # [batch_size, state_dim]
                target_q_values = agent.models["target_q_network"](target_q_inputs)[0]  # [batch_size, action_dim]
                max_next_q = target_q_values.max(dim=1)[0]  # [batch_size]
                max_next_q_values.append(max_next_q.unsqueeze(1))  # [batch_size, 1]
            
            # Stack Q-values
            chosen_actions_q_values = torch.cat(chosen_actions_q_values, dim=1)  # [batch_size, num_agents]
            max_next_q_values = torch.cat(max_next_q_values, dim=1)  # [batch_size, num_agents]
            
            # Calculate mixed Q-value
            mixed_q_values = self.mixing_network(chosen_actions_q_values, global_states)  # [batch_size, 1]
            
            # Calculate target mixed Q-value
            target_mixed_q_values = self.target_mixing_network(max_next_q_values, next_global_states)  # [batch_size, 1]
            
            # Calculate targets
            rewards_sum = rewards.sum(dim=1, keepdim=True)  # [batch_size, 1]
            dones_any = dones.any(dim=1, keepdim=True)  # [batch_size, 1]
            targets = rewards_sum + self.gamma * (1 - dones_any.float()) * target_mixed_q_values  # [batch_size, 1]
            
            # Calculate loss
            loss = F.mse_loss(mixed_q_values, targets.detach())
            
            # Optimize mixing network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update individual agents
            agent_info = {}
            for i, agent in enumerate(self.agents):
                # Skip update if the agent doesn't have enough samples
                if not hasattr(agent.memory, 'filled') or agent.memory.filled < self.batch_size:
                    agent_info[f"agent_{i}"] = {"skipped": True}
                    continue
                
                # Important: Each agent.update() call will:
                # 1. Sample from the agent's memory (separate from the QMIX memory)
                # 2. Compute the individual agent's loss (DQN loss)
                # 3. Perform backpropagation through the agent's Q-network
                # 4. Update the agent's Q-network parameters using the agent's optimizer
                try:
                    agent_update_info = agent.post_interaction(current_step, training_steps)
                    agent_info[f"agent_{i}"] = agent_update_info
                    
                    # Print info about the first agent's update on the first training step
                    if not self.checked_dqn_updates and i == 0 and self.training_steps == 1:
                        print("\nDQN Agent Update Details:")
                        print("  Each agent's Q-network parameters are updated independently through that agent's optimizer")
                        print("  Updates happen in two separate steps:")
                        print("    1. The mixing network is updated to better combine individual agent Q-values")
                        print("    2. Each agent's Q-network is updated independently to learn better individual policies")
                        print("  This dual update approach allows agents to learn both individual and coordinated strategies")
                        self.checked_dqn_updates = True
                except Exception as e:
                    print(f"Error updating agent {i}: {str(e)}")
                    agent_info[f"agent_{i}"] = {"error": str(e)}
            
            # Update target mixing network
            if self.training_steps % self.target_update_frequency == 0:
                # Copy weights from main networks to target networks
                self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
                
                # Update target networks for all individual agents
                for agent in self.agents:
                    agent.models["target_q_network"].load_state_dict(agent.models["q_network"].state_dict())
            
            # Return update info
            info = {
                "loss": loss.item(),
                "agents": agent_info
            }
            
            return info
            
        except Exception as e:
            print(f"Error during update: {e}")
            raise e
            return {}

    def update_target_networks(self):
        """Update all agent target networks"""
        for i in range(self.num_agents):
            self.agents[i].models["q_network"].update_target_network()


# Custom Memory for QMIX
class QMIXMemory(RandomMemory):
    """
    Memory class for QMIX algorithm that stores global states along with agent observations.
    Uses SKRL's tensor management system properly for optimal performance.
    """
    def __init__(self, memory_size, num_agents, device="cpu"):
        """
        Initialize the QMIX memory.
        
        Args:
            memory_size: Maximum size of the replay buffer
            num_agents: Number of agents in the environment
            device: Device to use for computation ("cpu" or "cuda:X")
        """
        # Initialize parent class
        super().__init__(memory_size=memory_size, device=device)
        self.num_agents = num_agents
        
        print(f"Initializing QMIXMemory for {num_agents} agents with size {memory_size}")
        
        # We will directly add global states as tensors in the RandomMemory storage
        # This approach keeps everything in one place and leverages SKRL's memory management
        # The standard tensors (observations, actions, rewards, next_observations, dones)
        # are already handled by RandomMemory
    
    def add_samples(self, observations=None, actions=None, rewards=None, next_observations=None, 
                   dones=None, global_states=None, next_global_states=None):
        """
        Add samples to the replay buffer, including global states.
        
        Args:
            observations: Agent observations [batch_size, num_agents, obs_dim]
            actions: Agent actions [batch_size, num_agents]
            rewards: Agent rewards [batch_size, num_agents] 
            next_observations: Next agent observations [batch_size, num_agents, obs_dim]
            dones: Done flags [batch_size, num_agents]
            global_states: Global state [batch_size, global_state_dim]
            next_global_states: Next global state [batch_size, global_state_dim]
        """
        # First, we need to ensure global_states tensors exist in memory
        if "global_states" not in self.tensors and global_states is not None:
            tensor_shape = (self.memory_size,) + tuple(global_states.shape[1:])
            self.tensors["global_states"] = torch.zeros(tensor_shape, dtype=torch.float32, device=self.device)
            print(f"Created 'global_states' tensor with shape {tensor_shape}")
            
        if "next_global_states" not in self.tensors and next_global_states is not None:
            tensor_shape = (self.memory_size,) + tuple(next_global_states.shape[1:])
            self.tensors["next_global_states"] = torch.zeros(tensor_shape, dtype=torch.float32, device=self.device)
            print(f"Created 'next_global_states' tensor with shape {tensor_shape}")
        
        # Next, we need to determine batch size
        batch_size = 0
        if observations is not None:
            batch_size = observations.shape[0]
        elif actions is not None:
            batch_size = actions.shape[0]
        elif rewards is not None:
            batch_size = rewards.shape[0]
        elif global_states is not None:
            batch_size = global_states.shape[0]
        
        if batch_size == 0:
            return
            
        # Calculate indices for this insertion (circular buffer)
        indices = torch.arange(self.memory_index, self.memory_index + batch_size) % self.memory_size
        
        # Store all tensors directly in self.tensors
        if observations is not None:
            # Handle observations (potentially)
            self._maybe_create_tensor("observations", observations, batch_size)
            self.tensors["observations"][indices] = observations
            
        if actions is not None:
            self._maybe_create_tensor("actions", actions, batch_size)
            self.tensors["actions"][indices] = actions
            
        if rewards is not None:
            self._maybe_create_tensor("rewards", rewards, batch_size)
            self.tensors["rewards"][indices] = rewards
            
        if next_observations is not None:
            self._maybe_create_tensor("next_observations", next_observations, batch_size)
            self.tensors["next_observations"][indices] = next_observations
            
        if dones is not None:
            self._maybe_create_tensor("dones", dones, batch_size)
            self.tensors["dones"][indices] = dones
            
        if global_states is not None and "global_states" in self.tensors:
            self.tensors["global_states"][indices] = global_states
            
        if next_global_states is not None and "next_global_states" in self.tensors:
            self.tensors["next_global_states"][indices] = next_global_states
        
        # Update indices
        prev_index = self.memory_index
        self.memory_index = (self.memory_index + batch_size) % self.memory_size
        
        # Update filled size
        if self.memory_index <= prev_index:  # We've wrapped around
            self.filled = self.memory_size
        else:
            self.filled = max(self.filled, self.memory_index)
    
    def _maybe_create_tensor(self, name, tensor, batch_size):
        """Create a tensor in memory if it doesn't already exist."""
        if name not in self.tensors:
            shape = (self.memory_size,) + tuple(tensor.shape[1:])
            self.tensors[name] = torch.zeros(shape, dtype=tensor.dtype, device=self.device)
            print(f"Created '{name}' tensor with shape {shape}")
        
    def sample(self, names, batch_size, mini_batches=1):
        """
        Sample a batch of experiences from memory.
        
        Args:
            names: Tuple of tensor names to sample
            batch_size: Number of samples to retrieve
            mini_batches: Number of mini-batches
            
        Returns:
            List of sampled tensors
        """
        if not hasattr(self, '_printed_tensors'):
            print(f"QMIXMemory available tensors: {list(self.tensors.keys())}")
            print(f"QMIXMemory filled: {self.filled}/{self.memory_size}")
            self._printed_tensors = True
            
        # Check if we have enough samples
        if self.filled < batch_size:
            raise RuntimeError(f"Not enough samples in memory. Have {self.filled}, requested {batch_size}")
            
        # Generate random indices within the filled portion of memory
        indices = torch.randint(0, self.filled, (batch_size,), device=self.device)
        
        # Sample tensors
        tensors = []
        for name in names:
            if name in self.tensors:
                tensors.append(self.tensors[name][indices])
            else:
                raise KeyError(f"Tensor '{name}' not found in memory. Available tensors: {list(self.tensors.keys())}")
                
        # Return as a list of lists (for mini-batches compatibility)
        return [tensors]


# Replace the existing QNetwork class with the one from dqn.py
class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_dim, action_dim):
        Model.__init__(self, observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,)), 
                      action_space=gym.spaces.Discrete(action_dim))
        DeterministicMixin.__init__(self, clip_actions=False)
        
        # Network architecture
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Store target network parameters
        self.target_layers = copy.deepcopy(self.layers)
        for param in self.target_layers.parameters():
            param.requires_grad = False
    
    def compute(self, inputs, role="q_network"):
        """
        Compute Q-values for given states.
        
        Args:
            inputs: Dictionary containing "states" tensor of shape [batch_size, state_dim]
            role: Which network to use ("q_network" or "target_q_network")
            
        Returns:
            q_values: Tensor of shape [batch_size, action_dim]
            info: Empty dictionary
        """
        # Get states from inputs dictionary and ensure it's float32
        states = inputs["states"].float()
        
        # Print debug info on first forward pass
        if not hasattr(self, '_printed_shapes'):
            print(f"QNetwork compute: states shape={states.shape}")
            self._printed_shapes = True
        
        # Forward pass through appropriate network
        if role == "target_q_network":
            return self.target_layers(states), {}
        return self.layers(states), {}
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_layers.load_state_dict(self.layers.state_dict())


# Add metrics logging functionality
def log_metrics(metrics, episode, log_dir=None):
    """
    Log metrics about the maze solving process.
    
    Args:
        metrics: Dictionary containing metrics to log
        episode: Current episode number
        log_dir: Directory to save metrics (if None, just print)
    """
    # Print current metrics
    print(f"\nEpisode {episode} Metrics:")
    print(f"  Total steps: {metrics.get('total_steps', 'N/A')}")
    print(f"  Total revisits: {metrics.get('total_revisits', 'N/A')}")
    print(f"  Unique states visited: {metrics.get('unique_states_visited', 'N/A')}")
    print(f"  Exploration percentage: {metrics.get('exploration_percentage', 'N/A'):.2f}%")
    
    if metrics.get('maze_solved', False):
        print(f"  Maze solved in {metrics.get('steps_to_solve', 'N/A')} steps")
    else:
        print("  Maze not solved yet")
    
    # Save metrics to file if log_dir is provided
    if log_dir is not None:
        import os
        import json
        from datetime import datetime
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"metrics_{timestamp}.json")
        
        # Add episode number to metrics
        metrics_with_episode = metrics.copy()
        metrics_with_episode['episode'] = episode
        
        # Write metrics to file
        with open(filename, 'w') as f:
            json.dump(metrics_with_episode, f)


def get_global_state(info, maze_size, global_state_dim):
    """Helper function to get global state from environment info."""
    if 'global_state' in info:
        return info['global_state']
    elif hasattr(info.get('maze_env', {}), '_get_full_obs'):
        # Use the full observation as a global state
        full_obs = info['maze_env']._get_full_obs()
        return full_obs.flatten().astype(np.float32)
    elif hasattr(info.get('env', {}), '_get_full_obs'):
        # Try another common attribute
        full_obs = info['env']._get_full_obs()
        return full_obs.flatten().astype(np.float32)
    else:
        # Fallback to a zero vector
        return np.zeros(global_state_dim, dtype=np.float32)


def train_qmix(
    maze_size=15,
    num_agents=2,
    memory_size=1000000, 
    learning_rate=1e-3,
    training_steps=10000,
    batch_size=256,
    gamma=0.99,
    target_update_frequency=10,
    agent_update_frequency=1,
    tau=1.0,
    initial_epsilon=1.0,
    final_epsilon=0.02,
    epsilon_steps=1000,
    exploration_percentage_threshold=90.0,
    log_every=10,
    seed=42,
    experiment_name=None,
    learning_starts=1000
):
    """
    Train a QMIX agent to solve a maze.
    
    Args:
        maze_size: Integer specifying the size of the maze
        num_agents: Number of agents to use
        memory_size: Size of the replay buffer
        learning_rate: Learning rate for optimizer
        training_steps: Total number of training steps
        batch_size: Batch size for training
        gamma: Discount factor
        target_update_frequency: How often to update target network
        agent_update_frequency: How often to update agent networks
        tau: Interpolation parameter for target network update
        initial_epsilon: Initial exploration rate
        final_epsilon: Final exploration rate
        epsilon_steps: Number of steps to decay epsilon over
        exploration_percentage_threshold: Stop training if exploration percentage exceeds this value
        log_every: How often to log metrics (in episodes)
        seed: Random seed
        experiment_name: Name for the experiment (for logging)
        learning_starts: Number of steps before starting to learn
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create experiment name with timestamp if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"qmix_maze_{maze_size}x{maze_size}_agents{num_agents}_{timestamp}"
    
    # Create logging directory
    log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting QMIX training for {training_steps} steps")
    print(f"Experiment: {experiment_name}")
    print(f"Log directory: {log_dir}")
    
    # Create multi-agent maze environment
    env = MazeEnv(
        maze_generator=RandomBlockMazeGenerator(
            maze_size=maze_size,
            obstacle_ratio=0.1,
        ),
        pob_size=1,  # Default partial observation size
        action_type='VonNeumann',  # Default action type
        obs_type='full',  # Use full observations
        live_display=False,  # Don't show live display during training
        render_trace=True  # Track agent paths
    )
    
    # Reset the environment to get initial observation shapes
    observations, info = env.reset(num_agents=num_agents, seed=seed)
    
    # Get observation dimensions
    obs_dim = observations[0].shape[0]  # Get the dimension of a single agent's observation
    action_dim = env.action_space.n
    
    print(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # Create individual agent models with correct input dimensions
    agent_models = []
    for i in range(num_agents):
        models = {}
        models["q_network"] = QNetwork(
            observation_dim=obs_dim, 
            action_dim=action_dim
        )
        models["target_q_network"] = QNetwork(
            observation_dim=obs_dim, 
            action_dim=action_dim
        )
        # Copy target network weights from main network
        models["target_q_network"].load_state_dict(models["q_network"].state_dict())
        agent_models.append(models)
    
    # Determine global state dimension
    if hasattr(env.unwrapped, '_get_full_obs'):
        # Use the full observation as a global state
        full_obs = env.unwrapped._get_full_obs()
        global_state_dim = full_obs.flatten().shape[0]
    else:
        # Fallback to a reasonable size if no global state is available
        global_state_dim = maze_size * maze_size * 3
    
    print(f"Global state dimension: {global_state_dim}")
    
    # Create mixing network with correct global state dimension
    mixing_network = MixingNetwork(
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(global_state_dim,)),
        action_space=env.action_space,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_agents=num_agents,
        mixing_embed_dim=32,
        hypernet_embed=64
    )
    
    # Create replay memory for experience replay
    memory = QMIXMemory(
        memory_size=memory_size,
        num_agents=num_agents,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # Create individual DQN agents
    dqn_agents = []
    for i in range(num_agents):
        agent_memory = RandomMemory(memory_size=memory_size, device="cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Explicitly initialize the memory tensors
        dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device="cuda:0" if torch.cuda.is_available() else "cpu")
        dummy_action = torch.zeros((1,), dtype=torch.long, device="cuda:0" if torch.cuda.is_available() else "cpu")
        dummy_reward = torch.zeros((1,), dtype=torch.float32, device="cuda:0" if torch.cuda.is_available() else "cpu")
        dummy_done = torch.zeros((1,), dtype=torch.bool, device="cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize the tensors with a dummy sample
        agent_memory.add_samples(
            observations=dummy_obs,
            actions=dummy_action,
            rewards=dummy_reward,
            next_observations=dummy_obs.clone(),
            dones=dummy_done
        )
        
        dqn_agent = DQN(
            models=agent_models[i],
            memory=agent_memory,
            cfg={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "target_update_frequency": target_update_frequency,
                "exploration": {
                    "initial_epsilon": initial_epsilon,
                    "final_epsilon": final_epsilon,
                    "timesteps": epsilon_steps
                }
            },
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
            action_space=env.action_space,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        # Set required attributes for DQN agent
        dqn_agent.tensors_names = ["observations", "actions", "rewards", "next_observations", "dones"]
        dqn_agent._batch_size = batch_size
        dqn_agent._learning_rate = learning_rate
        dqn_agent._gamma = gamma
        dqn_agent._target_update_frequency = target_update_frequency
        dqn_agent._exploration = {
            "initial_epsilon": initial_epsilon,
            "final_epsilon": final_epsilon,
            "timesteps": epsilon_steps
        }
        dqn_agents.append(dqn_agent)
    
    # Create QMIX agent
    agent = QMIXAgent(
        agents=dqn_agents,
        models=agent_models,
        mixing_network=mixing_network,
        memory=memory,
        cfg={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "target_update_frequency": target_update_frequency,
            "tau": tau
        },
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
        action_space=env.action_space,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # Start training loop
    episode = 0
    current_step = 0
    start_time = time.time()
    
    # Variables to track exploration metrics
    total_solved = 0
    total_episodes = 0
    
    # Main training loop
    while current_step < training_steps:
        # Reset environment at the start of each episode
        observations, info = env.reset(num_agents=num_agents, seed=seed)
        
        # Get initial global state
        global_state = get_global_state(info, maze_size, global_state_dim)
        
        # Convert to tensor if needed
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.tensor(global_state, device=agent.device, dtype=torch.float32)
        
        # Initialize tracking metrics for this episode
        episode_step = 0
        episode_reward = 0
        episode_done = False
        last_actions = []
        
        # Safety counter to prevent infinite loops
        max_episode_safety_limit = 10000
        
        # Run episode until it's done (maze is solved) or safety limit reached
        while not episode_done and episode_step < max_episode_safety_limit:
            # Get actions from agent
            actions = agent.act(observations, current_step, training_steps)
            last_actions = actions.copy()  # Store for logging
            
            # Take a step in the environment
            next_observations, rewards, dones, truncated, info = env.step(actions)
            
            # Log detailed reward information
            if episode_step % 1000 == 0 or sum(rewards) > 0:
                print(f"  Step {episode_step}, Rewards: {rewards}, Actions: {actions}")
                print(f"  Current exploration: {info.get('exploration_percentage', 0):.2f}% ({info.get('unique_states_visited', 0)} states)")
            
            # Get next global state
            next_global_state = get_global_state(info, maze_size, global_state_dim)
            
            # Convert to tensor if needed
            if not isinstance(next_global_state, torch.Tensor):
                next_global_state = torch.tensor(next_global_state, device=agent.device, dtype=torch.float32)
            
            # Record transition with global states
            agent.record_transition(
                states=observations,
                actions=actions,
                rewards=rewards,
                next_states=next_observations,
                dones=dones,
                global_state=global_state,
                next_global_state=next_global_state
            )
            
            # Update the state and global state
            observations = next_observations
            global_state = next_global_state
            
            # Update training metrics
            episode_step += 1
            current_step += 1
            episode_reward += sum(rewards)
            episode_done = any(dones) or any(truncated)
            
            # Only update the agent if we've collected enough samples
            if current_step >= learning_starts and current_step % agent_update_frequency == 0 and memory.filled >= batch_size:
                agent.post_interaction(current_step, training_steps)
            
            # Check if we've reached the training steps limit
            if current_step >= training_steps:
                break
            
            # Periodically log progress during episode
            if current_step % 1000 == 0:
                print(f"Step {current_step}/{training_steps} - Episode {episode+1} - Step {episode_step}")
                if current_step < learning_starts:
                    print(f"  Collecting initial samples: {current_step}/{learning_starts}")
                else:
                    print(f"  Training: {current_step - learning_starts}/{training_steps - learning_starts}")
                
                # Get tracking metrics from environment
                if 'total_steps' in info:
                    exploration_pct = info.get('exploration_percentage', 0)
                    print(f"  Exploration: {exploration_pct:.2f}% ({info.get('unique_states_visited', 0)} states)")
                    print(f"  Revisits: {info.get('total_revisits', 0)}")
                    if info.get('maze_solved', False):
                        print(f"  Maze solved in {info.get('steps_to_solve', 'N/A')} steps")
                        
                # Log memory stats
                print(f"  Memory filled: {memory.filled}/{memory.memory_size}")
                
                # Log recent actions and rewards
                print(f"  Recent actions: {last_actions}")
                print(f"  Current episode reward: {episode_reward:.2f}")
        
        # Check if episode ended due to safety limit
        if episode_step >= max_episode_safety_limit:
            print(f"\nWARNING: Episode {episode+1} terminated after {episode_step} steps (safety limit)")
            print(f"  Final exploration: {info.get('exploration_percentage', 0):.2f}% ({info.get('unique_states_visited', 0)} states)")
            print(f"  Final actions: {last_actions}")
            
        # Episode completed - capture metrics
        total_episodes += 1
        
        # Extract metrics from the environment
        maze_solved = info.get('maze_solved', False)
        if maze_solved:
            total_solved += 1
            
        metrics = {
            'total_steps': info.get('total_steps', episode_step),
            'total_revisits': info.get('total_revisits', 0),
            'unique_states_visited': info.get('unique_states_visited', 0),
            'exploration_percentage': info.get('exploration_percentage', 0),
            'maze_solved': maze_solved,
            'steps_to_solve': info.get('steps_to_solve', -1) if maze_solved else -1,
            'episode_reward': episode_reward,
            'success_rate': (total_solved / total_episodes) * 100
        }
        
        # Log metrics every N episodes
        if episode % log_every == 0:
            elapsed_time = time.time() - start_time
            
            # Print metrics
            print(f"\nEpisode {episode+1} completed after {episode_step} steps")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Exploration: {metrics['exploration_percentage']:.2f}% ({metrics['unique_states_visited']} states)")
            print(f"  Success rate: {metrics['success_rate']:.2f}%")
            print(f"  Time elapsed: {elapsed_time:.2f}s")
            print(f"  Memory filled: {memory.filled}/{memory.memory_size}")
            
            # Write metrics to CSV
            metrics_file = os.path.join(log_dir, "metrics.csv")
            write_header = not os.path.exists(metrics_file)
            with open(metrics_file, "a") as f:
                if write_header:
                    f.write("episode,total_steps,total_revisits,unique_states_visited,exploration_percentage,maze_solved,steps_to_solve,time_elapsed\n")
                f.write(f"{episode+1},{metrics['total_steps']},{metrics['total_revisits']},{metrics['unique_states_visited']},{metrics['exploration_percentage']},{1 if metrics['maze_solved'] else 0},{metrics['steps_to_solve']},{elapsed_time:.2f}\n")
        
        # Check if we've reached the exploration threshold
        if metrics['exploration_percentage'] > exploration_percentage_threshold:
            print(f"\nReached exploration threshold of {exploration_percentage_threshold}%!")
            print(f"Explored {metrics['unique_states_visited']} states out of {maze_size * maze_size} total")
            print(f"Training completed after {current_step} steps and {episode+1} episodes")
            break
        
        # Increment episode counter
        episode += 1
    
    # Save the trained agent
    model_path = os.path.join(log_dir, "qmix_agent.pt")
    agent.save(model_path)
    print(f"Agent saved to {model_path}")
    
    # Final metrics
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed:")
    print(f"  Total steps: {current_step}")
    print(f"  Total episodes: {episode}")
    print(f"  Success rate: {(total_solved / total_episodes) * 100:.2f}%")
    print(f"  Final exploration: {metrics['exploration_percentage']:.2f}%")
    print(f"  Time elapsed: {elapsed_time:.2f}s")
    
    return agent


def test_qmix(agent=None, maze_size=10, n_agents=2, obstacle_ratio=0.1, n_episodes=5, 
              device="cpu", path=None, render=True, obs_type="full", pob_size=1):
    """
    Test QMIX agent on the maze environment.
    
    Args:
        agent: Trained QMIX agent (if None and path is provided, will load from file)
        maze_size: Size of the maze
        n_agents: Number of agents
        obstacle_ratio: Ratio of obstacles in the maze
        n_episodes: Number of episodes to test
        device: Device to use for testing ("cpu" or "cuda:X")
        path: Path to load a pre-trained agent from
        render: Whether to render the environment
        obs_type: Observation type ("full" or "partial")
        pob_size: Size of partial observation window
        
    Returns:
        scores: List of rewards per episode
    """
    # Create maze generator
    maze_generator = RandomBlockMazeGenerator(maze_size=maze_size, obstacle_ratio=obstacle_ratio)
    
    # Create environment with multi-agent support
    env = MazeEnv(
        maze_generator=maze_generator,
        pob_size=pob_size,
        action_type='VonNeumann',  # Default action type
        obs_type=obs_type,
        live_display=render,  # Show live display during testing if render=True
        render_trace=True  # Track agent paths
    )
    
    # Reset to setup environment with correct number of agents
    observations, info = env.reset(num_agents=n_agents)
    
    # Get observation and action dimensions
    obs_dim = observations[0].shape[0]  # Get a single agent's observation dimension
    action_dim = env.action_space.n
    
    print(f"Testing with observation dimension: {obs_dim}, action dimension: {action_dim}")
    
    # Determine global state dimension
    if hasattr(env.unwrapped, '_get_full_obs'):
        # Use the full observation as a global state
        full_obs = env.unwrapped._get_full_obs()
        global_state_dim = full_obs.flatten().shape[0]
    else:
        # Fallback to a reasonable size
        global_state_dim = maze_size * maze_size * 3
        
    print(f"Global state dimension: {global_state_dim}")
    
    # Load agent if not provided but path is
    if agent is None and path is not None:
        # Create individual agent models with correct dimensions
        agent_models = []
        for i in range(n_agents):
            models = {}
            models["q_network"] = QNetwork(
                observation_dim=obs_dim, 
                action_dim=action_dim
            )
            models["target_q_network"] = QNetwork(
                observation_dim=obs_dim, 
                action_dim=action_dim
            )
            agent_models.append(models)
        
        # Create mixing network with correct global state dimension
        mixing_network = MixingNetwork(
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(global_state_dim,)),
            action_space=env.action_space,
            device=device,
            num_agents=n_agents,
            mixing_embed_dim=32,
            hypernet_embed=64
        )
        
        # Create memory
        memory = QMIXMemory(memory_size=1000, num_agents=n_agents, device=device)
        
        # Create individual DQN agents
        dqn_agents = []
        for i in range(n_agents):
            agent_memory = RandomMemory(memory_size=1000, device=device)
            dqn_agent = DQN(
                models=agent_models[i],
                memory=agent_memory,
                cfg={"exploration": {"initial_epsilon": 0.0, "final_epsilon": 0.0}},
                observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
                action_space=env.action_space,
                device=device
            )
            dqn_agents.append(dqn_agent)
        
        # Create QMIX agent
        agent = QMIXAgent(
            agents=dqn_agents,
            models=agent_models,
            mixing_network=mixing_network,
            memory=memory,
            cfg={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "target_update_frequency": 100
            },
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
            action_space=env.action_space,
            device=device
        )
        
        # Load weights
        agent.load(path)
    
    # Testing loop
    scores = []
    success_count = 0
    all_metrics = []
    
    for episode in range(1, n_episodes + 1):
        observations, info = env.reset(num_agents=n_agents)
        total_reward = 0
        steps = 0
        
        # Get initial global state
        current_global_state = get_global_state(info, maze_size, global_state_dim)
        
        # Continue episode until maze is solved
        while True:
            # Select actions (no exploration)
            actions = agent.act(observations, steps, steps)
            
            # Take actions in the environment
            next_observations, rewards, dones, truncated, info = env.step(actions)
            
            # Get next global state
            next_global_state = get_global_state(info, maze_size, global_state_dim)
            
            # For visualization only - we don't actually need to record transitions during testing
            if current_global_state is not None and next_global_state is not None:
                current_global_state_tensor = torch.tensor(current_global_state, 
                                                          device=agent.device)
                next_global_state_tensor = torch.tensor(next_global_state, 
                                                       device=agent.device)
                
                # Just for visualization purposes
                agent.record_transition(
                    states=observations,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_observations,
                    dones=dones,
                    global_state=current_global_state_tensor,
                    next_global_state=next_global_state_tensor
                )
            
            # Update observations and global state for the next step
            observations = next_observations
            current_global_state = next_global_state
            
            # Sum rewards
            total_reward += sum(rewards)
            steps += 1
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print(f"  Episode {episode}, Step {steps}: Reward so far: {total_reward:.2f}")
                print(f"  Exploration: {info.get('exploration_percentage', 0):.2f}% ({info.get('unique_states_visited', 0)} states)")
                print(f"  Recent actions: {actions}")
            
            # Check if any agent reached the goal
            if any(dones) and not any(truncated):
                success_count += 1
            
            # End episode if any agent is done
            if any(dones) or any(truncated):
                break
            
            # Safety check - terminate extremely long episodes after 10,000 steps
            if steps >= 10000:
                print(f"  Terminating episode {episode} after {steps} steps (safety limit)")
                break
        
        # Collect metrics for this episode
        episode_metrics = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'success': any(dones) and not any(truncated),
            'total_steps': info.get('total_steps', steps),
            'total_revisits': info.get('total_revisits', 0),
            'unique_states_visited': info.get('unique_states_visited', 0),
            'exploration_percentage': info.get('exploration_percentage', 0),
            'maze_solved': info.get('maze_solved', any(dones) and not any(truncated)),
            'steps_to_solve': info.get('steps_to_solve', steps if any(dones) and not any(truncated) else -1)
        }
        all_metrics.append(episode_metrics)
        
        # Log metrics
        print(f"\nTest Episode {episode}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Success: {any(dones) and not any(truncated)}")
        print(f"  Total revisits: {info.get('total_revisits', 'N/A')}")
        print(f"  Unique states visited: {info.get('unique_states_visited', 'N/A')}")
        print(f"  Exploration percentage: {info.get('exploration_percentage', 'N/A'):.2f}%")
        
        if info.get('maze_solved', False):
            print(f"  Maze solved in {info.get('steps_to_solve', steps)} steps")
        
        scores.append(total_reward)
    
    # Print summary statistics
    print(f"\nTest Results after {n_episodes} episodes:")
    print(f"  Average Reward: {np.mean(scores):.2f}")
    print(f"  Success Rate: {success_count/n_episodes*100:.2f}%")
    
    # Average metrics across all episodes
    if all_metrics:
        avg_steps = np.mean([m['steps'] for m in all_metrics])
        avg_revisits = np.mean([m.get('total_revisits', 0) for m in all_metrics])
        avg_unique_states = np.mean([m.get('unique_states_visited', 0) for m in all_metrics])
        
        print(f"  Average Steps: {avg_steps:.2f}")
        print(f"  Average Revisits: {avg_revisits:.2f}")
        print(f"  Average Unique States Visited: {avg_unique_states:.2f}")
        
        # Calculate average steps to solve for successful episodes only
        successful_metrics = [m for m in all_metrics if m.get('success', False)]
        if successful_metrics:
            avg_steps_to_solve = np.mean([m.get('steps_to_solve', m['steps']) for m in successful_metrics])
            print(f"  Average Steps to Solve (successful episodes): {avg_steps_to_solve:.2f}")
    
    return scores


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train and test QMIX agent on multi-agent maze environments")
    
    # General parameters
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training/testing (cuda:0, cuda:1, cpu, etc.)")
    parser.add_argument("--log_dir", type=str, default="/home/mila/m/munjuluv/scratch/logs/qmix_maze_logs",
                        help="Directory to save logs and models")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment run (default: auto-generated with timestamp)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Environment parameters
    parser.add_argument("--maze_size", type=int, nargs=2, default=[15, 15],
                        help="Size of the maze (width, height)")
    parser.add_argument("--num_agents", type=int, default=2,
                        help="Number of agents in the environment")
    parser.add_argument("--obstacle_ratio", type=float, default=0.1,
                        help="Ratio of obstacles in the maze")
    parser.add_argument("--obs_type", type=str, default="full", choices=["full", "partial"],
                        help="Observation type (full or partial)")
    parser.add_argument("--pob_size", type=int, default=1,
                        help="Size of partial observation window (if using partial observations)")
    
    # Training parameters
    parser.add_argument("--train", action="store_true", default=True,
                        help="Run training (default: True)")
    parser.add_argument("--training_steps", type=int, default=10000,
                        help="Total number of training steps")
    parser.add_argument("--n_episodes", type=int, default=500,
                        help="Maximum number of episodes to train")
    parser.add_argument("--memory_size", type=int, default=1000000,
                        help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--agent_update_frequency", type=int, default=1,
                        help="How often to update agent networks")
    parser.add_argument("--target_update_frequency", type=int, default=100,
                        help="How often to update target network")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Interpolation parameter for target network update")
    parser.add_argument("--initial_epsilon", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--final_epsilon", type=float, default=0.02,
                        help="Final exploration rate")
    parser.add_argument("--epsilon_steps", type=int, default=1000,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("--exploration_threshold", type=float, default=100.0,
                        help="Stop training if exploration percentage exceeds this value")
    parser.add_argument("--log_every", type=int, default=10,
                        help="How often to log metrics (in episodes)")
    parser.add_argument("--learning_starts", type=int, default=1000,
                        help="Number of steps to collect before starting learning")
    
    # Testing parameters
    parser.add_argument("--test", action="store_true", default=True,
                        help="Run testing after training (default: True)")
    parser.add_argument("--test_episodes", type=int, default=5,
                        help="Number of episodes to test")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained model (if not training)")
    parser.add_argument("--render", action="store_true", default=True,
                        help="Render environment during testing")
    
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Log directory: {args.log_dir}")
    
    # Create maze size as int
    maze_size = int(args.maze_size[0])
    
    # Print environment settings
    print("\nUsing built-in multi-agent support from gym_maze")
    print(f"Training with {args.num_agents} agents in a {maze_size}x{maze_size} maze")
    
    # Initialize agent
    agent = None
    
    # Train QMIX agent if specified
    if args.train:
        print("\nTraining QMIX agent...")
        agent = train_qmix(
            maze_size=maze_size,
            num_agents=args.num_agents,
            memory_size=args.memory_size,
            learning_rate=args.learning_rate,
            training_steps=args.training_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            target_update_frequency=args.target_update_frequency,
            agent_update_frequency=args.agent_update_frequency,
            tau=args.tau,
            initial_epsilon=args.initial_epsilon,
            final_epsilon=args.final_epsilon,
            epsilon_steps=args.epsilon_steps,
            exploration_percentage_threshold=args.exploration_threshold,
            log_every=args.log_every,
            seed=args.seed,
            experiment_name=args.experiment_name
        )
    
    # Test QMIX agent if specified
    if args.test:
        print("\nTesting QMIX agent...")
        # If agent is None and model_path is provided, it will be loaded in test_qmix
        test_qmix(
            agent=agent,
            maze_size=maze_size,
            n_agents=args.num_agents,
            obstacle_ratio=args.obstacle_ratio,
            n_episodes=args.test_episodes,
            device=args.device,
            path=args.model_path,  # This will be used only if agent is None
            render=args.render,
            obs_type=args.obs_type,
            pob_size=args.pob_size
        )
