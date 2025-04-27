import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from datetime import datetime
from skrl.agents.torch.dqn.dqn import DQN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
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
        super().__init__(observation_space, action_space, device, clip_actions)
        
        self.num_agents = num_agents
        self.state_dim = observation_space.shape[0]
        self.mixing_embed_dim = mixing_embed_dim
        self.hypernet_embed = hypernet_embed
        
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
    def __init__(self, agents, mixing_network, memory, 
                 cfg=None, observation_space=None, action_space=None, device="cpu"):
        """
        Initialize the QMIX agent.
        
        QMIX is a multi-agent reinforcement learning algorithm that learns a centralized 
        value function that factors into a per-agent value function to allow decentralized execution.
        
        Args:
            agents: List of DQN instances
            mixing_network: Mixing network instance
            memory: Memory instance
            cfg: Configuration dictionary
            observation_space: Observation space
            action_space: Action space
            device: Device to use for computation ("cpu" or "cuda:X")
        """
        super().__init__(agents=agents, device=device)
        
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
        
        self.memory = memory
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
        
        # Training steps counter
        self.training_steps = 0
        
        # Flag to indicate we've checked DQN agent updates
        self.checked_dqn_updates = False
    
    def act(self, states, timestep=0, timesteps=0):
        """
        Select actions for all agents based on their observations.
        
        Args:
            states: List of agent observations
            timestep: Current timestep
            timesteps: Total timesteps
            
        Returns:
            actions: List of actions for each agent
        """
        with torch.no_grad():
            actions = [agent.act(states[i], timestep, timesteps) 
                      for i, agent in enumerate(self.agents)]
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
        # Store individual transitions for each agent
        for i, agent in enumerate(self.agents):
            self.memory.add_samples(
                observations=states[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_states[i],
                dones=dones[i],
                global_states=global_state,
                next_global_states=next_global_state
            )
    
    def update(self):
        """
        Update the QMIX agent (mixing network and individual agents).
        
        This method samples from the replay buffer, computes the loss, and updates
        both the individual agent networks and the mixing network.
        
        Returns:
            info: Dictionary with training metrics
        """
        self.training_steps += 1
        
        # Sample batch from memory
        if self.memory.size < self.batch_size:
            return {}
        
        batch = self.memory.sample(self.batch_size)
        
        # Extract batch data
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        dones = batch["dones"]
        global_states = batch["global_states"]
        next_global_states = batch["next_global_states"]
        
        # Get Q-values for each agent for current observations
        agent_q_values = []
        target_agent_q_values = []
        chosen_actions_q_values = []
        max_next_q_values = []
        
        for i, agent in enumerate(self.agents):
            # Get Q-values from main network
            q_values = agent.models["q_network"](states[:, i])
            agent_q_values.append(q_values)
            
            # Get chosen action Q-values
            batch_indices = torch.arange(self.batch_size, device=self.device)
            chosen_actions_q = q_values[batch_indices, actions[:, i].long()]
            chosen_actions_q_values.append(chosen_actions_q.unsqueeze(1))
            
            # Get target Q-values for next state
            target_q_values = agent.models["target_q_network"](next_states[:, i])
            max_next_q = target_q_values.max(dim=1)[0]
            max_next_q_values.append(max_next_q.unsqueeze(1))
        
        # Stack Q-values
        chosen_actions_q_values = torch.cat(chosen_actions_q_values, dim=1)
        max_next_q_values = torch.cat(max_next_q_values, dim=1)
        
        # Calculate mixed Q-value
        mixed_q_values = self.mixing_network(chosen_actions_q_values, global_states)
        
        # Calculate target mixed Q-value
        target_mixed_q_values = self.target_mixing_network(max_next_q_values, next_global_states)
        
        # Calculate targets
        targets = rewards.sum(dim=1, keepdim=True) + self.gamma * (1 - dones.any(dim=1, keepdim=True)) * target_mixed_q_values
        
        # Calculate loss
        loss = F.mse_loss(mixed_q_values, targets.detach())
        
        # Optimize mixing network
        self.optimizer.zero_grad()
        loss.backward()
        
        # This backward pass updates the weights of the mixing network
        # The gradients flow through the mixing network but not through individual agent networks
        # Individual agents are updated separately in their own optimization steps
        self.optimizer.step()
        
        # Update individual agents
        agent_info = {}
        for i, agent in enumerate(self.agents):
            # Important: Each agent.update() call will:
            # 1. Sample from the agent's memory (separate from the QMIX memory)
            # 2. Compute the individual agent's loss (DQN loss)
            # 3. Perform backpropagation through the agent's Q-network
            # 4. Update the agent's Q-network parameters using the agent's optimizer
            #
            # This means each agent's parameters are updated separately from the mixing network
            # and from other agents' parameters, allowing for independent learning
            agent_update_info = agent.update()
            
            # Print info about the first agent's update on the first training step
            if not self.checked_dqn_updates and i == 0 and self.training_steps == 1:
                print("\nDQN Agent Update Details:")
                print("  Each agent's Q-network parameters are updated independently through that agent's optimizer")
                print("  Updates happen in two separate steps:")
                print("    1. The mixing network is updated to better combine individual agent Q-values")
                print("    2. Each agent's Q-network is updated independently to learn better individual policies")
                print("  This dual update approach allows agents to learn both individual and coordinated strategies")
                self.checked_dqn_updates = True
            
            agent_info[f"agent_{i}"] = agent_update_info
        
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


# Custom Memory for QMIX
class QMIXMemory(RandomMemory):
    """
    Memory class for QMIX algorithm that stores global states along with agent observations.
    """
    def __init__(self, memory_size, num_agents, device="cpu"):
        """
        Initialize the QMIX memory.
        
        Args:
            memory_size: Maximum size of the replay buffer
            num_agents: Number of agents in the environment
            device: Device to use for computation ("cpu" or "cuda:X")
        """
        super().__init__(memory_size=memory_size, device=device)
        self.num_agents = num_agents
        
        # Add global states to the memory
        # The actual shape will be determined when first global state is added
        self.tensors["global_states"] = None
        self.tensors["next_global_states"] = None
    
    def add_samples(self, observations=None, actions=None, rewards=None, next_observations=None, 
                   dones=None, global_states=None, next_global_states=None):
        """
        Add samples to the replay buffer, including global states.
        
        Args:
            observations: Agent observations
            actions: Agent actions
            rewards: Agent rewards
            next_observations: Next agent observations
            dones: Done flags
            global_states: Global state
            next_global_states: Next global state
        """
        # Create batch dimension if not already present
        if observations is not None and observations.dim() == 1:
            observations = observations.unsqueeze(0)
        if actions is not None and not isinstance(actions, torch.Tensor):
            actions = torch.tensor([actions], device=self.device)
        if rewards is not None and not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor([rewards], device=self.device, dtype=torch.float32)
        if next_observations is not None and next_observations.dim() == 1:
            next_observations = next_observations.unsqueeze(0)
        if dones is not None and not isinstance(dones, torch.Tensor):
            dones = torch.tensor([dones], device=self.device)
        if global_states is not None and global_states.dim() == 1:
            global_states = global_states.unsqueeze(0)
        if next_global_states is not None and next_global_states.dim() == 1:
            next_global_states = next_global_states.unsqueeze(0)
        
        # If this is the first global state we're adding, initialize the tensors
        if global_states is not None and self.tensors["global_states"] is None:
            global_state_shape = global_states.shape[1:]  # Remove batch dimension
            self.tensors["global_states"] = torch.zeros(
                (self.memory_size, *global_state_shape), 
                dtype=torch.float32, 
                device=self.device
            )
            self.tensors["next_global_states"] = torch.zeros(
                (self.memory_size, *global_state_shape), 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Call the parent class's add_samples method for observations, actions, rewards, etc.
        samples = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "dones": dones
        }
        
        super().add_samples(samples)
        
        # Add global states separately
        if global_states is not None:
            batch_size = global_states.shape[0]
            indices = torch.arange(self.head, self.head + batch_size) % self.memory_size
            self.tensors["global_states"][indices] = global_states
            
        if next_global_states is not None:
            batch_size = next_global_states.shape[0]
            indices = torch.arange(self.head, self.head + batch_size) % self.memory_size
            self.tensors["next_global_states"][indices] = next_global_states


# Add a wrapper function to correctly handle global states in SequentialTrainer
def qmix_training_wrapper(train_fn, env, agent):
    """
    A wrapper function for SequentialTrainer that ensures global states are properly passed to the agent.
    
    Args:
        train_fn: The original train function from SequentialTrainer
        env: The wrapped environment
        agent: The QMIX agent
    
    Returns:
        The original training function's return value
    """
    # Store original step function
    original_step = env.step
    
    # Define wrapped step function that extracts global states
    def wrapped_step(actions):
        next_obs, rewards, dones, truncateds, info = original_step(actions)
        
        # In case the environment doesn't provide global_state, create it if needed
        if 'global_state' not in info and hasattr(env.unwrapped, '_get_full_obs'):
            # Use the full observation as a global state if the environment doesn't provide one
            full_obs = env.unwrapped._get_full_obs()
            info['global_state'] = full_obs.flatten().astype(np.float32)
        
        # Store global states for record_transition
        if 'global_state' in info:
            if not hasattr(env, 'current_global_state'):
                # If first step, get the current global state from reset
                observations, reset_info = env.reset(num_agents=len(next_obs))
                
                # Handle case where reset_info doesn't have global_state
                if 'global_state' not in reset_info and hasattr(env.unwrapped, '_get_full_obs'):
                    full_obs = env.unwrapped._get_full_obs()
                    reset_info['global_state'] = full_obs.flatten().astype(np.float32)
                
                env.current_global_state = reset_info.get('global_state', None)
            
            # Store the next global state for record_transition
            env.next_global_state = info.get('global_state', None)
            
            # Store tracking metrics if available
            env.total_steps = info.get('total_steps', 0)
            env.total_revisits = info.get('total_revisits', 0)
            env.steps_to_solve = info.get('steps_to_solve', -1)
            env.maze_solved = info.get('maze_solved', False)
            env.unique_states_visited = info.get('unique_states_visited', 0)
            env.exploration_percentage = info.get('exploration_percentage', 0)
        
        return next_obs, rewards, dones, truncateds, info
    
    # Override environment's step method
    env.step = wrapped_step
    
    # Define hook function for the trainer
    def after_step_hook(timestep, observations, actions, rewards, next_observations, dones, infos):
        """
        Hook function called after each step in the environment.
        
        This function records the transition with global states and also prints
        tracking metrics periodically.
        """
        # Only record transition if the current and next global states are available
        if hasattr(env, 'current_global_state') and hasattr(env, 'next_global_state'):
            # Convert global states to tensors if needed
            current_gs = env.current_global_state
            next_gs = env.next_global_state
            
            if not isinstance(current_gs, torch.Tensor):
                current_gs = torch.tensor(current_gs, device=agent.device, dtype=torch.float32)
            if not isinstance(next_gs, torch.Tensor):
                next_gs = torch.tensor(next_gs, device=agent.device, dtype=torch.float32)
                
            # Record transition with global states
            agent.record_transition(
                states=observations,
                actions=actions,
                rewards=rewards,
                next_states=next_observations,
                dones=dones,
                global_state=current_gs,
                next_global_state=next_gs
            )
            
            # Update current global state for next step
            env.current_global_state = env.next_global_state
            
            # Print tracking metrics every 100 steps
            if hasattr(env, 'total_steps') and env.total_steps % 100 == 0:
                print(f"\nStep {env.total_steps}:")
                print(f"  Total revisits: {getattr(env, 'total_revisits', 'N/A')}")
                print(f"  Unique states visited: {getattr(env, 'unique_states_visited', 'N/A')}")
                print(f"  Exploration percentage: {getattr(env, 'exploration_percentage', 0):.2f}%")
                if getattr(env, 'maze_solved', False):
                    print(f"  Maze solved in {getattr(env, 'steps_to_solve', 'N/A')} steps")
    
    # Add hook to the train function
    def wrapped_train_fn():
        # Call the original train function with the hook
        return train_fn(after_step_hook=after_step_hook)
    
    # Call the wrapped train function
    result = wrapped_train_fn()
    
    # Restore original step function
    env.step = original_step
    
    return result


# Q-Network for MazeEnv
class QNetwork(Model):
    def __init__(self, observation_space, action_space, device="cpu", hidden_dim=64):
        """
        Q-Network for DQN agent.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            device: Device to use for computation
            hidden_dim: Hidden dimension of the network
        """
        super().__init__(observation_space, action_space, device)
        
        # Determine input size based on observation space
        if len(observation_space.shape) == 1:  # Vector observation
            input_dim = observation_space.shape[0]
        else:  # Image-like observation (maze grid)
            # Flatten the observation
            input_dim = np.prod(observation_space.shape)
        
        # Q-network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space.n)
        )
        
        self.to(device)
    
    def compute(self, inputs, role=""):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input dictionary with states
            role: Network role (not used here)
            
        Returns:
            values: Q-values for each action
        """
        # Get states from inputs
        states = inputs["states"]
        
        # Handle different observation types
        if len(states.shape) > 2:  # Image-like observation
            states = states.float().reshape(states.shape[0], -1)  # Flatten
        else:  # Vector observation
            states = states.float()
        
        # Forward pass
        q_values = self.network(states)
        
        return q_values, {}


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


def train_qmix(maze_size=10, n_agents=2, obstacle_ratio=0.1, n_episodes=500, max_steps=100, 
               gamma=0.99, learning_rate=1e-3, batch_size=32, memory_size=10000, 
               target_update_frequency=100, hidden_dim=64, mixing_embed_dim=32, 
               hypernet_embed=64, device="cpu", obs_type="full", pob_size=1,
               log_dir=None, log_frequency=10):
    """
    Train a QMIX agent on a multi-agent maze environment.
    
    Args:
        maze_size: Size of the maze (maze_size x maze_size)
        n_agents: Number of agents in the environment
        obstacle_ratio: Ratio of obstacles in the maze
        n_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        gamma: Discount factor
        learning_rate: Learning rate
        batch_size: Batch size for training
        memory_size: Size of the replay buffer
        target_update_frequency: How often to update target networks
        hidden_dim: Hidden dimension for agent networks
        mixing_embed_dim: Dimension of mixing embedding
        hypernet_embed: Dimension of hypernetwork embedding
        device: Device to use for training ("cpu" or "cuda:X")
        obs_type: Observation type ("full" or "partial")
        pob_size: Size of partial observation window
        log_dir: Directory to save metrics (if None, just print)
        log_frequency: How often to log metrics (in episodes)
        
    Returns:
        qmix_agent: Trained QMIX agent
        save_path: Path where the model was saved
    """
    # Create maze generator
    maze_generator = RandomBlockMazeGenerator(maze_size, obstacle_ratio=obstacle_ratio)
    
    # Create environment with multi-agent support
    env = MazeEnv(
        maze_generator=maze_generator,
        pob_size=pob_size,
        obs_type=obs_type,
        render_trace=True
    )
    
    # Wrap environment for skrl
    env = wrap_env(env)
    
    # Reset environment to get observation and action spaces for multiple agents
    observations, _ = env.reset(num_agents=n_agents)
    
    # Create agent models, handling different observation types
    agent_models = []
    for i in range(n_agents):
        models = {}
        models["q_network"] = QNetwork(
            env.observation_space[i], env.action_space[i], device=device, hidden_dim=hidden_dim
        )
        models["target_q_network"] = QNetwork(
            env.observation_space[i], env.action_space[i], device=device, hidden_dim=hidden_dim
        )
        agent_models.append(models)
    
    # Create mixing network - use the state dimension from the global state
    # Get a sample global state to determine its shape
    _, info = env.reset(num_agents=n_agents)
    global_state_shape = info.get('global_state').shape
    
    # Create a Box space for the global state
    global_state_space = gym.spaces.Box(
        low=0,
        high=6,  # Based on MazeEnv's observation space values
        shape=global_state_shape,
        dtype=np.float32
    )
    
    mixing_network = MixingNetwork(
        global_state_space, env.action_space[0], device=device,
        num_agents=n_agents, mixing_embed_dim=mixing_embed_dim, hypernet_embed=hypernet_embed
    )
    
    # Create memory
    memory = QMIXMemory(memory_size=memory_size, num_agents=n_agents, device=device)
    
    # Create agent configs
    agent_cfg = {
        "gamma": gamma,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "exploration": {
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "timesteps": n_episodes * max_steps * 0.5,  # Explore for 50% of training
        },
        "target_update_frequency": target_update_frequency,
        "train_frequency": 1,
        "double_q": True
    }
    
    # Create individual DQN agents
    dqn_agents = []
    for i in range(n_agents):
        agent_memory = RandomMemory(memory_size=memory_size, device=device)
        dqn_agent = DQN(
            models=agent_models[i],
            memory=agent_memory,
            cfg=agent_cfg,
            observation_space=env.observation_space[i],
            action_space=env.action_space[i],
            device=device
        )
        dqn_agents.append(dqn_agent)
    
    # Create QMIX agent
    qmix_cfg = {
        "gamma": gamma,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_update_frequency": target_update_frequency
    }
    
    qmix_agent = QMIXAgent(
        agents=dqn_agents,
        mixing_network=mixing_network,
        memory=memory,
        cfg=qmix_cfg,
        observation_space=global_state_space,  # Use the global state space
        action_space=env.action_space,
        device=device
    )
    
    # Track episode metrics
    episode_metrics = []
    
    # Create a custom hook for metrics tracking
    def episode_hook(episode, timestep):
        """Called at the end of each episode"""
        if episode % log_frequency == 0:
            # Get metrics from the environment
            metrics = {
                'total_steps': getattr(env, 'total_steps', 0),
                'total_revisits': getattr(env, 'total_revisits', 0),
                'unique_states_visited': getattr(env, 'unique_states_visited', 0),
                'exploration_percentage': getattr(env, 'exploration_percentage', 0),
                'maze_solved': getattr(env, 'maze_solved', False),
                'steps_to_solve': getattr(env, 'steps_to_solve', -1)
            }
            
            # Log metrics
            log_metrics(metrics, episode, log_dir)
            
            # Store metrics for later analysis
            episode_metrics.append(metrics)
    
    # Configure trainer
    cfg_trainer = {
        "timesteps": n_episodes * max_steps,
        "headless": True
    }
    
    trainer = SequentialTrainer(
        cfg=cfg_trainer,
        env=env,
        agents=qmix_agent
    )
    
    # Add our custom episode hook to the trainer
    trainer.on_episode_end = episode_hook
    
    # Train the agent using the wrapper function
    qmix_training_wrapper(trainer.train, env, qmix_agent)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"qmix_maze_agent_{timestamp}.pt"
    qmix_agent.save(save_path)
    
    print("\nTraining Summary:")
    print(f"  Total episodes: {n_episodes}")
    if episode_metrics:
        final_metrics = episode_metrics[-1]
        print(f"  Final exploration percentage: {final_metrics.get('exploration_percentage', 0):.2f}%")
        print(f"  Final unique states visited: {final_metrics.get('unique_states_visited', 0)}")
        print(f"  Total revisits: {final_metrics.get('total_revisits', 0)}")
        if final_metrics.get('maze_solved', False):
            print(f"  Maze solved in {final_metrics.get('steps_to_solve', -1)} steps")
    
    return qmix_agent, save_path


def test_qmix(agent=None, maze_size=10, n_agents=2, obstacle_ratio=0.1, n_episodes=5, 
              max_steps=100, device="cpu", path=None, render=True, obs_type="full", pob_size=1):
    """
    Test QMIX agent on the maze environment.
    
    Args:
        agent: Trained QMIX agent (if None and path is provided, will load from file)
        maze_size: Size of the maze
        n_agents: Number of agents
        obstacle_ratio: Ratio of obstacles in the maze
        n_episodes: Number of episodes to test
        max_steps: Maximum steps per episode
        device: Device to use for testing ("cpu" or "cuda:X")
        path: Path to load a pre-trained agent from
        render: Whether to render the environment
        obs_type: Observation type ("full" or "partial")
        pob_size: Size of partial observation window
        
    Returns:
        scores: List of rewards per episode
    """
    # Create maze generator
    maze_generator = RandomBlockMazeGenerator(maze_size, obstacle_ratio=obstacle_ratio)
    
    # Create environment with multi-agent support
    env = MazeEnv(
        maze_generator=maze_generator,
        pob_size=pob_size,
        obs_type=obs_type,
        render_trace=True,
        live_display=render
    )
    
    # Wrap environment for skrl
    env = wrap_env(env)
    
    # Reset to setup environment with correct number of agents
    observations, info = env.reset(num_agents=n_agents)
    
    # Get a sample global state to determine its shape for loading the agent
    global_state_shape = info.get('global_state').shape
    global_state_space = gym.spaces.Box(
        low=0,
        high=6,
        shape=global_state_shape,
        dtype=np.float32
    )
    
    # Load agent if not provided but path is
    if agent is None and path is not None:
        # Create a new agent with the correct architecture
        agent_models = []
        for i in range(n_agents):
            models = {}
            models["q_network"] = QNetwork(
                env.observation_space[i], env.action_space[i], device=device
            )
            models["target_q_network"] = QNetwork(
                env.observation_space[i], env.action_space[i], device=device
            )
            agent_models.append(models)
        
        # Create mixing network
        mixing_network = MixingNetwork(
            global_state_space, env.action_space[0], device=device,
            num_agents=n_agents, mixing_embed_dim=32, hypernet_embed=64
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
                observation_space=env.observation_space[i],
                action_space=env.action_space[i],
                device=device
            )
            dqn_agents.append(dqn_agent)
        
        # Create QMIX agent
        agent = QMIXAgent(
            agents=dqn_agents,
            mixing_network=mixing_network,
            memory=memory,
            observation_space=global_state_space,
            action_space=env.action_space,
            device=device
        )
        
        # Load weights
        agent.load(path)
    
    # Testing loop
    scores = []
    success_count = 0
    all_metrics = []
    
    for episode in range(1, n_episodes+1):
        observations, info = env.reset(num_agents=n_agents)
        total_reward = 0
        steps = 0
        
        # Get initial global state
        current_global_state = info.get('global_state', None)
        
        while steps < max_steps:
            # Select actions (no exploration)
            actions = agent.act(observations, timestep=0, timesteps=1)
            
            # Take actions in the environment
            next_observations, rewards, dones, truncateds, info = env.step(actions)
            
            # Get next global state
            next_global_state = info.get('global_state', None)
            
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
            
            # Check if any agent reached the goal
            if any(dones) and not any(truncateds):
                success_count += 1
            
            # End episode if any agent is done
            if any(dones) or any(truncateds):
                break
        
        # Collect metrics for this episode
        episode_metrics = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'success': any(dones) and not any(truncateds),
            'total_steps': info.get('total_steps', steps),
            'total_revisits': info.get('total_revisits', 0),
            'unique_states_visited': info.get('unique_states_visited', 0),
            'exploration_percentage': info.get('exploration_percentage', 0),
            'maze_solved': info.get('maze_solved', any(dones) and not any(truncateds)),
            'steps_to_solve': info.get('steps_to_solve', steps if any(dones) and not any(truncateds) else -1)
        }
        all_metrics.append(episode_metrics)
        
        # Log metrics
        print(f"\nTest Episode {episode}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Success: {any(dones) and not any(truncateds)}")
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
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create log directory
    log_dir = "/home/mila/m/munjuluv/scratch/logs/qmix_maze_logs"
    
    # Set parameters
    maze_size = 10
    n_agents = 2
    obstacle_ratio = 0.1
    n_episodes = 500
    max_steps = 100
    obs_type = "full"
    
    print("\nUsing built-in multi-agent support from gym_maze")
    print(f"Training with {n_agents} agents in a {maze_size}x{maze_size} maze")
    
    # Train QMIX agent
    print("\nTraining QMIX agent...")
    agent, save_path = train_qmix(
        maze_size=maze_size, 
        n_agents=n_agents, 
        obstacle_ratio=obstacle_ratio, 
        n_episodes=n_episodes, 
        max_steps=max_steps, 
        obs_type=obs_type, 
        device=device,
        log_dir=log_dir,
        log_frequency=10  # Log every 10 episodes
    )
    
    # Test QMIX agent
    print("\nTesting QMIX agent...")
    test_qmix(
        agent, 
        maze_size=maze_size, 
        n_agents=n_agents, 
        obstacle_ratio=obstacle_ratio, 
        n_episodes=5, 
        max_steps=max_steps, 
        obs_type=obs_type, 
        device=device,
        render=True  # Show visualization during testing
    ) 