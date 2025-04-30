import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import time
import os
import copy
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import json

from skrl.agents.torch.dqn.dqn import DQN
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.multi_agents.torch import MultiAgent

# Use gym_maze's built-in multi-agent support
from gym_maze.envs import MazeEnv
from gym_maze.envs.enhanced_maze import EnhancedMazeEnv
from gym_maze.envs.generators import RandomBlockMazeGenerator


# QMIX Mixing Network as a skrl Model
class MixingNetwork(Model):
    def __init__(self, observation_space, action_space, device="cpu", 
                 num_agents=2, mixing_embed_dim=32, hypernet_embed=64):
        """
        Mixer network for QMIX that takes individual Q-values and mixes them monotonically.
        """
        super().__init__(observation_space, action_space, device)
        
        self.num_agents = num_agents
        self.state_dim = observation_space.shape[0]
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetworks for weights and biases
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
        
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        """
        Forward pass through the mixing network.
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)  # Add batch dimension if missing
            
        batch_size = agent_qs.size(0)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, -1)
        b1 = self.hyper_b1(states).view(batch_size, 1, -1)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, -1, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        
        return (torch.bmm(hidden, w2) + b2).squeeze(-1)


# QMIX Agent implementation using skrl's MultiAgent
class QMIXAgent(MultiAgent):
    def __init__(self, agents, models, mixing_network, memory, 
                 cfg=None, observation_space=None, action_space=None, device="cpu"):
        """QMIX agent implementation."""
        super().__init__(
            possible_agents=[f"agent_{i}" for i in range(len(agents))],
            models={f"agent_{i}": models[i] for i in range(len(agents))},
            device=device,
        )
        
        self.mixing_network = mixing_network
        self.target_mixing_network = copy.deepcopy(mixing_network)
        self.memory = memory
        self.cfg = cfg if cfg is not None else {}
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Configure optimizer and hyperparameters
        self.optimizer = optim.Adam(self.mixing_network.parameters(), 
                                  lr=self.cfg.get("learning_rate", 1e-3))
        self.gamma = self.cfg.get("gamma", 0.99)
        self.batch_size = self.cfg.get("batch_size", 32)
        self.target_update_frequency = self.cfg.get("target_update_frequency", 1000)
        self.num_agents = len(agents)
        self.agents = agents
        self.training_steps = 0
        self.loss = 0.0
    
    def act(self, states, timestamp=0, timesteps=0):
        """Get actions for each agent based on their observations."""
        actions = []
        for i, state in enumerate(states):
            # Convert state to tensor if it's not already
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            
            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Get action from agent's network
            with torch.no_grad():
                action, _, _ = self.agents[i].act(state, timestamp, timesteps)
                actions.append(action.item() if isinstance(action, torch.Tensor) else int(action))
        return actions
    
    def _to_tensor(self, data, dtype=None, device=None):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype, device=device or self.device)
        if data.dim() == 0:
            data = data.unsqueeze(0)
        return data
    
    def record_transition(self, states, actions, rewards, next_states, dones, global_state=None, next_global_state=None):
        # Convert all inputs to tensors
        states = [self._to_tensor(s, torch.float32) for s in states]
        next_states = [self._to_tensor(s, torch.float32) for s in next_states]
        actions = [self._to_tensor(a, torch.long) for a in actions]
        rewards = [self._to_tensor(r, torch.float32) for r in rewards]
        dones = [self._to_tensor(d, torch.bool) for d in dones]
        
        # Record transitions for individual agents
        for i, agent in enumerate(self.agents):
            agent.memory.add_samples(
                observations=states[i],
                actions=actions[i],
                rewards=rewards[i],
                next_observations=next_states[i],
                dones=dones[i]
            )
        
        # Stack into batch tensors
        states_tensor = torch.cat(states, dim=0).unsqueeze(0)
        next_states_tensor = torch.cat(next_states, dim=0).unsqueeze(0)
        actions_tensor = torch.cat(actions, dim=0).unsqueeze(0)
        rewards_tensor = torch.cat(rewards, dim=0).unsqueeze(0)
        dones_tensor = torch.cat(dones, dim=0).unsqueeze(0)
        
        # Convert global states if provided
        if global_state is not None:
            global_state = self._to_tensor(global_state, torch.float32)
        if next_global_state is not None:
            next_global_state = self._to_tensor(next_global_state, torch.float32)
        
        # Store in QMIX memory
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
        """Update the QMIX agent (mixing network and individual agents)."""
        self.training_steps += 1
        
        # Skip update if we don't have enough samples
        if self.memory.filled < self.batch_size:
            return {}
        
        try:
            # Get required tensor names for sampling
            required_names = ["observations", "actions", "rewards", "next_observations", "dones", 
                           "global_states", "next_global_states"]
            
            # Sample a batch from memory
            batch = self.memory.sample(names=required_names, batch_size=self.batch_size)[0]
            
            # Extract batch data
            states = batch[0]
            actions = batch[1]
            rewards = batch[2]
            next_states = batch[3]
            dones = batch[4]
            global_states = batch[5]
            next_global_states = batch[6]
            
            # Get Q-values for each agent for current observations
            agent_q_values = []
            chosen_actions_q_values = []
            max_next_q_values = []
            
            for i, agent in enumerate(self.agents):
                # Get Q-values from main network
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
            self.loss = loss.item()
            
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
                
                agent_update_info = agent.post_interaction(current_step, training_steps)
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
            
        except Exception as e:
            logging.error(f"Error during update: {e}")
            return {}

    def update_target_networks(self):
        """Update all agent target networks"""
        for i in range(self.num_agents):
            self.agents[i].models["q_network"].update_target_network()


# Q-Network for individual agents
class QNetwork(DeterministicMixin, Model):
    def __init__(self, observation_dim, action_dim, hidden_size=128):
        Model.__init__(self, 
                      observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,)), 
                      action_space=gym.spaces.Discrete(action_dim))
        DeterministicMixin.__init__(self, clip_actions=False)
        
        # Network architecture
        self.layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Target network
        self.target_layers = copy.deepcopy(self.layers)
        for param in self.target_layers.parameters():
            param.requires_grad = False
    
    def compute(self, inputs, role="q_network"):
        """Compute Q-values for given states."""
        states = inputs["states"].float()
        return (self.target_layers(states) if role == "target_q_network" else self.layers(states), {})
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_layers.load_state_dict(self.layers.state_dict())


# Custom Memory for QMIX
class QMIXMemory(RandomMemory):
    """Memory class for QMIX algorithm that stores global states along with agent observations."""
    def __init__(self, memory_size, num_agents, device="cpu"):
        super().__init__(memory_size=memory_size, device=device)
        self.num_agents = num_agents
    
    def _ensure_tensor_exists(self, name, tensor):
        if name not in self.tensors and tensor is not None:
            shape = (self.memory_size,) + tuple(tensor.shape[1:])
            self.tensors[name] = torch.zeros(shape, dtype=tensor.dtype, device=self.device)
    
    def add_samples(self, observations=None, actions=None, rewards=None, next_observations=None, 
                   dones=None, global_states=None, next_global_states=None):
        # Ensure tensors exist
        self._ensure_tensor_exists("global_states", global_states)
        self._ensure_tensor_exists("next_global_states", next_global_states)
        
        # Get batch size from first non-None input
        batch_size = next((x.shape[0] for x in [observations, actions, rewards, global_states] if x is not None), 0)
        if batch_size == 0:
            return
            
        # Calculate indices
        indices = torch.arange(self.memory_index, self.memory_index + batch_size) % self.memory_size
        
        # Store all tensors
        for name, tensor in [
            ("observations", observations),
            ("actions", actions),
            ("rewards", rewards),
            ("next_observations", next_observations),
            ("dones", dones),
            ("global_states", global_states),
            ("next_global_states", next_global_states)
        ]:
            if tensor is not None and name in self.tensors:
                self.tensors[name][indices] = tensor
        
        # Update indices and filled size
        prev_index = self.memory_index
        self.memory_index = (self.memory_index + batch_size) % self.memory_size
        self.filled = self.memory_size if self.memory_index <= prev_index else max(self.filled, self.memory_index)
    
    def sample(self, names, batch_size, mini_batches=1):
        if self.filled < batch_size:
            raise RuntimeError(f"Not enough samples in memory. Have {self.filled}, requested {batch_size}")
            
        indices = torch.randint(0, self.filled, (batch_size,), device=self.device)
        return [[self.tensors[name][indices] for name in names if name in self.tensors]]


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


def plot_training_metrics(log_dir, save_path=None, metrics_file="metrics.csv"):
    """
    Plot training metrics from CSV data with improved visualization.
    Displays: rewards, training loss, maze exploration, and completion metrics.
    
    Args:
        log_dir: Directory containing the metrics CSV file
        save_path: Optional path to save the plot
        metrics_file: Filename of the metrics CSV
    """
    metrics_path = os.path.join(log_dir, metrics_file)
    if not os.path.exists(metrics_path):
        logging.error(f"Metrics file not found at {metrics_path}")
        return
    
    # Try to read metrics from CSV first
    try:
        df = pd.read_csv(metrics_path)
        if df.empty:
            logging.error("Metrics file is empty")
            return
        logging.info(f"Loaded {len(df)} data points from CSV")
    except Exception as e:
        logging.error(f"Error loading CSV metrics: {e}")
        return
    
    # Also try to read tensorboard logs if available
    tb_metrics = {}
    try:
        # Import here so it's optional
        from tensorboard.backend.event_processing import event_accumulator
        
        # Find the most recent events file
        events_files = sorted(Path(log_dir).glob("events.out.tfevents.*"), 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        if events_files:
            events_file = str(events_files[0])
            logging.info(f"Found tensorboard events file: {events_file}")
            
            # Load the events
            ea = event_accumulator.EventAccumulator(events_file)
            ea.Reload()
            
            # Extract metrics from tensorboard
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                if events:
                    tb_metrics[tag] = np.array([(x.step, x.value) for x in events])
                    logging.info(f"Loaded {len(events)} points for {tag}")
    except (ImportError, Exception) as e:
        logging.warning(f"Could not load tensorboard metrics: {e}")
    
    # Set up the figure with a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('QMIX Training Metrics', fontsize=16, fontweight='bold')
    
    # Define colors for consistency
    colors = {
        'reward': '#1f77b4',      # Blue
        'reward_avg': '#ff7f0e',  # Orange
        'loss': '#d62728',        # Red
        'loss_avg': '#e377c2',    # Pink
        'exploration': '#9467bd', # Purple
        'success': '#2ca02c',     # Green
        'revisits': '#8c564b',    # Brown
        'steps': '#17becf',       # Cyan
        'q_value': '#7f7f7f',     # Gray
    }
    
    # Pre-process data
    # Sort by episode/total_steps for consistent plotting
    df = df.sort_values('total_steps')
    
    # Calculate moving averages with adaptive window sizes
    window_size = max(1, min(20, len(df) // 10))  # Adaptive window size
    
    # 1. Plot Episode Rewards
    ax = axes[0, 0]
    if 'episode_reward' in df.columns:
        # Scatter plot for individual rewards
        ax.scatter(df['total_steps'], df['episode_reward'], 
                  color=colors['reward'], alpha=0.3, s=15, label='Episode Reward')
        
        # Plot moving average
        rewards_ma = df['episode_reward'].rolling(window=window_size, min_periods=1).mean()
        ax.plot(df['total_steps'], rewards_ma, 
               color=colors['reward_avg'], linewidth=2, label=f'{window_size}-ep Moving Avg')
        
        # Set reasonable y-limits
        reward_min = df['episode_reward'].min()
        reward_max = df['episode_reward'].max()
        y_margin = max(1, (reward_max - reward_min) * 0.1)
        ax.set_ylim(reward_min - y_margin, reward_max + y_margin)
        
        # Also check tensorboard for reward data
        if 'rewards/episode_reward' in tb_metrics:
            steps, values = tb_metrics['rewards/episode_reward'].T
            # Only plot if not duplicate of CSV data
            if len(steps) > len(df) or not np.array_equal(steps, df['total_steps']):
                logging.info("Using reward data from tensorboard")
                ax.plot(steps, pd.Series(values).rolling(window=window_size, min_periods=1).mean(), 
                      'g--', linewidth=1.5, alpha=0.7, label='TB Rewards (MA)')
    else:
        ax.text(0.5, 0.5, 'No reward data available', 
              ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_title('Episode Rewards', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 2. Plot Training Loss and Q-values
    ax = axes[0, 1]
    
    # Plot loss values
    if False and 'loss' in df.columns and not df['loss'].isna().all():
        # Handle any extreme values
        df['loss'] = df['loss'].replace([np.inf, -np.inf], np.nan)
        if df['loss'].max() > 1000:  # If losses are unusually high
            df['loss'] = df['loss'].clip(upper=df['loss'].quantile(0.95))  # Clip to 95th percentile
        
        # Plot individual loss values 
        ax.scatter(df['total_steps'], df['loss'], 
                 color=colors['loss'], alpha=0.2, s=10, label='Loss')
        
        # Plot moving average
        loss_ma = df['loss'].rolling(window=window_size, min_periods=1).mean()
        ax.plot(df['total_steps'], loss_ma, 
              color=colors['loss_avg'], linewidth=2, label=f'Loss {window_size}-ep MA')
        
        # Set reasonable y-limits
        loss_min = df['loss'].min()
        loss_max = df['loss'].max()
        y_margin = max(0.1, (loss_max - loss_min) * 0.1)
        ax.set_ylim(max(0, loss_min - y_margin), loss_max + y_margin)
    elif 'charts/Q-network loss' in tb_metrics:
        steps, values = tb_metrics['charts/Q-network loss'].T
        ax.plot(steps, values, color=colors['loss'], alpha=0.3, label='Loss (TB)')
        ax.plot(steps, pd.Series(values).rolling(window=window_size, min_periods=1).mean(), 
              color=colors['loss_avg'], linewidth=2, label=f'Loss {window_size}-ep MA')
    else:
        ax.text(0.5, 0.5, 'No loss data available', 
              ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # Add Q-values on the same plot with twin y-axis if available
    if 'q_values' in df.columns and not df['q_values'].isna().all() and df['q_values'].max() > 0:
        ax2 = ax.twinx()
        q_ma = df['q_values'].rolling(window=window_size, min_periods=1).mean()
        ax2.plot(df['total_steps'], q_ma, 
               color=colors['q_value'], linewidth=2, label='Q-value (MA)')
        ax2.set_ylabel('Average Q-value', color=colors['q_value'])
        ax2.tick_params(axis='y', labelcolor=colors['q_value'])
        
        # Add legend for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)
    elif 'charts/q_values' in tb_metrics:
        ax2 = ax.twinx()
        steps, values = tb_metrics['charts/q_values'].T
        q_ma = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
        ax2.plot(steps, q_ma, color=colors['q_value'], linewidth=2, label='Q-value (MA)')
        ax2.set_ylabel('Average Q-value', color=colors['q_value'])
        ax2.tick_params(axis='y', labelcolor=colors['q_value'])
        
        # Add legend for both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)
    else:
        ax.legend(loc='best', frameon=True)
    
    ax.set_title('Training Loss & Q-values', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # 3. Plot Maze Exploration and Success Rate
    ax = axes[1, 0]
    if 'exploration_percentage' in df.columns:
        # Create light area fill for exploration
        ax.fill_between(df['total_steps'], 0, df['exploration_percentage'], 
                       color=colors['exploration'], alpha=0.2)
        
        # Plot exploration line
        ax.plot(df['total_steps'], df['exploration_percentage'], 
               color=colors['exploration'], linewidth=2, label='Exploration %')
        
        # Add success rate on same axis
        if 'success_rate' in df.columns:
            ax.plot(df['total_steps'], df['success_rate'], 
                   color=colors['success'], linewidth=2, label='Success Rate (%)')
        elif 'maze_solved' in df.columns:
            # Calculate success rate from maze_solved
            df['cum_success'] = df['maze_solved'].cumsum()
            df['success_rate'] = 100 * df['cum_success'] / (df.index + 1)
            ax.plot(df['total_steps'], df['success_rate'], 
                   color=colors['success'], linewidth=2, label='Success Rate (%)')
        
        # Add a line at 100% for reference
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_ylim(0, 105)  # Percentage scale with small margin
    elif 'exploration/percentage' in tb_metrics:
        steps, values = tb_metrics['exploration/percentage'].T
        ax.fill_between(steps, 0, values, color=colors['exploration'], alpha=0.2)
        ax.plot(steps, values, color=colors['exploration'], linewidth=2, label='Exploration %')
        
        # Try to get success rate from tensorboard
        if 'metrics/success_rate' in tb_metrics:
            steps_success, values_success = tb_metrics['metrics/success_rate'].T
            ax.plot(steps_success, values_success, 
                   color=colors['success'], linewidth=2, label='Success Rate (%)')
        
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 105)
    else:
        ax.text(0.5, 0.5, 'No exploration data available', 
              ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
    ax.set_title('Maze Exploration & Success Rate', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Percentage (%)')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    # 4. Plot Completion Metrics (Revisits and Steps to Solve)
    ax = axes[1, 1]
    
    # Create second y-axis for steps to solve
    ax2 = ax.twinx() if 'steps_to_solve' in df.columns else None
    
    # Plot revisits
    if 'total_revisits' in df.columns:
        # Scatter for raw data
        ax.scatter(df['total_steps'], df['total_revisits'], 
                 color=colors['revisits'], alpha=0.2, s=10, label='Revisits')
                
        # Moving average for trend
        revisits_ma = df['total_revisits'].rolling(window=window_size, min_periods=1).mean()
        ax.plot(df['total_steps'], revisits_ma, 
              color=colors['revisits'], linewidth=2, label=f'Revisits ({window_size}-ep MA)')
        
        ax.set_ylabel('Number of Revisits', color=colors['revisits'])
        ax.tick_params(axis='y', labelcolor=colors['revisits'])
    elif 'metrics/revisits' in tb_metrics:
        steps, values = tb_metrics['metrics/revisits'].T
        ax.scatter(steps, values, color=colors['revisits'], alpha=0.2, s=10, label='Revisits')
        revisits_ma = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
        ax.plot(steps, revisits_ma, color=colors['revisits'], linewidth=2, 
               label=f'Revisits ({window_size}-ep MA)')
        
        ax.set_ylabel('Number of Revisits', color=colors['revisits'])
        ax.tick_params(axis='y', labelcolor=colors['revisits'])
    
    # Plot steps to solve
    if 'steps_to_solve' in df.columns and ax2 is not None:
        # Get only solved episodes (steps_to_solve >= 0)
        solved_df = df[df['steps_to_solve'] >= 0]
        
        if not solved_df.empty:
            # Plot individual steps to solve
            ax2.scatter(solved_df['total_steps'], solved_df['steps_to_solve'], 
                      color=colors['steps'], alpha=0.5, s=20, label='Steps to Solve')
            
            # Plot moving average if we have enough solved episodes
            if len(solved_df) >= 3:
                steps_window = min(len(solved_df) // 2, 10)
                steps_ma = solved_df['steps_to_solve'].rolling(window=steps_window, min_periods=1).mean()
                ax2.plot(solved_df['total_steps'], steps_ma, 
                       color=colors['steps'], linewidth=2, label=f'Steps ({steps_window}-ep MA)')
            
            ax2.set_ylabel('Steps to Solve', color=colors['steps'])
            ax2.tick_params(axis='y', labelcolor=colors['steps'])
            
            # Set reasonable y-limits
            steps_max = solved_df['steps_to_solve'].max()
            ax2.set_ylim(0, steps_max * 1.1)
    elif 'metrics/steps_to_solve' in tb_metrics and ax2 is None:
        ax2 = ax.twinx()
        steps, values = tb_metrics['metrics/steps_to_solve'].T
        
        # Filter out any negative values (unsuccessful episodes)
        mask = values >= 0
        if np.any(mask):
            steps_filtered = steps[mask]
            values_filtered = values[mask]
            
            ax2.scatter(steps_filtered, values_filtered, 
                       color=colors['steps'], alpha=0.5, s=20, label='Steps to Solve')
            
            if len(steps_filtered) >= 3:
                steps_ma = pd.Series(values_filtered).rolling(window=min(5, len(steps_filtered)//2), min_periods=1).mean()
                ax2.plot(steps_filtered, steps_ma, 
                       color=colors['steps'], linewidth=2, label='Steps (MA)')
            
            ax2.set_ylabel('Steps to Solve', color=colors['steps'])
            ax2.tick_params(axis='y', labelcolor=colors['steps'])
    
    # If we have neither revisits nor steps data
    if ('total_revisits' not in df.columns and 'metrics/revisits' not in tb_metrics and 
        'steps_to_solve' not in df.columns and 'metrics/steps_to_solve' not in tb_metrics):
        ax.text(0.5, 0.5, 'No completion metrics available', 
              ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_title('Maze Completion Metrics', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.grid(True, alpha=0.3)
    
    # Add legend for both axes if needed
    if ax2 is not None:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best', frameon=True)
    else:
        ax.legend(loc='best', frameon=True)
    
    # Add a overall caption with key information
    try:
        # Get config if available
        config_path = os.path.join(log_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create config text
            config_text = (
                f"Maze: {config.get('maze_size', 'N/A')}×{config.get('maze_size', 'N/A')}, "
                f"Agents: {config.get('num_agents', 'N/A')}, "
                f"Obs: {config.get('observation_type', 'N/A')}, "
                f"LR: {config.get('learning_rate', 'N/A')}, "
                f"Batch: {config.get('batch_size', 'N/A')}"
            )
            
            fig.text(0.5, 0.01, config_text, ha='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    except Exception as e:
        logging.warning(f"Could not add config caption: {e}")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    if save_path is None:
        save_path = os.path.join(log_dir, "training_metrics.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved metrics plot to {save_path}")
    plt.close()
    
    return save_path


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
    learning_starts=1000,
    obs_type="full",
    pob_size=1,
    obstacle_ratio=0.1,
    log_dir=None
):
    """
    Train a QMIX agent to solve a maze.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create experiment name with timestamp if not provided
    if experiment_name is None:
        timestamp = int(time.time())
        experiment_name = f"qmix_maze_{num_agents}_{timestamp}"
    
    # Create logging directory
    if log_dir is None:
        log_dir = os.path.join("logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting QMIX training for {training_steps} steps")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Training with {num_agents} agents in a {maze_size}x{maze_size} maze")
    logger.info(f"Observation type: {obs_type}, POB size: {pob_size}")
    
    # Setup tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    
    # Save configuration info to tensorboard
    config = {
        'maze_size': maze_size,
        'num_agents': num_agents,
        'observation_type': obs_type,
        'pob_size': pob_size,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'gamma': gamma,
        'memory_size': memory_size,
        'target_update_frequency': target_update_frequency,
        'agent_update_frequency': agent_update_frequency,
        'training_steps': training_steps,
        'seed': seed,
        'exploration': {
            'initial_epsilon': initial_epsilon,
            'final_epsilon': final_epsilon,
            'timesteps': epsilon_steps
        },
        'exploration_threshold': exploration_percentage_threshold,
    }
    
    # Log hyperparameters to TensorBoard
    writer.add_text("hyperparameters", str(config))
    
    # Also save as JSON for reference
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Create multi-agent maze environment
    env = MazeEnv(
        maze_generator=RandomBlockMazeGenerator(
            maze_size=maze_size,
            obstacle_ratio=obstacle_ratio,
        ),
        pob_size=pob_size,
        action_type='VonNeumann',
        obs_type=obs_type,
        live_display=False,
        render_trace=True
    )
    
    # Reset the environment to get initial observation shapes
    observations, info = env.reset(num_agents=num_agents, seed=seed)
    
    # Get observation dimensions
    obs_dim = observations[0].shape[0]
    action_dim = env.action_space.n
    
    logger.info(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")
    
    # Create individual agent models
    agent_models = []
    for i in range(num_agents):
        models = {}
        models["q_network"] = QNetwork(obs_dim, action_dim)
        models["target_q_network"] = QNetwork(obs_dim, action_dim)
        # Copy target network weights from main network
        models["target_q_network"].load_state_dict(models["q_network"].state_dict())
        agent_models.append(models)
    
    # Determine global state dimension
    if hasattr(env.unwrapped, '_get_full_obs'):
        full_obs = env.unwrapped._get_full_obs()
        global_state_dim = full_obs.flatten().shape[0]
    else:
        global_state_dim = maze_size * maze_size * 3
    
    logger.info(f"Global state dimension: {global_state_dim}")
    
    # Create mixing network
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mixing_network = MixingNetwork(
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(global_state_dim,)),
        action_space=env.action_space,
        device=device,
        num_agents=num_agents,
        mixing_embed_dim=32,
        hypernet_embed=64
    )
    
    # Create replay memory
    memory = QMIXMemory(
        memory_size=memory_size,
        num_agents=num_agents,
        device=device
    )
    
    # Create individual DQN agents
    dqn_agents = []
    for i in range(num_agents):
        agent_memory = RandomMemory(memory_size=memory_size, device=device)
        
        # Initialize the memory tensors
        dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=device)
        dummy_action = torch.zeros((1,), dtype=torch.long, device=device)
        dummy_reward = torch.zeros((1,), dtype=torch.float32, device=device)
        dummy_done = torch.zeros((1,), dtype=torch.bool, device=device)
        
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
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "target_update_frequency": target_update_frequency,
            "tau": tau
        },
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
        action_space=env.action_space,
        device=device
    )
    
    # Start training loop
    episode = 0
    current_step = 0
    start_time = time.time()
    
    # Tracking variables
    total_solved = 0
    total_episodes = 0
    best_exploration = 0
    best_reward = float('-inf')
    
    # Create progress bar
    from tqdm import tqdm
    pbar = tqdm(total=training_steps, desc="Training")
    
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
        episode_loss = 0
        episode_q_values = 0
        
        # Safety counter to prevent infinite loops
        max_episode_safety_limit = 2000
        
        # Run episode until it's done (maze is solved) or safety limit reached
        while not episode_done and episode_step < max_episode_safety_limit:
            # Get actions from agent
            actions = agent.act(observations, current_step, training_steps)
            
            # Take a step in the environment
            next_observations, rewards, dones, truncated, info = env.step(actions)
            
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
            # if episode_reward < -500:
            #     episode_done = True
            #     continue
            episode_done = any(dones) or any(truncated)
            
            # Get current exploration percentage and revisits count
            exploration_percentage = info.get('exploration_percentage', 0)
            total_revisits = info.get('total_revisits', 0)
            unique_states_visited = info.get('unique_states_visited', 0)
            
            # Log these metrics at each step to show progress over time
            writer.add_scalar("charts/exploration", exploration_percentage, current_step)
            writer.add_scalar("charts/revisits", total_revisits, current_step)
            writer.add_scalar("charts/unique_states", unique_states_visited, current_step)
            
            # Only update the agent if we've collected enough samples
            if current_step >= learning_starts and current_step % agent_update_frequency == 0 and memory.filled >= batch_size:
                update_info = agent.post_interaction(current_step, training_steps)
                if 'loss' in update_info:
                    episode_loss = update_info['loss']
                    writer.add_scalar("charts/Q-network loss", update_info['loss'], current_step)
            
            # Check if we've reached the training steps limit
            if current_step >= training_steps:
                break
            
            # Log agent Q-values and exploration periodically
            if current_step % 100 == 0:
                # Log average Q-values across all agents
                avg_q_values = 0
                q_count = 0
                for i, agent_model in enumerate(agent_models):
                    with torch.no_grad():
                        # Sample states from the agent's memory
                        if dqn_agents[i].memory.filled > 0:
                            states_sample = dqn_agents[i].memory.tensors["observations"][:min(100, dqn_agents[i].memory.filled)]
                            q_values = agent_model["q_network"].compute({"states": states_sample})[0]
                            avg_q = q_values.mean().item()
                            avg_q_values += avg_q
                            q_count += 1
                            writer.add_scalar(f"charts/agent_{i}_q_values", avg_q, current_step)
                
                if q_count > 0:
                    episode_q_values = avg_q_values / q_count
                    writer.add_scalar("charts/q_values", episode_q_values, current_step)
                
                # Log current exploration epsilon
                epsilon = max([agent.cfg["exploration"]["final_epsilon"] for agent in dqn_agents]) 
                if current_step < epsilon_steps:
                    # Calculate current epsilon based on linear decay
                    epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (current_step / epsilon_steps)
                writer.add_scalar("Exploration / Exploration epsilon", epsilon, current_step)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "episode": episode,
                "reward": episode_reward,
                "steps": episode_step,
                "explored": f"{exploration_percentage:.1f}%"
            })
        
        # Handle episode completion
        total_episodes += 1
        maze_solved = info.get('maze_solved', False)
        if maze_solved:
            total_solved += 1
            steps_to_solve = info.get('steps_to_solve', episode_step)
            writer.add_scalar("charts/steps_to_solve", steps_to_solve, current_step)
            writer.add_scalar("charts/solved", 1, current_step)  # Binary flag for solving
        else:
            writer.add_scalar("charts/solved", 0, current_step)  # Not solved
        
        # Calculate success rate
        success_rate = (total_solved / total_episodes) * 100
        
        # Track best exploration so far
        if exploration_percentage > best_exploration:
            best_exploration = exploration_percentage
        
        # Track best reward so far
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Log episode-level metrics
        writer.add_scalar("charts/episode_reward", episode_reward, current_step)
        writer.add_scalar("charts/episode_length", episode_step, current_step)
        writer.add_scalar("charts/success_rate", success_rate, current_step)
        writer.add_scalar("charts/best_exploration", best_exploration, current_step)
        writer.add_scalar("charts/best_reward", best_reward, current_step)
        
        # Log terminal output every N episodes
        if episode % log_every == 0:
            elapsed_time = time.time() - start_time
            
            logger.info(f"\nEpisode {episode}/{training_steps//max_episode_safety_limit} "
                        f"({current_step}/{training_steps} steps)")
            logger.info(f"  Reward: {episode_reward:.2f} (best: {best_reward:.2f})")
            logger.info(f"  Steps: {episode_step}")
            logger.info(f"  Exploration: {exploration_percentage:.2f}% (best: {best_exploration:.2f}%)")
            logger.info(f"  Success Rate: {success_rate:.2f}%")
            logger.info(f"  Loss: {episode_loss:.4f}")
            logger.info(f"  Time elapsed: {elapsed_time:.1f}s")
            logger.info(f"  Unique states: {info.get('unique_states_visited', 0)}")
            logger.info(f"  Revisits: {total_revisits}")
        
        # Check if we've reached the exploration threshold
        if exploration_percentage > exploration_percentage_threshold:
            logger.info(f"Reached exploration threshold of {exploration_percentage_threshold}%!")
            logger.info(f"Explored {info.get('unique_states_visited', 0)} states out of {maze_size * maze_size} total")
            logger.info(f"Training completed after {current_step} steps and {episode+1} episodes")
            break
        
        # Increment episode counter
        episode += 1
    
    # Close progress bar
    pbar.close()
    
    # Save the trained agent
    model_path = os.path.join(log_dir, "qmix_agent.pt")
    torch.save({
        'mixing_network': mixing_network.state_dict(),
        'agent_models': [m["q_network"].state_dict() for m in agent_models],
    }, model_path)
    logger.info(f"Agent saved to {model_path}")
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Generate plots using plotting module
    from plotting import plot_training_metrics
    plot_path = plot_training_metrics(log_dir)
    logger.info(f"Training metrics plots saved to: {plot_path}")
    
    # Final metrics
    elapsed_time = time.time() - start_time
    logger.info(f"\nTraining completed:")
    logger.info(f"  Total steps: {current_step}")
    logger.info(f"  Total episodes: {episode}")
    logger.info(f"  Success rate: {success_rate:.2f}%")
    logger.info(f"  Final exploration: {exploration_percentage:.2f}%")
    logger.info(f"  Time elapsed: {elapsed_time:.2f}s")
    
    return agent


def test_qmix(agent=None, maze_size=10, n_agents=2, obstacle_ratio=0.1, n_episodes=5, 
              device="cpu", path=None, render=True, obs_type="full", pob_size=1, log_dir=None):
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
    env = EnhancedMazeEnv(
        maze_generator=maze_generator,
        pob_size=pob_size,
        action_type='VonNeumann',  # Default action type
        obs_type=obs_type,
        live_display=False,  # Show live display during testing if render=True
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
    
    # Initialize metric lists
    episode_rewards = []
    episode_steps = []
    exploration_percentages = []
    revisits = []
    unique_states = []
    steps_to_solve = []
    
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
            env.render()

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
                
        # Try to get video, but don't fail if it's not available

        if total_reward > 0:
            env._get_video(interval=200, gif_path=os.path.join(log_dir, f"episode_{episode}.gif")).to_html5_video()
        
        # Store metrics for this episode
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        exploration_percentages.append(info.get('exploration_percentage', 0))
        revisits.append(info.get('total_revisits', 0))
        unique_states.append(info.get('unique_states_visited', 0))
        
        if any(dones) and not any(truncated):
            steps_to_solve.append(steps)
    
    # Calculate aggregate statistics
    def calculate_stats(values):
        if not values:
            return 0, 0, 0
        return max(values), np.mean(values), np.std(values)
    
    # Calculate statistics for each metric
    reward_best, reward_avg, reward_std = calculate_stats(episode_rewards)
    steps_best, steps_avg, steps_std = calculate_stats(episode_steps)
    exp_best, exp_avg, exp_std = calculate_stats(exploration_percentages)
    rev_best, rev_avg, rev_std = calculate_stats(revisits)
    unique_best, unique_avg, unique_std = calculate_stats(unique_states)
    solve_best, solve_avg, solve_std = calculate_stats(steps_to_solve)
    
    # Print aggregate results
    print("\n" + "="*50)
    print("Test Results Summary:")
    print(f"  Success Rate: {success_count/n_episodes*100:.2f}%")
    print("\nRewards:")
    print(f"  Best: {reward_best:.2f}")
    print(f"  Average: {reward_avg:.2f} ± {reward_std:.2f}")
    
    print("\nSteps:")
    print(f"  Best: {steps_best}")
    print(f"  Average: {steps_avg:.2f} ± {steps_std:.2f}")
    
    print("\nExploration:")
    print(f"  Best: {exp_best:.2f}%")
    print(f"  Average: {exp_avg:.2f}% ± {exp_std:.2f}%")
    
    print("\nRevisits:")
    print(f"  Best: {rev_best}")
    print(f"  Average: {rev_avg:.2f} ± {rev_std:.2f}")
    
    print("\nUnique States:")
    print(f"  Best: {unique_best}")
    print(f"  Average: {unique_avg:.2f} ± {unique_std:.2f}")
    
    if steps_to_solve:
        print("\nSteps to Solve (successful episodes only):")
        print(f"  Best: {solve_best}")
        print(f"  Average: {solve_avg:.2f} ± {solve_std:.2f}")
    print("="*50)
    
    # Return all metrics
    test_results = {
        'success_rate': success_count/n_episodes*100,
        'rewards': {
            'best': reward_best,
            'average': reward_avg,
            'std': reward_std
        },
        'steps': {
            'best': steps_best,
            'average': steps_avg,
            'std': steps_std
        },
        'exploration': {
            'best': exp_best,
            'average': exp_avg,
            'std': exp_std
        },
        'revisits': {
            'best': rev_best,
            'average': rev_avg,
            'std': rev_std
        },
        'unique_states': {
            'best': unique_best,
            'average': unique_avg,
            'std': unique_std
        },
        'steps_to_solve': {
            'best': solve_best,
            'average': solve_avg,
            'std': solve_std
        } if steps_to_solve else None
    }
    
    return test_results


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
    parser.add_argument("--agent_update_frequency", type=int, default=1000,
                        help="How often to update agent networks")
    parser.add_argument("--target_update_frequency", type=int, default=1000,
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
            learning_starts=args.learning_starts,
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
            experiment_name=args.experiment_name,
            obs_type=args.obs_type,
            pob_size=args.pob_size,
            obstacle_ratio=args.obstacle_ratio,
            log_dir=args.log_dir
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
            pob_size=args.pob_size,
            log_dir=args.log_dir
        )
