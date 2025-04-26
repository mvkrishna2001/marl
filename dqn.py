# dqn_agent.py
import argparse
import os
import random
import time
from distutils.util import strtobool
import copy
from tqdm import tqdm
import logging
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import skrl components
from skrl.agents.torch.dqn import DQN as SKRL_DQN
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model

# Import gym_maze environment
import gym_maze


# Define Q-network architecture
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
        states = inputs["states"].float()  # Ensure float32 dtype
        if role == "target_q_network":
            return self.target_layers(states), {}
        return self.layers(states), {}
    
    def update_target_network(self):
        """Update target network parameters"""
        self.target_layers.load_state_dict(self.layers.state_dict())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="dqn_maze",
                        help="Name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, `torch.backends.cudnn.deterministic=True`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="If toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="If toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
                        help="The wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="The entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Whether to capture videos of the agent performances")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory where logs and results will be stored")
    
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="Learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000000,
                        help="Size of the replay buffer")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
                        help="The timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for sampling from replay buffer")
    parser.add_argument("--start-e", type=float, default=1,
                        help="Start epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
                        help="End epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="Fraction of `total-timesteps` it takes from start-e to end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
                        help="Timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1,
                        help="The agent's model update frequency")
    
    # Maze environment specific arguments
    parser.add_argument("--maze-size", type=int, default=10,
                        help="Size of the maze grid")
    parser.add_argument("--observation-type", type=str, default="full", choices=["full", "partial"],
                        help="Type of observation (full or partial)")
    parser.add_argument("--pob-size", type=int, default=1,
                        help="Size of the partial observable window")
    
    args = parser.parse_args()
    return args


def make_env(args):
    from gym_maze.envs import MazeEnv
    from gym_maze.envs.generators import RandomMazeGenerator, RandomBlockMazeGenerator
    
    # Create maze generator
    # maze_generator = RandomMazeGenerator(args.maze_size, args.maze_size)
    maze_generator = RandomBlockMazeGenerator(args.maze_size, obstacle_ratio=0.1)  # Use 10% obstacle ratio
    
    # Create maze environment
    env = MazeEnv(
        maze_generator=maze_generator,
        pob_size=args.pob_size,
        obs_type=args.observation_type,
        render_trace=True,
    )
    
    # Wrap environment for compatibility
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{args.exp_name}")
    
    return env


def train_dqn(args):
    # Set up logging
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create directories for checkpoints
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Create writer for logging - ensure only one writer instance
    if 'writer' in locals():
        writer.close()
    writer = SummaryWriter(log_dir)
    
    # Create environment
    env = make_env(args)
    logger.info(f"Created maze environment with size {args.maze_size}x{args.maze_size}")
    logger.info(f"Observation type: {args.observation_type}, POB size: {args.pob_size}")
    
    # Extract state dimensions and action space
    if args.observation_type == "full":
        state_dim = env.unwrapped.maze_size[0] * env.unwrapped.maze_size[1]
        flat_observation = True
    else:  # partial
        state_dim = (args.pob_size * 2 + 1) ** 2
        flat_observation = True
    
    action_dim = env.action_space.n
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create Q-networks
    online_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    logger.info("Created and initialized Q-networks")
    
    # Configure DQN optimizer
    optimizer = optim.Adam(online_net.parameters(), lr=args.learning_rate)
    
    # Configure memory
    memory = RandomMemory(
        memory_size=args.buffer_size,
        num_envs=1,
        device=device
    )
    logger.info(f"Created replay buffer with size {args.buffer_size}")
    
    # Configure agent using skrl's DQN
    agent = SKRL_DQN(
        models={
            "q_network": online_net,
            "target_q_network": target_net
        },
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg={
            "random_timesteps": args.learning_starts,
            "learning_starts": args.learning_starts,
            "exploration": {
                "initial_epsilon": args.start_e,
                "final_epsilon": args.end_e,
                "timesteps": int(args.exploration_fraction * args.total_timesteps)
            },
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "target_update_interval": args.target_network_frequency,
            "update_interval": args.train_frequency,
            "gradient_steps": 1,
            "learning_rate": args.learning_rate,
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": state_dim, "device": device},
            "experiment": {
                "directory": "logs",
                "experiment_name": run_name,
                "write_interval": 100,
                "checkpoint_interval": 10000
            }
        }
    )
    
    # Initialize agent
    agent.init()
    agent.writer = writer
    logger.info("Initialized DQN agent")
    
    # Training loop
    obs, _ = env.reset(seed=args.seed)
    if flat_observation:
        obs = obs.flatten()
    obs = obs.astype(np.float32)  # Convert observation to float32
    
    # Tracking variables
    global_step = 0
    episode_rewards = []
    episode_lengths = []
    episode_exploration = []  # Track exploration percentage per episode
    episode_count = 0
    episode_reward = 0
    episode_length = 0
    episode_explored = 0  # Track exploration for current episode
    
    # Create progress bar
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    
    for global_step in range(args.total_timesteps):
        # Get action from the agent
        action = agent.act(torch.from_numpy(obs).view(1, -1).to(device), timestep=global_step, timesteps=args.total_timesteps)[0]
        action = action.item()  # get scalar value
        
        # Execute action in environment
        next_obs, reward, done, truncated, info = env.step(action)
        if flat_observation:
            next_obs = next_obs.flatten()
        next_obs = next_obs.astype(np.float32)  # Convert next observation to float32
        
        # Update episode statistics
        episode_reward += reward
        episode_length += 1
        episode_explored = info.get('exploration_percentage', 0)  # Get exploration percentage
        
        # Record transition
        agent.record_transition(
            states=torch.tensor(obs, dtype=torch.float32).unsqueeze(0),
            actions=torch.tensor(action, dtype=torch.long).unsqueeze(0),
            rewards=torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            next_states=torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0),
            terminated=torch.tensor(done, dtype=torch.bool).unsqueeze(0),
            truncated=torch.tensor(False, dtype=torch.bool).unsqueeze(0),
            infos={},
            timestep=global_step,
            timesteps=args.total_timesteps
        )
        
        # Update agent
        agent.post_interaction(timestep=global_step, timesteps=args.total_timesteps)
        
        # Log additional metrics
        if global_step % 100 == 0:
            # Log Q-values
            with torch.no_grad():
                q_values = online_net.compute({"states": torch.from_numpy(obs).float().unsqueeze(0).to(device)})[0]
                writer.add_scalar("charts/q_values", q_values.mean().item(), global_step)
            
            # Log loss if available from the agent's latest update
            if hasattr(agent, 'loss'):
                writer.add_scalar("charts/loss", agent.loss, global_step)
            
            # Log exploration percentage
            writer.add_scalar("charts/exploration_percentage", episode_explored, global_step)
        
        # Update observation
        obs = next_obs
        
        # Handle episode completion
        if done:
            # Reset environment
            obs, _ = env.reset()
            if flat_observation:
                obs = obs.flatten()
            
            # Log episode statistics
            episode_count += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_exploration.append(episode_explored)  # Store final exploration percentage
            
            # Log to tensorboard
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            writer.add_scalar("charts/episode_exploration", episode_explored, global_step)
            writer.flush()  # Force write to disk
            
            # Log to console
            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                avg_exploration = np.mean(episode_exploration[-10:])
                logger.info(f"Episode {episode_count} - Average Reward (last 10): {avg_reward:.2f}, Average Length: {avg_length:.2f}, Average Exploration: {avg_exploration:.2f}%")
            
            # Reset episode statistics
            episode_reward = 0
            episode_length = 0
            episode_explored = 0
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "episode": episode_count,
            "reward": episode_reward,
            "length": episode_length,
            "explored": f"{episode_explored:.1f}%"
        })
    
    # Close progress bar
    pbar.close()
    
    # Save final model
    torch.save(online_net.state_dict(), f"{log_dir}/final_model_{run_name}.pt")
    logger.info(f"Saved final model to {log_dir}/final_model_{run_name}.pt")
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Log final statistics
    logger.info(f"Training completed after {args.total_timesteps} timesteps")
    logger.info(f"Total episodes: {episode_count}")
    logger.info(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    # Generate plots after writer is closed
    from plotting import plot_training_metrics
    plot_training_metrics(log_dir, save_path=os.path.join(log_dir, "training_metrics.png"))


if __name__ == "__main__":
    args = parse_args()
    train_dqn(args)