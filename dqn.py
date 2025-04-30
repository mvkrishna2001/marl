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
from pathlib import Path

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
    parser.add_argument("--log-dir", type=str, default="/home/mila/m/munjuluv/scratch/logs/manual_triggers/",
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
    
    # Training and testing options
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Whether to train the agent or just test")
    parser.add_argument("--test", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Whether to test the agent after training")
    parser.add_argument("--test-episodes", type=int, default=5,
                        help="Number of episodes to run during testing")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to pre-trained model to load (for testing only)")
    parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Whether to render the environment during testing")
    
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
    # env = gym.wrappers.RecordEpisodeStatistics(env)
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
    nested_obs, _ = env.reset(seed=args.seed)
    obs = nested_obs[0]  # only one agent for DQN impl.
    if flat_observation:
        obs = obs.flatten()
    obs = obs.astype(np.float32)  # Convert observation to float32
    
    # Tracking variables
    global_step = 0
    episode_count = 0
    episode_reward = 0
    episode_length = 0
    episode_explored = 0  # Track exploration for current episode
    
    # Performance tracking
    total_solved = 0
    best_reward = float('-inf')
    best_exploration = 0
    
    # Create progress bar
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    
    for global_step in range(args.total_timesteps):
        # Get action from the agent
        action = agent.act(torch.from_numpy(obs).view(1, -1).to(device), timestep=global_step, timesteps=args.total_timesteps)[0]
        action = action.item()  # get scalar value
        
        # Execute action in environment
        unflattened_next_obs, unflattened_reward, unflattened_done, truncated, info = env.step([action])    # [action] because env.step() expects a list of actions.
        next_obs = unflattened_next_obs[0]
        reward = unflattened_reward[0]
        done = unflattened_done[0]
        if flat_observation:
            next_obs = next_obs.flatten()
        next_obs = next_obs.astype(np.float32)  # Convert next observation to float32
        
        # Update episode statistics
        episode_reward += reward
        episode_length += 1
        
        # Get maze metrics
        exploration_percentage = info.get('exploration_percentage', 0)
        total_revisits = info.get('total_revisits', 0)
        unique_states_visited = info.get('unique_states_visited', 0)
        maze_solved = info.get('maze_solved', False)
        
        # Log metrics at each step to show progress over time
        writer.add_scalar("charts/exploration", exploration_percentage, global_step)
        writer.add_scalar("charts/revisits", total_revisits, global_step)
        writer.add_scalar("charts/unique_states", unique_states_visited, global_step)
        
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
                writer.add_scalar("Loss / Q-network loss", agent.loss, global_step)
            
            # Log epsilon
            epsilon = agent.cfg["exploration"]["final_epsilon"]
            if global_step < agent.cfg["exploration"]["timesteps"]:
                # Calculate current epsilon based on linear decay
                epsilon = agent.cfg["exploration"]["initial_epsilon"] - (agent.cfg["exploration"]["initial_epsilon"] - 
                              agent.cfg["exploration"]["final_epsilon"]) * (global_step / agent.cfg["exploration"]["timesteps"])
            writer.add_scalar("Exploration / Exploration epsilon", epsilon, global_step)
        
        # Update observation
        obs = next_obs
        
        # Safety check - terminate very long episodes
        max_episode_safety_limit = 2000
        if episode_length >= max_episode_safety_limit:
            logger.info(f"Terminating episode after {episode_length} steps (safety limit)")
            done = True
        
        # Handle episode completion
        if done:
            # Log if maze was solved and steps to solve
            if maze_solved:
                total_solved += 1
                steps_to_solve = info.get('steps_to_solve', episode_length)
                writer.add_scalar("charts/steps_to_solve", steps_to_solve, global_step)
                writer.add_scalar("charts/solved", 1, global_step)  # Binary flag for solving
            else:
                writer.add_scalar("charts/solved", 0, global_step)  # Not solved
            
            # Calculate success rate
            episode_count += 1
            success_rate = (total_solved / episode_count) * 100
            
            # Track best metrics
            if exploration_percentage > best_exploration:
                best_exploration = exploration_percentage
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # Log episode statistics
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            writer.add_scalar("charts/success_rate", success_rate, global_step)
            writer.add_scalar("charts/best_exploration", best_exploration, global_step)
            writer.add_scalar("charts/best_reward", best_reward, global_step)
            
            # Log to console
            if episode_count % 10 == 0:
                logger.info(f"Episode {episode_count} - Step {global_step}")
                logger.info(f"  Reward: {episode_reward:.2f} (best: {best_reward:.2f})")
                logger.info(f"  Steps: {episode_length}")
                logger.info(f"  Exploration: {exploration_percentage:.2f}% (best: {best_exploration:.2f}%)")
                logger.info(f"  Success Rate: {success_rate:.2f}%")
                logger.info(f"  Unique states visited: {unique_states_visited}")
                logger.info(f"  Revisits: {total_revisits}")
                if maze_solved:
                    logger.info(f"  Solved in {steps_to_solve} steps!")
                
                # Save model checkpoint periodically
                if episode_count % 100 == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_ep{episode_count}.pt")
                    torch.save(online_net.state_dict(), checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Reset environment
            nested_obs, _ = env.reset(seed=args.seed)
            obs = nested_obs[0]  # only one agent for DQN impl.
            if flat_observation:
                obs = obs.flatten()
            
            # Reset episode statistics
            episode_reward = 0
            episode_length = 0
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "episode": episode_count,
            "reward": episode_reward,
            "exploration": f"{exploration_percentage:.1f}%"
        })
    
    # Close progress bar
    pbar.close()
    
    # Save final model
    torch.save(online_net.state_dict(), f"{log_dir}/final_model_{run_name}.pt")
    logger.info(f"Saved final model to {log_dir}/final_model_{run_name}.pt")
    
    # Save performance metrics
    performance = {
        "total_episodes": episode_count,
        "total_steps": global_step,
        "total_solved": total_solved,
        "success_rate": success_rate,
        "best_reward": best_reward,
        "best_exploration": best_exploration
    }
    
    with open(os.path.join(log_dir, "performance.json"), "w") as f:
        json.dump(performance, f, indent=4)
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Generate plots using plotting module
    from plotting import plot_training_metrics
    plot_path = plot_training_metrics(log_dir)
    logger.info(f"Training metrics plots saved to: {plot_path}")
    
    # Return trained agent and path
    return agent, log_dir


def test_dqn(agent=None, model_path=None, maze_size=15, observation_type="full", pob_size=1, 
           n_episodes=5, device="cpu", render=True):
    """
    Test a trained DQN agent on the maze environment.
    
    Args:
        agent: Trained DQN agent (if None, will load from model_path)
        model_path: Path to a saved model file
        maze_size: Size of the maze
        observation_type: Type of observation space ("full" or "partial")
        pob_size: Size of partial observation window if using partial obs
        n_episodes: Number of episodes to test
        device: Device to run inference on
        render: Whether to render the environment during testing
        
    Returns:
        scores: Dictionary of test performance metrics
    """
    # Set device
    device = torch.device(device)
    from gym_maze.envs import MazeEnv
    from gym_maze.envs.generators import RandomMazeGenerator, RandomBlockMazeGenerator    
    # Create environment
    env = MazeEnv(
        maze_generator=RandomBlockMazeGenerator(maze_size, obstacle_ratio=0.1),
        pob_size=pob_size,
        obs_type=observation_type,
        render_trace=True,
        live_display=render
    )
    
    # Reset to get observation shape
    nested_obs, _ = env.reset()
    obs = nested_obs[0]
    if observation_type in ["full", "partial"]:
        flat_observation = True
        if observation_type == "full":
            state_dim = maze_size * maze_size
        else:  # partial
            state_dim = (pob_size * 2 + 1) ** 2
    else:
        raise ValueError(f"Unsupported observation type: {observation_type}")
    
    # Flatten observation if needed
    if flat_observation:
        obs = obs.flatten()
    
    # Get action dimension
    action_dim = env.action_space.n
    
    print(f"Testing with observation dimension: {state_dim}, action dimension: {action_dim}")
    
    # Create or load the agent
    if agent is None and model_path is not None:
        # Create networks
        q_network = QNetwork(state_dim, action_dim).to(device)
        target_network = QNetwork(state_dim, action_dim).to(device)
        
        # Load weights
        q_network.load_state_dict(torch.load(model_path, map_location=device))
        target_network.load_state_dict(q_network.state_dict())
        
        # Create dummy memory (not used for testing)
        memory = RandomMemory(memory_size=10, device=device)
        
        # Create agent with simplified configuration
        agent = SKRL_DQN(
            models={
                "q_network": q_network,
                "target_q_network": target_network
            },
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            cfg={
                "exploration": {
                    "initial_epsilon": 0.0,  # No exploration during testing
                    "final_epsilon": 0.0,
                    "timesteps": 1
                }
            }
        )
        agent.init()
        
        # Disable state preprocessing
        agent._state_preprocessor = None
    elif agent is None:
        raise ValueError("Either agent or model_path must be provided")
    
    # Run test episodes
    scores = []
    success_count = 0
    steps_to_solve = []
    exploration_percentages = []
    revisits_list = []
    unique_states_list = []
    
    for episode in range(1, n_episodes + 1):
        # Reset environment
        nested_obs, _ = env.reset()
        obs = nested_obs[0]
        if flat_observation:
            obs = obs.flatten()
        obs = obs.astype(np.float32)
        
        # Initialize episode variables
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\nStarting test episode {episode}/{n_episodes}")
        
        # Episode loop
        while not done:
            # Get action using the trained agent (no exploration)
            # Move observation to the correct device before passing to agent
            obs_tensor = torch.from_numpy(obs).view(1, -1).to(device)
            action = agent.act(obs_tensor, timestep=0, timesteps=1)[0]
            action = action.item()
            
            # Take action in environment
            next_nested_obs, rewards, dones, truncated, info = env.step([action])
            next_obs = next_nested_obs[0]
            reward = rewards[0]
            done = dones[0] or truncated[0]
            
            # Flatten if needed
            if flat_observation:
                next_obs = next_obs.flatten()
            next_obs = next_obs.astype(np.float32)
            
            # Update episode statistics
            episode_reward += reward
            episode_steps += 1
            
            # Get current metrics
            if episode_steps % 100 == 0:
                print(f"  Step {episode_steps}, Reward so far: {episode_reward:.2f}")
                print(f"  Exploration: {info.get('exploration_percentage', 0):.2f}%, "
                      f"Unique states: {info.get('unique_states_visited', 0)}")
            
            # Update observation
            obs = next_obs
            
            # Safety check - terminate very long episodes
            if episode_steps >= 5000:
                print(f"  Terminating episode after {episode_steps} steps (safety limit)")
                break
        
        # Episode complete
        # Record metrics
        maze_solved = info.get('maze_solved', False)
        exploration_pct = info.get('exploration_percentage', 0)
        revisits = info.get('total_revisits', 0)
        unique_states = info.get('unique_states_visited', 0)
        
        scores.append(episode_reward)
        exploration_percentages.append(exploration_pct)
        revisits_list.append(revisits)
        unique_states_list.append(unique_states)
        
        if maze_solved:
            success_count += 1
            steps_to_complete = info.get('steps_to_solve', episode_steps)
            steps_to_solve.append(steps_to_complete)
            solved_text = f"SOLVED in {steps_to_complete} steps"
        else:
            solved_text = "NOT SOLVED"
        
        # Print episode summary
        print(f"Episode {episode} finished: {solved_text}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_steps}")
        print(f"  Exploration: {exploration_pct:.2f}%")
        print(f"  Revisits: {revisits}")
        print(f"  Unique states: {unique_states}")
    
    # Calculate summary statistics
    success_rate = (success_count / n_episodes) * 100
    avg_reward, std_reward = np.mean(scores), np.std(scores)
    avg_exploration, std_exploration = np.mean(exploration_percentages), np.std(exploration_percentages)
    avg_revisits, std_revisits = np.mean(revisits_list), np.std(revisits_list)
    avg_unique_states, std_unique_states = np.mean(unique_states_list), np.std(unique_states_list)
    
    # Calculate average steps to solve if there were successful episodes
    if steps_to_solve:
        avg_steps_to_solve, std_steps_to_solve = np.mean(steps_to_solve), np.std(steps_to_solve)
    else:
        avg_steps_to_solve = None
    
    # Print summary
    print("\n" + "="*50)
    print(f"Test Results ({n_episodes} episodes):")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average Exploration: {avg_exploration:.2f}% ± {std_exploration:.2f}%")
    print(f"  Average Revisits: {avg_revisits:.2f} ± {std_revisits:.2f}")
    print(f"  Average Unique States: {avg_unique_states:.2f} ± {std_unique_states:.2f}")
    if avg_steps_to_solve is not None:
        print(f"  Average Steps to Solve: {avg_steps_to_solve:.2f} ± {std_steps_to_solve:.2f}")
    print("="*50)
    
    # Close environment
    env.close()
    
    # Return results
    test_results = {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_exploration': avg_exploration,
        'avg_revisits': avg_revisits,
        'avg_unique_states': avg_unique_states,
        'avg_steps_to_solve': avg_steps_to_solve,
        'episode_rewards': scores,
        'exploration_percentages': exploration_percentages,
        'revisits': revisits_list,
        'unique_states': unique_states_list,
        'steps_to_solve': steps_to_solve
    }
    
    return test_results


if __name__ == "__main__":
    args = parse_args()
    
    # Train the agent
    if getattr(args, 'train', True):
        agent, log_dir = train_dqn(args)
        print(f"Training completed. Model saved to {log_dir}")
    
    # Test the agent if requested
    if getattr(args, 'test', True):
        if not getattr(args, 'train', True):
            # If we didn't train, we need a model path
            model_path = args.model_path if hasattr(args, 'model_path') else None
            if not model_path:
                # Try to find the most recent model
                log_dirs = sorted(Path("logs").glob("dqn_*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if log_dirs:
                    model_path = str(next(log_dirs[0].glob("final_model_*.pt"), None))
                    if not model_path:
                        print("No model found for testing. Please specify --model_path")
                        exit(1)
                    print(f"Using most recent model: {model_path}")
            
            test_results = test_dqn(
                model_path=model_path,
                maze_size=args.maze_size,
                observation_type=args.observation_type,
                pob_size=args.pob_size,
                n_episodes=getattr(args, 'test_episodes', 5),
                render=getattr(args, 'render', True)
            )
        else:
            # Use the agent we just trained
            test_results = test_dqn(
                agent=agent,
                maze_size=args.maze_size,
                observation_type=args.observation_type,
                pob_size=args.pob_size,
                n_episodes=getattr(args, 'test_episodes', 5),
                render=getattr(args, 'render', True)
            )
        
        # Save test results
        if 'log_dir' in locals():
            import json
            with open(os.path.join(log_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
