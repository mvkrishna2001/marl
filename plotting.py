import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Optional
from tensorboard.backend.event_processing import event_accumulator
import json
import argparse
import sys

def plot_training_metrics(log_dir: str, save_path: Optional[str] = None):
    """
    Plot training metrics from a single run.
    
    Args:
        log_dir: Directory containing the tensorboard logs
        save_path: Optional path to save the plot
    """
    # Read tensorboard logs - get the most recent events file
    try:
        events_files = sorted(Path(log_dir).glob("events.out.tfevents.*"), 
                            key=lambda x: x.stat().st_mtime, reverse=True)
        if not events_files:
            print(f"No tensorboard files found in {log_dir}")
            return
        events_file = events_files[0]  # Use most recent file
        print(f"Reading events from: {events_file}")
    except Exception as e:
        print(f"Error reading tensorboard files: {e}")
        return
        
    ea = event_accumulator.EventAccumulator(str(events_file))
    ea.Reload()
    
    # Extract metrics
    metrics = {}
    available_tags = ea.Tags()['scalars']
    print(f"Available tags: {available_tags}")
    
    for tag in available_tags:
        events = ea.Scalars(tag)
        if events:  # Only add if we have data
            metrics[tag] = np.array([(x.step, x.value) for x in events])
            print(f"Loaded {len(events)} points for {tag}")
    
    if not metrics:
        print("No metrics found in the logs")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot episode reward
    if 'charts/episode_reward' in metrics:
        steps, rewards = metrics['charts/episode_reward'].T
        print(f"Plotting rewards: {len(rewards)} points, range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
        axes[0, 0].plot(steps, rewards, 'b-', alpha=0.3, label='Episode Reward')
        # Add moving average
        window_size = min(100, len(rewards))
        if window_size > 0:
            rewards_ma = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
            axes[0, 0].plot(steps, rewards_ma, 'r-', linewidth=2, label=f'{window_size}-episode MA')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        print("No reward data found")

    # Plot exploration percentage
    if 'Exploration / Exploration epsilon' in metrics:
        steps, exploration = metrics['Exploration / Exploration epsilon'].T
        print(f"Plotting exploration: {len(exploration)} points, range: [{np.min(exploration):.2f}, {np.max(exploration):.2f}]")
        axes[0, 1].plot(steps, exploration, 'g-', alpha=0.3, label='Exploration %')
        # Add moving average
        window_size = min(100, len(exploration))
        if window_size > 0:
            exploration_ma = pd.Series(exploration).rolling(window=window_size, min_periods=1).mean()
            axes[0, 1].plot(steps, exploration_ma, 'r-', linewidth=2, label=f'{window_size}-episode MA')
        axes[0, 1].set_title('Maze Exploration')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Exploration fraction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)  # Set y-axis to percentage scale
    else:
        print("No exploration data found")
    
    # Plot Q-values (if available)
    if 'charts/q_values' in metrics:
        steps, q_values = metrics['charts/q_values'].T
        print(f"Plotting Q-values: {len(q_values)} points, range: [{np.min(q_values):.2f}, {np.max(q_values):.2f}]")
        axes[1, 0].plot(steps, q_values, 'g-', alpha=0.7, label='Average Q-value')
        axes[1, 0].set_title('Q-values Evolution')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Average Q-value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        print("No Q-value data found")
    
    # Plot loss (if available)
    if 'Loss / Q-network loss' in metrics:
        steps, losses = metrics['Loss / Q-network loss'].T
        print(f"Plotting losses: {len(losses)} points, range: [{np.min(losses):.2f}, {np.max(losses):.2f}]")
        axes[1, 1].plot(steps, losses, 'm-', alpha=0.7, label='Training Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        print("No loss data found")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    if save_path:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_runs(run_dirs: List[str], labels: List[str], save_path: Optional[str] = None):
    """
    Compare metrics across multiple runs.
    
    Args:
        run_dirs: List of directories containing tensorboard logs
        labels: List of labels for each run
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for run_dir, label in zip(run_dirs, labels):
        events_file = next(Path(run_dir).glob("events.out.tfevents.*"))
        ea = event_accumulator.EventAccumulator(str(events_file))
        ea.Reload()
        
        if 'charts/episode_reward' in ea.Tags()['scalars']:
            steps, rewards = np.array([(x.step, x.value) for x in ea.Scalars('charts/episode_reward')]).T
            # Calculate moving average
            rewards_ma = pd.Series(rewards).rolling(window=100).mean()
            plt.plot(steps, rewards_ma, label=label)
    
    plt.title('Comparison of Training Progress')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward (100-episode MA)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_hyperparameter_sweep(sweep_dir: str, save_path: Optional[str] = None):
    """Plot aggregated results from hyperparameter sweep experiments."""
    results = []
    
    # Collect results from all experiment directories
    for exp_dir in Path(sweep_dir).glob("dqn_lr*"):
        if not exp_dir.is_dir():
            continue
        
        # Parse hyperparameters from directory name
        exp_name = exp_dir.name
        params = dict(param.split('_') for param in exp_name.split('_')[1:])
        
        # Read tensorboard logs
        try:
            events_file = next(exp_dir.glob("events.out.tfevents.*"))
            ea = event_accumulator.EventAccumulator(str(events_file))
            ea.Reload()
            
            if 'charts/episode_reward' in ea.Tags()['scalars']:
                rewards = np.array([x.value for x in ea.Scalars('charts/episode_reward')])
                final_reward = np.mean(rewards[-100:])  # Average of last 100 episodes
                results.append({**params, 'final_reward': final_reward})
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
    
    if not results:
        print("No results found to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Comparison', fontsize=16)
    
    # Plot learning rate comparison
    sns.boxplot(data=df, x='lr', y='final_reward', ax=axes[0, 0])
    axes[0, 0].set_title('Effect of Learning Rate')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Final Reward')
    
    # Plot batch size comparison
    sns.boxplot(data=df, x='batch', y='final_reward', ax=axes[0, 1])
    axes[0, 1].set_title('Effect of Batch Size')
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Final Reward')
    
    # Plot observation type comparison
    sns.boxplot(data=df, x='obs', y='final_reward', ax=axes[1, 0])
    axes[1, 0].set_title('Effect of Observation Type')
    axes[1, 0].set_xlabel('Observation Type')
    axes[1, 0].set_ylabel('Final Reward')
    
    # Plot gamma comparison
    sns.boxplot(data=df, x='gamma', y='final_reward', ax=axes[1, 1])
    axes[1, 1].set_title('Effect of Discount Factor')
    axes[1, 1].set_xlabel('Gamma')
    axes[1, 1].set_ylabel('Final Reward')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics and hyperparameter sweeps')
    parser.add_argument('--mode', choices=['single', 'sweep'], default='single',
                      help='Plot mode: single run or hyperparameter sweep')
    parser.add_argument('--dir', type=str, default='logs',
                      help='Directory containing logs')
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Find most recent log directory if not specified
        if args.dir == 'logs':
            log_dirs = sorted(Path("logs").glob("dqn_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not log_dirs:
                print("No log directories found")
                sys.exit(1)
            log_dir = str(log_dirs[0])
        else:
            log_dir = args.dir
        print(f"Plotting single run from: {log_dir}")
        plot_training_metrics(log_dir, save_path=os.path.join(log_dir, "training_metrics.png"))
    else:
        print(f"Plotting hyperparameter sweep from: {args.dir}")
        plot_hyperparameter_sweep(args.dir, save_path=os.path.join(args.dir, "hyperparameter_comparison.png"))
