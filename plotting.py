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
    
    # Try to load config if available
    config = {}
    try:
        config_path = Path(log_dir) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Could not load config: {e}")
    
    # Create figure with subplots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Set title based on experiment type
    algo_name = "QMIX" if "qmix" in log_dir.lower() else "DQN"
    
    # Extract experiment details for title
    maze_size = config.get('maze_size', 'N/A')
    num_agents = config.get('num_agents', 1)
    title = f"{algo_name} Training Metrics - {maze_size}×{maze_size} Maze, {num_agents} Agent{'s' if num_agents > 1 else ''}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Define plot colors for consistency
    colors = {
        'reward': '#1f77b4',      # Blue
        'reward_avg': '#ff7f0e',  # Orange
        'loss': '#d62728',        # Red
        'exploration': '#9467bd', # Purple
        'success': '#2ca02c',     # Green
        'revisits': '#8c564b',    # Brown
        'steps': '#17becf',       # Cyan
        'q_value': '#7f7f7f',     # Gray
    }
    
    # 1. Plot episode rewards
    reward_tags = [tag for tag in metrics if 'reward' in tag.lower() and 'best' not in tag.lower()]
    if reward_tags:
        # Prefer episode reward if available
        reward_tag = next((tag for tag in reward_tags if 'episode_reward' in tag), reward_tags[0])
        steps, rewards = metrics[reward_tag].T
        print(f"Plotting rewards: {len(rewards)} points, range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
        
        # Raw rewards as scatter
        axes[0, 0].scatter(steps, rewards, color=colors['reward'], alpha=0.3, s=10, label='Episode Reward')
        
        # Add moving average
        window_size = min(100, max(1, len(rewards) // 10))  # Adaptive window size
        if window_size > 0:
            rewards_ma = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
            axes[0, 0].plot(steps, rewards_ma, color=colors['reward_avg'], linewidth=2, 
                          label=f'{window_size}-step MA')
        
        # If we have best reward, add as reference line
        if 'charts/best_reward' in metrics:
            best_steps, best_rewards = metrics['charts/best_reward'].T
            # Take the final best reward
            if len(best_rewards) > 0:
                final_best = best_rewards[-1]
                axes[0, 0].axhline(y=final_best, color='green', linestyle='--', 
                                 alpha=0.7, label=f'Best: {final_best:.1f}')
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        print("No reward data found")
        axes[0, 0].text(0.5, 0.5, 'No reward data available', 
                      ha='center', va='center', transform=axes[0, 0].transAxes)

    # 2. Plot exploration and success rate
    exploration_tags = [tag for tag in metrics if 'exploration' in tag.lower() and 'epsilon' not in tag.lower()]
    success_tags = [tag for tag in metrics if 'success' in tag.lower() or 'solved' in tag.lower()]
    
    if exploration_tags or 'Exploration / Exploration epsilon' in metrics:
        if exploration_tags:
            # Prefer exploration percentage if available
            exploration_tag = next((tag for tag in exploration_tags if 'percentage' in tag or 'chart' in tag), exploration_tags[0])
            steps, exploration = metrics[exploration_tag].T
        elif 'Exploration / Exploration epsilon' in metrics:
            # For DQN, we have epsilon instead of exploration percentage
            steps, exploration = metrics['Exploration / Exploration epsilon'].T
            # For epsilon, scale to percentage and invert (1.0 → 0%, 0.0 → 100%)
            exploration = (1.0 - exploration) * 100
        
        print(f"Plotting exploration: {len(exploration)} points")
        
        # Fill area for exploration
        axes[0, 1].fill_between(steps, 0, exploration, color=colors['exploration'], alpha=0.2)
        
        # Plot exploration line
        axes[0, 1].plot(steps, exploration, color=colors['exploration'], 
                       linewidth=2, label='Exploration %')
        
        axes[0, 1].set_title('Maze Exploration')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].set_ylim(0, 105)  # Set y-axis to percentage scale with margin
        
        # Add reference line at 100%
        axes[0, 1].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        print("No exploration data found")
        axes[0, 1].text(0.5, 0.5, 'No exploration data available', 
                      ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. Plot loss and q-values
    loss_tags = [tag for tag in metrics if 'loss' in tag.lower()]
    q_value_tags = [tag for tag in metrics if 'q_value' in tag.lower() or 'q-value' in tag.lower()]
    
    ax = axes[1, 0]
    ax2 = None  # For possible twin axis
    
    if loss_tags:
        loss_tag = loss_tags[0]
        steps, losses = metrics[loss_tag].T
        print(f"Plotting loss: {len(losses)} points")
        
        # Handle any extreme outliers for better visualization
        if np.max(losses) > 1000:
            cap_value = np.percentile(losses, 95)  # Cap at 95th percentile
            losses = np.minimum(losses, cap_value)
        
        # Plot loss values
        ax.plot(steps, losses, color=colors['loss'], alpha=0.5, label='Training Loss')
        
        # Add moving average for cleaner trend
        window_size = min(50, max(1, len(losses) // 20))  # Adaptive window size
        if window_size > 0:
            loss_ma = pd.Series(losses).rolling(window=window_size, min_periods=1).mean()
            ax.plot(steps, loss_ma, color=colors['loss'], linewidth=2, label=f'Loss {window_size}-pt MA')
        
        ax.set_title('Training Loss & Q-values')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Add Q-values on twin axis if available
        if q_value_tags:
            q_tag = q_value_tags[0]
            q_steps, q_values = metrics[q_tag].T
            
            # Create twin axis for Q-values
            ax2 = ax.twinx()
            ax2.plot(q_steps, q_values, color=colors['q_value'], alpha=0.7, label='Q-values')
            
            # Add moving average
            window_size = min(50, max(1, len(q_values) // 20))
            if window_size > 0:
                q_ma = pd.Series(q_values).rolling(window=window_size, min_periods=1).mean()
                ax2.plot(q_steps, q_ma, color=colors['q_value'], linewidth=2, label=f'Q-value {window_size}-pt MA')
            
            ax2.set_ylabel('Average Q-value', color=colors['q_value'])
            ax2.tick_params(axis='y', labelcolor=colors['q_value'])
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)
        else:
            ax.legend()
    elif q_value_tags:
        # If we have only Q-values but no loss
        q_tag = q_value_tags[0]
        q_steps, q_values = metrics[q_tag].T
        ax.plot(q_steps, q_values, color=colors['q_value'], alpha=0.7, label='Q-values')
        
        # Add moving average
        window_size = min(50, max(1, len(q_values) // 20))
        if window_size > 0:
            q_ma = pd.Series(q_values).rolling(window=window_size, min_periods=1).mean()
            ax.plot(q_steps, q_ma, color=colors['q_value'], linewidth=2, label=f'Q-value {window_size}-pt MA')
        
        ax.set_title('Q-values Evolution')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Average Q-value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        print("No loss or Q-value data found")
        ax.text(0.5, 0.5, 'No loss or Q-value data available', 
              ha='center', va='center', transform=ax.transAxes)
    
    # 4. Plot maze completion metrics (revisits and steps to solve)
    revisit_tags = [tag for tag in metrics if 'revisit' in tag.lower()]
    steps_to_solve_tags = [tag for tag in metrics if 'steps_to_solve' in tag.lower() or 'solve' in tag.lower()]
    
    ax = axes[1, 1]
    ax2 = None  # For possible twin axis
    
    if revisit_tags:
        revisit_tag = revisit_tags[0]
        revisit_steps, revisits = metrics[revisit_tag].T
        print(f"Plotting revisits: {len(revisits)} points")
        
        # Plot revisits
        ax.scatter(revisit_steps, revisits, color=colors['revisits'], alpha=0.2, s=10, label='Revisits')
        
        # Add moving average for trend
        window_size = min(50, max(1, len(revisits) // 20))
        if window_size > 0:
            revisits_ma = pd.Series(revisits).rolling(window=window_size, min_periods=1).mean()
            ax.plot(revisit_steps, revisits_ma, color=colors['revisits'], 
                   linewidth=2, label=f'Revisits ({window_size}-pt MA)')
        
        ax.set_ylabel('Number of Revisits', color=colors['revisits'])
        ax.tick_params(axis='y', labelcolor=colors['revisits'])
        
        # Add steps to solve on twin axis if available
        if steps_to_solve_tags:
            steps_tag = steps_to_solve_tags[0]
            solve_steps, steps_counts = metrics[steps_tag].T
            
            # Filter out invalid entries (e.g. -1 for episodes without solution)
            valid_idx = steps_counts >= 0
            if np.any(valid_idx):
                solve_steps = solve_steps[valid_idx]
                steps_counts = steps_counts[valid_idx]
                
                # Create twin axis for steps to solve
                ax2 = ax.twinx()
                ax2.scatter(solve_steps, steps_counts, color=colors['steps'], 
                           alpha=0.5, s=20, label='Steps to Solve')
                
                # Add moving average if we have enough data points
                if len(steps_counts) >= 3:
                    window_size = min(10, max(1, len(steps_counts) // 5))
                    steps_ma = pd.Series(steps_counts).rolling(window=window_size, min_periods=1).mean()
                    ax2.plot(solve_steps, steps_ma, color=colors['steps'], 
                            linewidth=2, label=f'Steps ({window_size}-pt MA)')
                
                ax2.set_ylabel('Steps to Solve', color=colors['steps'])
                ax2.tick_params(axis='y', labelcolor=colors['steps'])
                
                # Set reasonable y-limits
                ax2.set_ylim(0, np.max(steps_counts) * 1.1)
                
                # Combined legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)
            else:
                ax.legend()
        else:
            ax.legend()
    elif steps_to_solve_tags:
        # If we have only steps to solve but no revisits
        steps_tag = steps_to_solve_tags[0]
        solve_steps, steps_counts = metrics[steps_tag].T
        
        # Filter out invalid entries
        valid_idx = steps_counts >= 0
        if np.any(valid_idx):
            solve_steps = solve_steps[valid_idx]
            steps_counts = steps_counts[valid_idx]
            
            ax.scatter(solve_steps, steps_counts, color=colors['steps'], 
                     alpha=0.5, s=20, label='Steps to Solve')
            
            # Add moving average if we have enough data points
            if len(steps_counts) >= 3:
                window_size = min(10, max(1, len(steps_counts) // 5))
                steps_ma = pd.Series(steps_counts).rolling(window=window_size, min_periods=1).mean()
                ax.plot(solve_steps, steps_ma, color=colors['steps'], 
                      linewidth=2, label=f'Steps ({window_size}-pt MA)')
            
            ax.set_title('Steps to Solve Maze')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Steps to Solve')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No successful maze solves recorded', 
                  ha='center', va='center', transform=ax.transAxes)
    else:
        print("No completion metrics found")
        ax.text(0.5, 0.5, 'No completion metrics available', 
              ha='center', va='center', transform=ax.transAxes)
    
    # Title and final touches
    if revisit_tags or steps_to_solve_tags:
        ax.set_title('Maze Completion Metrics')
        ax.set_xlabel('Timesteps')
        ax.grid(True, alpha=0.3)
    
    # Add a configuration summary if available
    if config:
        try:
            # Format key config parameters
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
            print(f"Could not add config caption: {e}")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        save_path = os.path.join(log_dir, "training_metrics.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics and hyperparameter sweeps')
    parser.add_argument('--dir', type=str, default='logs',
                      help='Directory containing logs')
    args = parser.parse_args()
    
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
