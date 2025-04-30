import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib import colors
from past.utils import old_div

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import torch
import os

from gym_maze.envs.maze import MazeEnv, SparseMazeEnv

"""Enhanced Maze Environment with improved visualization capabilities"""

class EnhancedMazeEnv(MazeEnv):
    """
    Enhanced Maze Environment with better visualization and rendering.
    Extends the original MazeEnv with improved visual features.
    """
    
    def __init__(self,
                 maze_generator,
                 pob_size=1,
                 action_type='VonNeumann',
                 obs_type='full',
                 live_display=False,
                 render_trace=False,
                 figsize=(12, 12),
                 dpi=100,
                 custom_colors=None):
        """
        Initialize the enhanced maze environment.
        
        Args:
            maze_generator: Maze generator instance
            pob_size: Size of the partial observation window
            action_type: Action space type ('VonNeumann' or 'Moore')
            obs_type: Observation type ('full' or 'partial')
            live_display: Whether to update display in real-time
            render_trace: Whether to render the agents' paths
            figsize: Figure size for rendering (width, height)
            dpi: DPI for rendering
            custom_colors: Dictionary of custom colors for different elements
        """
        # Initialize the parent class
        super().__init__(
            maze_generator=maze_generator,
            pob_size=pob_size,
            action_type=action_type,
            obs_type=obs_type,
            live_display=live_display,
            render_trace=render_trace
        )
        
        # Store enhanced rendering parameters
        self.figsize = figsize
        self.dpi = dpi
        
        # Define custom color schemes
        self.agent_colors = ['blue', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'lime', 'brown', 'pink']
        self.path_colors = ['lightblue', 'lightcoral', 'plum', 'bisque', 'lightcyan', 'violet', 'khaki', 'lightgreen', 'sandybrown', 'lightpink']
        
        # Override with any custom colors if provided
        if custom_colors:
            for key, value in custom_colors.items():
                setattr(self, key, value)
                
        # Text elements for rendering
        self.title_text = None
        self.timestamp_text = None
        self.goal_status_text = None
        
        # For tracking steps in visualization
        self.current_step = 0
        self.max_steps_displayed = 300  # Default max steps for display
        
        # Setup custom colormap
        self._setup_colormap()
    
    def _setup_colormap(self):
        """Setup custom colormap based on the number of agents"""
        # Base colors for empty space and walls
        colors_list = ['white', 'black']  # Empty space, wall
        
        # We don't know how many agents there will be yet, so we'll update this in reset()
        self.cmap = colors.ListedColormap(colors_list)
        self.bounds = list(range(len(colors_list) + 1))
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        # Constants for indexing
        self.EMPTY = 0
        self.WALL = 1
        self.AGENT_START_IDX = 2  # The starting index for agent colors
        # TRACE_START_IDX will be set in reset()
    
    def reset(self, num_agents=1, seed=None, options=None):
        """Reset the environment and update colormap for the number of agents"""
        observations, info = super().reset(num_agents=num_agents, seed=seed, options=options)
        
        # Reset step counter
        self.current_step = 0
        
        # Update colormap based on number of agents
        colors_list = ['white', 'black']  # Empty space, wall
        
        # Add agent/goal colors
        for i in range(min(num_agents, len(self.agent_colors))):
            colors_list.append(self.agent_colors[i])
        
        # Add path trace colors
        for i in range(min(num_agents, len(self.path_colors))):
            colors_list.append(self.path_colors[i])
        
        # Update colormap
        self.cmap = colors.ListedColormap(colors_list)
        self.bounds = list(range(len(colors_list) + 1))
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        # Set the trace start index
        self.TRACE_START_IDX = self.AGENT_START_IDX + num_agents
        
        return observations, info
    
    def step(self, actions):
        """Step the environment and update step counter"""
        observations, rewards, dones, truncated, info = super().step(actions)
        self.current_step += 1
        return observations, rewards, dones, truncated, info
    
    def _get_enhanced_obs(self):
        """Get enhanced observation with custom colors for agents, goals, and paths"""
        # Get the base maze
        obs = np.array(self.maze)
        
        # Add traces if enabled
        if self.render_trace:
            for i, trace in enumerate(self.traces):
                trace_value = self.TRACE_START_IDX + i
                for pos in trace[:-1]:  # Skip current position
                    # Only mark as trace if it's not a wall and not a goal
                    if obs[pos[0]][pos[1]] == self.EMPTY and not any(
                        np.array_equal(pos, goal) for goal in self.goal_states
                    ):
                        obs[pos[0]][pos[1]] = trace_value
        
        # Add goals with agent colors
        for i, goal in enumerate(self.goal_states):
            obs[goal[0]][goal[1]] = self.AGENT_START_IDX + i
        
        # Add agents (overriding other elements)
        for i, state in enumerate(self.states):
            obs[state[0]][state[1]] = self.AGENT_START_IDX + i
        
        return obs
    
    def render(self, mode='human', close=False, show_status=True):
        """
        Enhanced rendering function with better visualization.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            close: Whether to close the figure
            show_status: Whether to show status information (step count, goal status)
        
        Returns:
            fig: Figure if mode is 'human', RGB array if mode is 'rgb_array'
        """
        if close:
            plt.close()
            return
        
        # Get enhanced observation
        obs = self._get_enhanced_obs()
        
        # Create/update figure
        if not hasattr(self, 'fig') or self.fig is None:
            plt.close('all')  # Close any existing figures
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            self.ax.axis('off')
            
            # Add title
            self.ax.set_title("Maze Environment", fontsize=16, fontweight='bold')
            
            # Add status text if requested
            if show_status:
                # Step counter
                self.timestamp_text = self.ax.text(
                    0.02, 0.02, f"Step: {self.current_step}/{self.max_steps_displayed}", 
                    transform=self.ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
                )
                
                # Goal status
                goal_status = []
                for i in range(len(self.states)):
                    status = "✓" if self._goal_test(self.states[i], self.goal_states[i]) else "..."
                    goal_status.append(f"Agent {i+1}: {status}")
                
                self.goal_status_text = self.ax.text(
                    0.98, 0.02, '\n'.join(goal_status),
                    transform=self.ax.transAxes, fontsize=12,
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
                )
        
        # Update status text if it exists
        if show_status and hasattr(self, 'timestamp_text') and self.timestamp_text is not None:
            self.timestamp_text.set_text(f"Step: {self.current_step}/{self.max_steps_displayed}")
            
            # Update goal status
            if hasattr(self, 'goal_status_text') and self.goal_status_text is not None:
                goal_status = []
                for i in range(len(self.states)):
                    status = "✓" if self._goal_test(self.states[i], self.goal_states[i]) else "..."
                    goal_status.append(f"Agent {i+1}: {status}")
                self.goal_status_text.set_text('\n'.join(goal_status))
        
        # Create/update the image
        if self.live_display and hasattr(self, 'ax_full_img'):
            # Update existing image for efficiency
            self.ax_full_img.set_data(obs)
        else:
            # Create a new image
            self.ax_full_img = self.ax.imshow(
                obs, cmap=self.cmap, norm=self.norm, 
                animated=True, interpolation='nearest'
            )
            
            # Add legend on first render
            if len(self.ax_imgs) == 0:
                legend_elements = []
                
                # Agent/goal legend entries
                for i in range(min(len(self.states), len(self.agent_colors))):
                    legend_elements.append(
                        mpatches.Patch(color=self.agent_colors[i], label=f'Agent/Goal {i+1}')
                    )
                
                # Path trace legend entries if enabled
                if self.render_trace:
                    for i in range(min(len(self.states), len(self.path_colors))):
                        legend_elements.append(
                            mpatches.Patch(color=self.path_colors[i], label=f'Path {i+1}')
                        )
                
                # Wall legend entry
                legend_elements.append(mpatches.Patch(color='black', label='Wall'))
                
                # Add legend
                self.ax.legend(
                    handles=legend_elements, 
                    loc='upper center', 
                    bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, 
                    shadow=True, 
                    ncol=min(5, len(legend_elements))
                )
        
        # Handle different display modes
        plt.draw()
        if self.live_display:
            # Update the display immediately
            self.fig.canvas.draw()
        else:
            # Store for animation
            self.ax_imgs.append([self.ax_full_img])
        
        # Small pause to update the display
        plt.pause(0.01)
        
        if mode == 'rgb_array':
            # Return RGB array for programmatic use
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
        
        return self.fig
    
    def _get_video(self, interval=100, gif_path=None, mp4_path=None):
        """
        Generate a video animation of the maze exploration.
        
        Args:
            interval: Delay between frames in milliseconds
            gif_path: Path to save GIF file (optional)
            mp4_path: Path to save MP4 file (optional)
            
        Returns:
            Animation object
        """
        if self.live_display:
            print('Warning: Generating animation with live_display=True not supported.')
            return None
        
        if not self.ax_imgs:
            print('Warning: No frames to animate. Make sure render() is called with live_display=False.')
            return None
        
        # Create animation
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)
        
        # Save as GIF if requested
        if gif_path is not None:
            print(f"Saving animation to {gif_path} (this may take a moment)...")
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            anim.save(gif_path, writer='pillow', dpi=self.dpi)
            print(f"Animation saved to {gif_path}")
        
        # Save as MP4 if requested
        if mp4_path is not None:
            print(f"Saving animation to {mp4_path} (this may take a moment)...")
            os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
            # Need ffmpeg for mp4
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=1000/interval, metadata=dict(artist='EnhancedMazeEnv'), bitrate=1800)
                anim.save(mp4_path, writer=writer, dpi=self.dpi)
                print(f"Animation saved to {mp4_path}")
            except Exception as e:
                print(f"Error saving MP4: {e}")
                print("To save as MP4, make sure ffmpeg is installed.")
        
        return anim


class EnhancedSparseMazeEnv(EnhancedMazeEnv, SparseMazeEnv):
    """
    Enhanced Sparse Maze Environment with better visualization.
    Combines the enhanced visualization with the sparse reward structure.
    """
    
    def step(self, action):
        # Override to use the SparseMazeEnv step function with enhanced visualization
        obs, reward, done, info = SparseMazeEnv.step(self, action)
        # Update step counter for visualization
        self.current_step += 1
        return obs, reward, done, info 