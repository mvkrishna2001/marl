import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from past.utils import old_div

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import torch
import os 

''' Adapted from https://github.com/rpinsler/gym-maze/tree/master''' 

class MazeEnv(gym.Env):
    """Configurable environment for maze. """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 maze_generator,
                 pob_size=1,
                 action_type='VonNeumann',
                 obs_type='full',
                 live_display=False,
                 render_trace=False):
        """Initialize the maze. DType: list"""
        # Random seed with internal gym seeding
        self.seed()

        # Maze: 0: free space, 1: wall
        self.maze_generator = maze_generator
        self.maze = np.array(self.maze_generator.get_maze())
        self.maze_size = self.maze.shape
        # self.init_state, self.goal_states = self.maze_generator.sample_state()
        self.init_states = None  # Will be set in reset()
        self.goal_states = None  # Will be set in reset()
        
        # Initialize goal reached status for each agent
        self.goal_reached = []  # Will be set in reset()

        self.render_trace = render_trace
        self.traces = []
        self.VISITED = 5 
        self.action_type = action_type
        self.obs_type = obs_type

        # Track explored cells
        self.visited_cells = set()
        self.total_free_cells = np.sum(self.maze == 0)  # Count free cells (non-walls)
        
        # Advanced tracking metrics
        self.state_visit_counts = {}  # Tracks how many times each state has been visited
        self.total_revisits = 0       # Total count of revisits to already visited states
        self.total_steps = 0          # Total steps taken by all agents
        self.steps_to_solve = 0       # Steps taken until maze is solved
        self.maze_solved = False      # Flag to indicate if maze has been solved
        
        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

        self.states = None   # Will be set in reset()
        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        if self.action_type == 'VonNeumann':  # Von Neumann neighborhood
            self.num_actions = 4
        elif action_type == 'Moore':  # Moore neighborhood
            self.num_actions = 8
        else:
            raise TypeError('Action type must be either \'VonNeumann\' or \'Moore\'')
        self.action_space = spaces.Discrete(self.num_actions)
        self.all_actions = list(range(self.action_space.n))

        # Size of the partial observable window
        self.pob_size = pob_size

        # Observation space
        low_obs = 0  # Lowest integer in observation
        high_obs = 6  # Highest integer in observation
        if self.obs_type == 'full':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=self.maze_size, )
            # dtype=np.float32)
        elif self.obs_type == 'partial':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=(self.pob_size * 2 + 1, self.pob_size * 2 + 1), )
            # dtype=np.float32)
        else:
            raise TypeError('Observation type must be either \'full\' or \'partial\'')

        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])

        self.bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

        self.ax_imgs = []  # For generating videos

        self.EMPTY = 0
        self.WALL = 1
        self.AGENT = 2
        self.GOAL = 3

    def step(self, actions):
        rewards = []
        dones = []
        infos = []
        new_states = []
        
        # Increment total steps
        self.total_steps += 1
        if not self.maze_solved:
            self.steps_to_solve += 1

        for i, action in enumerate(actions):
            if self.goal_reached[i]:
                # Agent already reached goal, no movement
                new_states.append(self.states[i])
                rewards.append(0)
                dones.append(True)
                infos.append({})
                continue
            
            old_state = self.states[i]
            new_state = self._next_state(old_state, action)

            # Track visited cells
            self.visited_cells.add(tuple(new_state))
            
            # Update state visit counts
            pos_tuple = tuple(new_state)
            if pos_tuple in self.state_visit_counts:
                self.state_visit_counts[pos_tuple] += 1
                self.total_revisits += 1
            else:
                self.state_visit_counts[pos_tuple] = 1
                
            self.traces[i].append(new_state)
            new_states.append(new_state)

            if self._goal_test(new_state, self.goal_states[i]):
                self.goal_reached[i] = True
                # If any agent reaches the goal for the first time, mark maze as solved
                if not self.maze_solved:
                    self.maze_solved = True
                reward = +1 * self.maze.size
                done = True
            elif new_state == old_state:
                reward = -1
                done = False
            else:
                reward = -0.01 * self.num_actions
                done = False

        # Calculate exploration percentage
        exploration_percentage = (len(self.visited_cells) / self.total_free_cells) * 100

        # Additional info
        info = {'exploration_percentage': exploration_percentage}
        rewards.append(reward)
        dones.append(done)
        infos.append({})  # Customize if needed

        # Calculate exploration percentage
        exploration_percentage = (len(self.visited_cells) / self.total_free_cells) * 100

        # Create unified info dictionary
        info = {
            'exploration_percentage': exploration_percentage,
            'total_revisits': self.total_revisits,
            'total_steps': self.total_steps,
            'steps_to_solve': self.steps_to_solve if self.maze_solved else -1,
            'maze_solved': self.maze_solved,
            'unique_states_visited': len(self.state_visit_counts),
            'global_state': self._get_global_state()  # Add global state for QMIX
        }

        self.states = new_states
        return [self._get_obs(i) for i in range(len(self.states))], rewards, dones, False, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, num_agents=1, seed=None, options=None):
        # Set seed if provided
        if seed is not None:
            self.seed(seed)

        # Reset maze
        self.maze = np.array(self.maze_generator.get_maze())

        # Sample initial and goal states for each agent
        self.init_states, self.goal_states = self.maze_generator.sample_multi_agent_states(num_agents=num_agents)

        # Set the current position of each agent
        self.states = self.init_states.copy()
        
        # Initialize goal reached status for each agent
        self.goal_reached = [False] * len(self.states)  # Reset goal status for each agent

        # Initialize traces and video frames
        self.traces = [[s] for s in self.init_states]  # One trace per agent

        # Reset visited cells tracking
        self.visited_cells = set()
        for state in self.states:
            self.visited_cells.add(tuple(state))
        self.total_free_cells = np.sum(self.maze == 0)
        
        # Reset tracking metrics
        self.state_visit_counts = {}
        for pos in self.states:
            pos_tuple = tuple(pos)
            self.state_visit_counts[pos_tuple] = 1
        
        self.total_revisits = 0
        self.total_steps = 0
        self.steps_to_solve = 0
        self.maze_solved = False

        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []
        
        # Create info dictionary with metrics
        info = {
            'total_revisits': self.total_revisits,
            'total_steps': self.total_steps,
            'steps_to_solve': self.steps_to_solve,
            'maze_solved': self.maze_solved,
            'unique_states_visited': len(self.state_visit_counts),
            'global_state': self._get_global_state()  # Add global state for QMIX
        }

        # Return initial observations for each agent
        return [self._get_obs(i) for i in range(len(self.states))], info
        
    def _get_global_state(self):
        """
        Create a global state representation for QMIX. In this implementation,
        the global state includes the flattened maze with agent and goal positions.
        
        Returns:
            global_state: Flattened global state (numpy array)
        """
        # Get the full observation which includes maze, agents, and goals
        full_obs = self._get_full_obs()
        
        # For QMIX, we need a flat representation
        return full_obs.flatten().astype(np.float32)

    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return

        obs = self._get_full_obs()
        # partial_obs = [self._get_partial_obs(self.pob_size, pos=self.states[i]) for i in range(len(self.states))]

        # For rendering traces: Only for visualization, does not affect the observation data
        if self.render_trace:
            # Iterate through each agent's trace and mark their path on the maze (observation)
            for i, trace in enumerate(self.traces):  # Loop through each agent's trace
                for y, x in trace[:-1]:  # Skip the last position to avoid re-rendering the final position
                    obs[y, x] = self.VISITED  # Mark the agent's path as visited

        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            # self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
            self.fig, self.ax_full = plt.subplots()
        self.ax_full.axis('off')
        # self.ax_partial.axis('off')

        self.fig.show()
        if self.live_display:
            # Only create the image the first time
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            # if not hasattr(self, 'ax_partial_img'):
            #     self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
            # Update the image data for efficient live video
            self.ax_full_img.set_data(obs)
            # self.ax_partial_img.set_data(partial_obs)
        else:
            # Create a new image each time to allow an animation to be created
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            # self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)

        plt.draw()

        if self.live_display:
            # Update the figure display immediately
            self.fig.canvas.draw()
        else:
            # Put in AxesImage buffer for video generation
            # self.ax_imgs.append([self.ax_full_img, self.ax_partial_img])  # List of axes to update figure frame
            self.ax_imgs.append([self.ax_full_img])  # Adjusted to avoid partial image

            self.fig.set_dpi(100)

        plt.pause(.1)
        return self.fig

    def _goal_test(self, state, goal):
        """Return True if current state is a goal state."""
        return tuple(state) == tuple(goal)

    def _next_state(self, state, action):
        """Return the next state from a given state by taking a given action."""

        # Transition table to define movement for each action
        if self.action_type == 'VonNeumann':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1]}
        elif self.action_type == 'Moore':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1],
                           4: [-1, +1], 5: [+1, +1], 6: [-1, -1], 7: [+1, -1]}

        new_state = [state[0] + transitions[action][0], state[1] + transitions[action][1]]
        if self.maze[new_state[0]][new_state[1]] == 1:  # Hit wall, stay there
            return state
        else:  # Valid move for 0, 2, 3, 4
            return new_state

    def _get_obs(self, agent_idx):
        state = self.states[agent_idx]
        if self.obs_type == 'full':
            return self._get_full_obs().flatten()  # same for all agents
        elif self.obs_type == 'partial':
            return self._get_partial_obs(self.pob_size, state).flatten()

    def _get_full_obs(self):
        """Return a 2D array representation of maze with all agents and goals."""
        obs = np.array(self.maze)
        # The colours for the goal and end positions are the same 
        # Set goal positions with unique values per agent
        for i, goal in enumerate(self.goal_states):
            obs[goal[0]][goal[1]] = 2 + i  

        # Set agent positions (after goals, to avoid hiding agents under goals)
        for i, state in enumerate(self.states):
            obs[state[0]][state[1]] = 2 + i 

        return obs

    def _get_partial_obs(self, size=1, pos=None):
        """Get partial observable window according to Moore neighborhood"""
        # Get maze with indicated location of current position and goal positions
        if pos is None:
            raise ValueError("Agent position must be provided for partial observation.")

        # Get full maze with current agent and goal markings
        maze = self._get_full_obs()
        pos = np.array(pos)

        under_offset = np.min(pos - size)
        over_offset = np.min(len(maze) - (pos + size + 1))
        offset = np.min([under_offset, over_offset])

        if offset < 0:  # Need padding
            maze = np.pad(maze, np.abs(offset), 'constant', constant_values=1)
            pos += np.abs(offset)

        return maze[pos[0] - size: pos[0] + size + 1, pos[1] - size: pos[1] + size + 1]

    def _get_video(self, interval=400, gif_path=None):
        if self.live_display:
            # TODO: Find a way to create animations without slowing down the live display
            print('Warning: Generating an Animation when live_display=True not yet supported.')
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)

        if gif_path is not None:
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            anim.save(gif_path, writer='pillow')
        return anim

    def render_learning(self, policy, qf, vmin, vmax):
        # adapted from https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/GridWorld.py
        obs = self._get_full_obs()
        ROWS, COLS = self.maze_size
        MIN_RETURN = None
        MAX_RETURN = None
        SHIFT = .1
        cmap_actions = colors.ListedColormap(['.5', 'k'], 'Actions')
        V = np.zeros((ROWS, COLS))

        if not hasattr(self, 'fig_value'):
            plt.figure("Value Function")
            self.fig_value = plt.imshow(np.array(self.maze), cmap=self.cmap, norm=self.norm, animated=False)
            self.im_value = plt.imshow(np.zeros_like(obs), cmap=plt.cm.RdYlGn, vmin=vmin, vmax=vmax, alpha=0.7)
            plt.colorbar()
            plt.xticks(np.arange(COLS), fontsize=12)
            plt.yticks(np.arange(ROWS), fontsize=12)

        if not hasattr(self, 'fig_policy'):
            plt.figure("Policy")
            self.fig_policy = plt.imshow(obs, cmap=self.cmap, norm=self.norm, animated=False)

            plt.xticks(np.arange(COLS), fontsize=12)
            plt.yticks(np.arange(ROWS), fontsize=12)
            # Create quivers for each action. 4 in total
            X = np.arange(ROWS) - SHIFT
            Y = np.arange(COLS)
            X, Y = np.meshgrid(X, Y)
            DX = DY = np.ones(X.shape)
            C = np.zeros(X.shape)
            C[0, 0] = 1  # Making sure C has both 0 and 1
            # length of arrow/width of bax. Less then 0.5 because each arrow is
            # offset, 0.4 looks nice but could be better/auto generated
            arrow_ratio = 0.4
            Max_Ratio_ArrowHead_to_ArrowLength = 0.25
            ARROW_WIDTH = 0.5 * Max_Ratio_ArrowHead_to_ArrowLength / 5.0
            self.upArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='y', cmap=cmap_actions, width=-1 * ARROW_WIDTH,
                scale_units="height", scale=old_div(ROWS, arrow_ratio))
            self.upArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(ROWS) + SHIFT
            Y = np.arange(COLS)
            X, Y = np.meshgrid(X, Y)
            self.downArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='y', cmap=cmap_actions, width=-1 * ARROW_WIDTH,
                scale_units="height", scale=old_div(ROWS, arrow_ratio))
            self.downArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(ROWS)
            Y = np.arange(COLS) - SHIFT
            X, Y = np.meshgrid(X, Y)
            self.leftArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='x', cmap=cmap_actions, width=ARROW_WIDTH,
                scale_units="width", scale=old_div(COLS, arrow_ratio))
            self.leftArrows_fig.set_clim(vmin=0, vmax=1)
            X = np.arange(ROWS)
            Y = np.arange(COLS) + SHIFT
            X, Y = np.meshgrid(X, Y)
            self.rightArrows_fig = plt.quiver(
                Y, X, DY, DX, C,
                units='x', cmap=cmap_actions, width=ARROW_WIDTH,
                scale_units="width", scale=old_div(COLS, arrow_ratio))
            self.rightArrows_fig.set_clim(vmin=0, vmax=1)
            plt.show()

        plt.figure("Policy")
        # Boolean 3 dimensional array. The third array highlights the action.
        # Thie mask is used to see in which cells what actions should exist
        Mask = np.ones((COLS, ROWS, self.num_actions), dtype='bool')
        arrowSize = np.zeros((COLS, ROWS, self.num_actions), dtype='float')
        # 0 = suboptimal action, 1 = optimal action
        arrowColors = np.zeros((COLS, ROWS, self.num_actions), dtype='uint8')
        for r in range(ROWS):
            for c in range(COLS):
                if obs[r, c] != self.WALL:
                    _obs = np.array(self.maze)
                    for goal in self.goal_states: _obs[goal[0], goal[1]] = self.GOAL
                    _obs[r, c] = self.AGENT
                    act, _ = policy.get_action(_obs.reshape(1, -1))
                    Qs = [qf(torch.from_numpy(_obs).view(1, -1),
                             torch.from_numpy(a).view(1, -1))
                          for a in torch.eye(self.num_actions)]
                    V[r, c] = torch.max(torch.cat(Qs)).item()
                    Mask[c, r, :] = False
                    arrowColors[c, r, act] = 1
                    arrowSize[c, r, :] = policy.get_actions(obs.reshape(1, -1))

        # Show Policy Up Arrows
        DX = arrowSize[:, :, 2]
        DY = np.zeros((ROWS, COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 2])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 2])
        C = np.ma.masked_array(arrowColors[:, :, 2], mask=Mask[:, :, 2])
        self.upArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Down Arrows
        DX = -arrowSize[:, :, 3]
        DY = np.zeros((ROWS, COLS))
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 3])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 3])
        C = np.ma.masked_array(arrowColors[:, :, 3], mask=Mask[:, :, 3])
        self.downArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Left Arrows
        DX = np.zeros((ROWS, COLS))
        DY = -arrowSize[:, :, 0]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 0])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 0])
        C = np.ma.masked_array(arrowColors[:, :, 0], mask=Mask[:, :, 0])
        self.leftArrows_fig.set_UVC(DY, DX, C)
        # Show Policy Right Arrows
        DX = np.zeros((ROWS, COLS))
        DY = arrowSize[:, :, 1]
        DX = np.ma.masked_array(DX, mask=Mask[:, :, 1])
        DY = np.ma.masked_array(DY, mask=Mask[:, :, 1])
        C = np.ma.masked_array(arrowColors[:, :, 1], mask=Mask[:, :, 1])
        self.rightArrows_fig.set_UVC(DY, DX, C)
        plt.draw()
        plt.pause(.1)

        plt.figure("Value Function")
        self.im_value.set_data(V)
        plt.draw()
        plt.pause(.1)


class SparseMazeEnv(MazeEnv):
    def step(self, action):
        obs, reward, done, info = super()._step(action)

        # Indicator reward function
        if reward != 1:
            reward = 0

        return obs, reward, done, info
