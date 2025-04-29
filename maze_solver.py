import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from IPython.display import HTML

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, RandomBlockMazeGenerator, \
                                     UMazeGenerator, TMazeGenerator, WaterMazeGenerator
from gym_maze.envs.algorithms.a_star import AstarSolver
from gym_maze.envs.algorithms.bfs import BFSSolver
from gym_maze.envs.algorithms.dfs import DFSSolver
from gym_maze.envs.algorithms.dijkstra import DijkstraSolver


def get_solver(name, env, goal, start_state):
    if name == "astar":
        return AstarSolver(env, goal, start_state)
    elif name == "bfs":
        return BFSSolver(env, goal)
    elif name == "dfs":
        return DFSSolver(env, goal)
    elif name == "dijkstra":
        return DijkstraSolver(env, goal, start_state)
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def get_maze(name):
    if name == "simple":
        return RandomBlockMazeGenerator(maze_size=4, obstacle_ratio=0.0)
    elif name == "random":
        return RandomMazeGenerator(width=20, height=15, complexity=.75, density=.75)
    elif name == "block":
        return RandomBlockMazeGenerator(maze_size=30, obstacle_ratio=0.2)
    elif name == "u":
        return UMazeGenerator(len_long_corridor=14, len_short_corridor=4, width=4, wall_size=4)
    elif name == "t":
        return TMazeGenerator(3, [5, 3], [3, 3])
    # elif name == "water":
    #     return WaterMazeGenerator()
    else:
        raise ValueError(f"Unknown maze type: {name}")


def solvemaze(maze, algorithm, num_agents, action_type='VonNeumann', render_trace=False, gif_file='video.gif'):
    env = MazeEnv(maze, action_type=action_type, render_trace=render_trace)
    env.reset(num_agents=num_agents) # Initialize the environment with the specified number of agents

    solvers = []
    for i in range(num_agents):
        solver = get_solver(algorithm, env, env.goal_states[i], start_state=env.states[i])
        if not solver.solvable():
            raise ValueError(f"Maze is not solvable for agent {i}")
        solvers.append(solver)

    # Assumes each solver provides a list of actions
    # Shorten to match the longest sequence by repeating the last action
    all_actions = [solver.get_actions() for solver in solvers]
    max_len = max(len(a) for a in all_actions)
    for i in range(num_agents):
        last_action = all_actions[i][-1] if all_actions[i] else 0
        all_actions[i] += [last_action] * (max_len - len(all_actions[i]))

    # Step through environment
    for t in range(max_len):
        actions = [all_actions[i][t] for i in range(num_agents)]
        env.step(actions)
        env.render()

    return env._get_video(interval=200, gif_path=gif_file).to_html5_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maze Solver")
    parser.add_argument("--alg", type=str, default="astar", choices=["astar", "bfs", "dfs", "dijkstra"],
                        help="Search algorithm to use")
    parser.add_argument("--gif", type=str, default="data/maze.gif", help="Path to output GIF file")
    parser.add_argument("--maze", type=str, default="block",
                    choices=["simple", "random", "block", "u", "t"],
                    help="Maze type to generate")
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents in the maze')

    args = parser.parse_args()

    maze = get_maze(args.maze) 
    video_html = solvemaze(maze, algorithm=args.alg, num_agents=args.num_agents, render_trace=True, gif_file=args.gif)
    print("Maze solved and video saved.")