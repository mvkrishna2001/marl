

import pickle
import time

from gym.core import ObservationWrapper
from gym_maze.envs.generators import MazeGenerator, SimpleMazeGenerator, RandomMazeGenerator, RandomBlockMazeGenerator, \
                                     UMazeGenerator, TMazeGenerator, WaterMazeGenerator
from gym_maze.envs import MazeEnv
from maze_solver import get_solver

def generate_and_save_mazes(filename, num_mazes=10, num_agents=2, maze_size=30, obstacle_ratio=0.2):
    '''Generates random mazes and saves them to a pkl file''' 

    saved_mazes = []

    for _ in range(num_mazes):
        maze_generator = RandomBlockMazeGenerator(maze_size=maze_size, obstacle_ratio=obstacle_ratio)
        maze = maze_generator.get_maze()
        start_states, goal_states = maze_generator.sample_multi_agent_states(num_agents)
        saved_mazes.append((maze_generator, start_states, goal_states))

        # print("Maze shape:", maze.shape)
        # print("Start states:", start_states)
        # print("Goal states:", goal_states)

    with open(filename, 'wb') as f:
        pickle.dump(saved_mazes, f)

    print(f"Saved {len(saved_mazes)} mazes to {filename}!")


def load_saved_mazes(filename):
    ''' Loads the mazes from the pkl file'''
    with open(filename, 'rb') as f:
        saved_mazes = pickle.load(f)
    print(f"Loaded {len(saved_mazes)} mazes from {filename}!")
    return saved_mazes


def evaluate_algorithms_on_mazes(saved_mazes, algorithms, num_agents=2, action_type='VonNeumann', render=False):
    ''' Evaluates Algorithms on Mazes '''
    results = {}

    for alg_name in algorithms:
        print(f"\nEvaluating algorithm: {alg_name}")
        alg_results = []
        alg_times = []  # <=== Track times

        for maze_idx, (maze_gen, start_states, goal_states) in enumerate(saved_mazes):
            print(f"  Solving maze {maze_idx}")

            env = MazeEnv(maze_gen, action_type=action_type, render_trace=render)
            env.reset(num_agents=num_agents)
            
            env.states = list(start_states)
            env.goal_states = list(goal_states)

            solvers = []
            solvable = True

            for i in range(num_agents):
                solver = get_solver(alg_name, env, goal_states[i], start_state=start_states[i])
                if not solver.solvable():
                    print(f"    Maze {maze_idx}: Agent {i} not solvable!")
                    solvable = False
                    break
                solvers.append(solver)

            if solvable:
                all_actions = [solver.get_actions() for solver in solvers]
                max_len = max(len(a) for a in all_actions)

                for i in range(num_agents):
                    last_action = all_actions[i][-1] if all_actions[i] else 0
                    all_actions[i] += [last_action] * (max_len - len(all_actions[i]))

                # === Time this maze solving ===
                start_time = time.time()

                for t in range(max_len):
                    actions = [all_actions[i][t] for i in range(num_agents)]
                    env.step(actions)
                    if render:
                        env.render()

                elapsed_time = (time.time() - start_time) * 1000  # milliseconds

                alg_results.append(max_len)
                alg_times.append(elapsed_time)

        results[alg_name] = {
            'steps': alg_results,
            'times': alg_times
        }

    return results


def main():
    maze_save_file = 'saved_mazes.pkl'
    num_mazes = 10
    num_agents = 2
    maze_size = 500
    obstacle_ratio = 0.3
    algorithms = ['astar', 'dijkstra', 'bfs', 'dfs']  # List your algorithms here

    generate_and_save_mazes(maze_save_file, num_mazes=num_mazes, maze_size=maze_size, num_agents=num_agents, obstacle_ratio=obstacle_ratio)

    saved_mazes = load_saved_mazes(maze_save_file)

    results = evaluate_algorithms_on_mazes(saved_mazes, algorithms, num_agents=num_agents, render=False)

    # === Print results nicely ===
    print("\n=== Evaluation Results ===")
    for alg_name, data in results.items():
        avg_steps = sum(data['steps']) / len(data['steps'])
        avg_time_ms = sum(data['times']) / len(data['times'])
        print(f"{alg_name}: Average steps = {avg_steps:.2f}, Average time = {avg_time_ms:.4f} ms over {len(data['steps'])} mazes")


if __name__ == "__main__":
    main()