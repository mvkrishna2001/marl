## Top-level comments
- `qmix.py` has the MARL algorithm we implemented.
- More details regarding the results can be found in [these slides](https://docs.google.com/presentation/d/1as4J6RhhmO1J0AUniK_rIjEl3sd3wa7g/edit?usp=sharing&ouid=109685225193745201596&rtpof=true&sd=true).
- `RobotLearningProjectReport.pdf` is an extensive report walking through the algorithm and the results in greater detail.

## Installation

1. Create a Conda env that contains Python 3.10

```angular2html
conda create -n marl_venv python=3.10
```

2. Active the env
```angular2html
source activate marl_venv
```

3. Install the requirements
```angular2html
pip install -r requirements.txt
```

4. Install ffmpeg for maze visualizations
```angular2html
sudo apt install ffmpeg
```

## Running the Maze

Use the following command to run the maze solver interface:
```angular2html
python maze_solver.py
```

### Arguments
* `--alg`. Choices :`['astar', 'bfs', 'dfs', 'dijkstra']`
* `--maze`. Choices: `["simple", "random", "block", "u", "t"]`. This gives the choice between a simple empty maze, a random block maze, a random maze, a U-maze, or a multiple T-maze. The specific parameters of each maze can be modified in the `maze_solver.py` file.
* `--num_agents`. An integer that specified the number of agents in the maze. We tested with [1, 2, 3]. 
* `--gif`. This specifies the path to output GIF file


Example of use:
```angular2html
python maze_solver.py --alg astar --maze block --num_agents 2 --gif data/block_maze.gif
```
## Evaluating the mazes 

Use the following command to evaluate on the same set of generated mazes:
```angular2html
python python mazes_evaluation.py
```
## File Structure
```
marl/
├── data/                        # GIF outputs of maze_solver
├── gym_maze/
│   └── envs/
│       ├── generators.py        # Maze generator definitions
│       ├── maze.py              # Gym maze environment configuration
│       ├── node.py              # Node class shared by algorithms
│       └── algorithms/
│           ├── a_star.py
│           ├── bfs.py
│           ├── dfs.py
│           └── dijkstra.py

```


## Notes
The gym maze environment code is taken from https://github.com/rpinsler/gym-maze/tree/master, with minor modifications. 


Open Qs:
1. Size and kind of the maze for evaluating the algorithms
2. Number of agents for a given size of the maze
3. 
